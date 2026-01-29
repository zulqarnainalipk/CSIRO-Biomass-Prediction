"""
Training script for CSIRO Biomass Prediction.

This script trains a DINO v3 based model for pasture biomass estimation
using 3-fold stratified group cross-validation.

Author: Zulqarnain Ali
"""

import os
import gc
import math
import random
import warnings
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
import timm

from src.config import train_config
from src.dataset import BiomassDataset
from src.model import BiomassModel
from src.loss import WeightedBiomassLoss
from src.metrics import weighted_r2_score
from src.transforms import get_train_transforms, get_val_transforms


# Suppress warnings
warnings.filterwarnings('ignore')

# Optimize PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

print("=" * 60)
print("CSIRO Biomass Prediction - Training Script")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"timm version: {timm.__version__}")


def seed_everything(seed=train_config.seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_train_data():
    """Load and prepare training data."""
    df = pd.read_csv(train_config.get_train_csv())
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Pivot to wide format
    df_wide = df.pivot_table(
        index=['image_id', 'image_path'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    # Handle missing columns
    for col in train_config.target_cols:
        if col not in df_wide.columns:
            df_wide[col] = 0.0

    # Create stratification bins
    df_wide['total_bin'] = pd.qcut(
        df_wide['Dry_Total_g'],
        q=5,
        labels=False,
        duplicates='drop'
    )

    # Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(
        n_splits=train_config.n_folds,
        shuffle=True,
        random_state=train_config.seed
    )
    df_wide['fold'] = -1

    for fold, (_, val_idx) in enumerate(sgkf.split(
        df_wide,
        df_wide['total_bin'],
        groups=df_wide['image_id']
    )):
        df_wide.loc[val_idx, 'fold'] = fold

    print(f"Loaded {len(df_wide)} training images")
    print(f"Fold distribution:\n{df_wide['fold'].value_counts().sort_index()}")

    return df_wide


def load_test_data():
    """Load test data."""
    df = pd.read_csv(train_config.get_test_csv())
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_unique = df.drop_duplicates('image_id')[['image_id', 'image_path']].reset_index(drop=True)
    print(f"Loaded {len(df_unique)} test images")
    return df_unique


def build_optimizer(model):
    """Build optimizer with differential learning rates."""
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    return optim.AdamW([
        {'params': backbone_params, 'lr': train_config.lr_backbone},
        {'params': head_params, 'lr': train_config.lr_head}
    ], weight_decay=train_config.weight_decay)


def build_scheduler(optimizer, total_steps):
    """Build learning rate scheduler."""
    def lr_lambda(step):
        warmup_steps = train_config.warmup_epochs * (total_steps // train_config.epochs)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc='Training')
    scaler = GradScaler()

    for i, (left, right, labels) in enumerate(pbar):
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            preds = model(left, right)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    all_preds = []
    all_labels = []

    for left, right, labels in tqdm(loader, desc='Validating'):
        left = left.to(device)
        right = right.to(device)

        with autocast():
            preds = model(left, right)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    weighted_r2, per_target_r2 = weighted_r2_score(all_labels, all_preds)
    return weighted_r2, per_target_r2


def train_fold(fold, train_df, model_dir):
    """Train a single fold."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING FOLD {fold}")
    print(f"{'=' * 60}")

    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create datasets
    train_dataset = BiomassDataset(
        train_data,
        train_config.get_train_image_dir(),
        get_train_transforms(train_config.img_height, train_config.img_width)
    )
    val_dataset = BiomassDataset(
        val_data,
        train_config.get_train_image_dir(),
        get_val_transforms(train_config.img_height, train_config.img_width)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True
    )

    # Create model
    model = BiomassModel(
        train_config.model_name,
        pretrained=True
    ).to(train_config.device)

    criterion = WeightedBiomassLoss()
    optimizer = build_optimizer(model)
    total_steps = len(train_loader) * train_config.epochs
    scheduler = build_scheduler(optimizer, total_steps)

    # Training loop
    best_r2 = -float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    epoch_pbar = tqdm(range(train_config.epochs), desc=f'Fold {fold} Epochs')

    for epoch in epoch_pbar:
        print(f"\nEpoch {epoch + 1}/{train_config.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, train_config.device
        )
        val_r2, per_r2 = validate(model, val_loader, train_config.device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val R2: {val_r2:.4f}")
        print(f"Per-target: Green={per_r2[0]:.3f}, Dead={per_r2[1]:.3f}, "
              f"Clover={per_r2[2]:.3f}, GDM={per_r2[3]:.3f}, Total={per_r2[4]:.3f}")

        epoch_pbar.set_postfix({
            'loss': f'{train_loss:.4f}',
            'val_r2': f'{val_r2:.4f}',
            'best_r2': f'{best_r2:.4f}'
        })

        # Check for improvement
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, f'fold{fold}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (R2={best_r2:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= train_config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print(f"\nFold {fold} Best: R2={best_r2:.4f} at epoch {best_epoch}")

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader, criterion
    gc.collect()
    torch.cuda.empty_cache()

    return best_r2


def main():
    """Main training function."""
    # Set seeds
    seed_everything()

    # Create output directories
    os.makedirs(train_config.model_dir, exist_ok=True)
    os.makedirs(train_config.output_dir, exist_ok=True)

    # Print configuration
    print(f"\n{'=' * 60}")
    print("CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Device: {train_config.device}")
    print(f"Model: {train_config.model_name}")
    print(f"Image Size: {train_config.img_height}x{train_config.img_width}")
    print(f"Batch Size: {train_config.batch_size}")
    print(f"Epochs: {train_config.epochs}")
    print(f"Folds: {train_config.n_folds}")

    # Load data
    print(f"\n{'=' * 60}")
    print("STEP 1: Loading Data")
    print(f"{'=' * 60}")
    train_df = load_train_data()
    test_df = load_test_data()

    # Train folds
    print(f"\n{'=' * 60}")
    print("STEP 2: Training Models")
    print(f"{'=' * 60}")

    fold_scores = []
    fold_pbar = tqdm(train_config.folds_to_train, desc='Training Folds')

    for fold in fold_pbar:
        fold_pbar.set_description(f'Training Fold {fold}')
        score = train_fold(fold, train_df, train_config.model_dir)
        fold_scores.append(score)
        fold_pbar.set_postfix({
            'current_r2': f'{score:.4f}',
            'mean_r2': f'{np.mean(fold_scores):.4f}'
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Fold scores: {fold_scores}")
    print(f"Mean CV R2: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # Save summary
    summary = {
        'model': train_config.model_name,
        'folds': train_config.n_folds,
        'epochs': train_config.epochs,
        'batch_size': train_config.batch_size,
        'image_size': f'{train_config.img_height}x{train_config.img_width}',
        'fold_scores': [float(s) for s in fold_scores],
        'mean_cv': float(np.mean(fold_scores)),
        'std_cv': float(np.std(fold_scores))
    }

    import json
    with open(f'{train_config.output_dir}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nModels saved to: {train_config.model_dir}/")
    print(f"  - fold0_best.pth")
    print(f"  - fold1_best.pth")
    print(f"  - fold2_best.pth")


if __name__ == "__main__":
    main()

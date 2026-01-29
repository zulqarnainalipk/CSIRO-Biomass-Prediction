"""
Utility functions for CSIRO Biomass Prediction.

Author: Zulqarnain Ali
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_image(image, patch_size=520, overlap=16):
    """
    Split image into patches with overlap.

    Args:
        image: Input image (H, W, C)
        patch_size: Size of each patch
        overlap: Overlap between patches

    Returns:
        List of image patches
    """
    h, w, c = image.shape
    stride = patch_size - overlap
    patches = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size)
            x1 = max(0, x2 - patch_size)
            patches.append(image[y1:y2, x1:x2, :])

    return patches


def build_optimizer(model, lr_backbone=1e-5, lr_head=5e-4, weight_decay=1e-2):
    """
    Build optimizer with differential learning rates.

    Args:
        model: PyTorch model
        lr_backbone: Learning rate for backbone
        lr_head: Learning rate for head layers
        weight_decay: Weight decay

    Returns:
        Configured optimizer
    """
    import torch.optim as optim

    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    return optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=weight_decay)


def build_scheduler(optimizer, total_steps, warmup_epochs=6, epochs=180):
    """
    Build learning rate scheduler with warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        warmup_epochs: Number of warmup epochs
        epochs: Total number of epochs

    Returns:
        LambdaLR scheduler
    """
    import math

    def lr_lambda(step):
        warmup_steps = warmup_epochs * (total_steps // epochs)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def load_train_data(csv_path, target_cols):
    """
    Load and process training data.

    Args:
        csv_path: Path to training CSV
        target_cols: List of target column names

    Returns:
        Wide-format DataFrame
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    df = pd.read_csv(csv_path)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Pivot to wide format
    df_wide = df.pivot_table(
        index=['image_id', 'image_path'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    # Handle missing columns
    for col in target_cols:
        if col not in df_wide.columns:
            df_wide[col] = 0.0

    # Create stratification bins
    df_wide['total_bin'] = pd.qcut(
        df_wide['Dry_Total_g'],
        q=5,
        labels=False,
        duplicates='drop'
    )

    return df_wide


def load_test_data(csv_path):
    """
    Load test data.

    Args:
        csv_path: Path to test CSV

    Returns:
        DataFrame with unique images
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_unique = df.drop_duplicates('image_id')[['image_id', 'image_path']].reset_index(drop=True)

    return df_unique

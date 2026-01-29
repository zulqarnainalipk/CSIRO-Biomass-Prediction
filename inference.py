"""
Inference script for CSIRO Biomass Prediction.

This script runs inference using the trained DINO v3 models and
SigLIP + GBDT ensemble to generate predictions.

Author: Zulqarnain Ali
"""

import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

import timm

from src.config import inference_config
from src.dataset import TestDataset, collate_fn
from src.model import BiomassModel, LocalMambaBlock
from src.utils import split_image, seed_everything


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60)
print("CSIRO Biomass Prediction - Inference Script")
print("=" * 60)
print(f"Device: {inference_config.device}")
print(f"Model: {inference_config.model_name}")
print(f"Models Dir: {inference_config.models_dir}")


def predict_dino(model, loader, device):
    """Run DINO v3 inference."""
    model.eval()
    preds_all = []

    for lefts, rights, _ in tqdm(loader, desc="DINO Inference"):
        lefts = lefts.to(device)
        rights = rights.to(device)

        with autocast():
            pred = model(lefts, rights)

        preds_all.append(pred.cpu().numpy())

    return np.vstack(preds_all)


def compute_siglip_embeddings(model_path, df, img_dir):
    """Compute SigLIP embeddings for images."""
    print(f"Computing SigLIP embeddings for {len(df)} images...")

    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(inference_config.device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img_path = Path(img_dir) / row['image_path']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patches = split_image(img)
            images = [Image.fromarray(p) for p in patches]

            inputs = processor(images=images, return_tensors="pt").to(inference_config.device)

            with torch.no_grad():
                features = model.get_image_features(**inputs)

            embeddings.append(features.mean(dim=0).cpu().numpy())

        except Exception as e:
            print(f"Error: {e}")
            embeddings.append(np.zeros(1152))

    del model
    torch.cuda.empty_cache()

    return np.stack(embeddings)


def generate_semantic_features(embeddings, model_path):
    """Generate semantic features from embeddings."""
    model = AutoModel.from_pretrained(model_path).to(inference_config.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    concepts = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"]
    }

    concept_vectors = {}

    with torch.no_grad():
        for name, prompts in concepts.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(inference_config.device)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)

    img_tensor = torch.tensor(embeddings, dtype=torch.float32).to(inference_config.device)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)

    scores = {}

    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()

    df_scores = pd.DataFrame(scores)
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)

    del model
    torch.cuda.empty_cache()

    return df_scores.values


class SupervisedEmbeddingEngine:
    """Engine for supervised feature embedding."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.80, random_state=42)
        self.pls = PLSRegression(n_components=8, scale=False)
        self.gmm = GaussianMixture(n_components=6, covariance_type='diag', random_state=42)
        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        """Fit the engine."""
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)

        if y is not None:
            self.pls.fit(X_scaled, y)
            self.pls_fitted_ = True

        return self

    def transform(self, X, X_semantic=None):
        """Transform features."""
        X_scaled = self.scaler.transform(X)
        feats = [self.pca.transform(X_scaled)]

        if self.pls_fitted_:
            feats.append(self.pls.transform(X_scaled))

        feats.append(self.gmm.predict_proba(X_scaled))

        if X_semantic is not None:
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            feats.append(sem_norm)

        return np.hstack(feats)


def train_gbdt_cv(model_cls, params, train_data, test_data, sem_tr, sem_te, emb_cols):
    """Train GBDT model with cross-validation."""
    target_max_arr = np.array([inference_config.target_max[t] for t in inference_config.target_names])
    y_pred_test = np.zeros([len(test_data), len(inference_config.target_names)])
    n_splits = int(train_data['fold'].nunique())

    X_train = train_data[emb_cols].values.astype(np.float32)
    X_test = test_data[emb_cols].values.astype(np.float32)
    y_train = train_data[inference_config.target_names].values.astype(np.float32)

    for fold in range(n_splits):
        train_mask = train_data['fold'] != fold
        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask] / target_max_arr
        sem_tr_fold = sem_tr[train_mask]

        eng = SupervisedEmbeddingEngine()
        eng.fit(X_tr, y=y_tr, X_semantic=sem_tr_fold)

        x_tr_eng = eng.transform(X_tr, X_semantic=sem_tr_fold)
        x_te_eng = eng.transform(X_test, X_semantic=sem_te)

        for k, target in enumerate(inference_config.target_names):
            if target == 'Dry_Clover_g':
                continue
            model = model_cls(**params)
            model.fit(x_tr_eng, y_tr[:, k])
            y_pred_test[:, k] += model.predict(x_te_eng) * target_max_arr[k]

    return y_pred_test / n_splits


def main():
    """Main inference function."""
    # Set seeds
    seed_everything(inference_config.seed)

    print("\n[1/6] Loading test data...")
    test_df_raw = pd.read_csv(inference_config.data_path / 'test.csv')
    test_wide = test_df_raw[["image_path"]].drop_duplicates().reset_index(drop=True)
    print(f"Test images: {len(test_wide)}")

    # DINO v3 Inference
    print("\n[2/6] Running DINO HUGE inference...")

    test_dataset = TestDataset(
        test_wide,
        inference_config.data_path,
        inference_config.img_height,
        inference_config.img_width
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    all_fold_preds = []

    for fold in range(inference_config.n_folds):
        model_path = Path(inference_config.models_dir) / f'fold{fold}_best.pth'

        if not model_path.exists():
            print(f"Fold {fold} not found, skipping...")
            continue

        print(f"Loading fold {fold}...")
        model = BiomassModel(inference_config.model_name, pretrained=False).to(inference_config.device)
        state_dict = torch.load(model_path, map_location=inference_config.device)

        # Handle DataParallel checkpoints
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        fold_preds = predict_dino(model, test_loader, inference_config.device)
        all_fold_preds.append(fold_preds)

        del model, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    dino_preds = np.mean(all_fold_preds, axis=0)
    print(f"DINO predictions shape: {dino_preds.shape}")

    # Post-process DINO predictions
    dino_df = test_wide.copy()
    dino_df['Dry_Green_g'] = dino_preds[:, 0]
    dino_df['Dry_Dead_g'] = dino_preds[:, 1]
    dino_df['Dry_Clover_g'] = dino_preds[:, 2] * 0.8

    for i in range(len(dino_df)):
        if dino_df.loc[i, 'Dry_Dead_g'] > 20:
            dino_df.loc[i, 'Dry_Dead_g'] *= 1.1
        elif dino_df.loc[i, 'Dry_Dead_g'] < 10:
            dino_df.loc[i, 'Dry_Dead_g'] *= 0.9

    dino_df['GDM_g'] = dino_df['Dry_Green_g'] + dino_df['Dry_Clover_g']
    dino_df['Dry_Total_g'] = dino_df['GDM_g'] + dino_df['Dry_Dead_g']

    # SigLIP Inference
    print("\n[3/6] Running SigLIP inference...")

    train_split = pd.read_csv(inference_config.split_path)
    cols_keep = [c for c in train_split.columns if not c.startswith('emb')]
    train_split = train_split[cols_keep]

    if not str(train_split['image_path'].iloc[0]).startswith('/'):
        train_split['image_path'] = train_split['image_path'].apply(
            lambda p: str(inference_config.data_path / 'train' / Path(p).name)
        )

    test_siglip = test_wide.copy()
    test_siglip['image_path'] = test_siglip['image_path'].apply(
        lambda p: str(inference_config.data_path / p)
    )

    print("Computing train embeddings...")
    train_emb = compute_siglip_embeddings(inference_config.siglip_path, train_split, inference_config.data_path)

    print("Computing test embeddings...")
    test_emb = compute_siglip_embeddings(inference_config.siglip_path, test_siglip, inference_config.data_path)

    emb_cols = [f"emb{i}" for i in range(train_emb.shape[1])]
    train_feat = pd.concat([train_split, pd.DataFrame(train_emb, columns=emb_cols)], axis=1)
    test_feat = pd.concat([test_siglip.reset_index(drop=True), pd.DataFrame(test_emb, columns=emb_cols)], axis=1)

    print("Generating semantic features...")
    all_emb = np.vstack([train_emb, test_emb])
    all_sem = generate_semantic_features(all_emb, inference_config.siglip_path)
    sem_train = all_sem[:len(train_split)]
    sem_test = all_sem[len(train_split):]

    # Train GBDT models
    print("\n[4/6] Training GBDT models...")

    params_hist = {'max_iter': 300, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42}
    params_gb = {'n_estimators': 1354, 'learning_rate': 0.01, 'max_depth': 3, 'random_state': 42}
    params_cat = {'iterations': 1900, 'learning_rate': 0.045, 'depth': 4, 'verbose': 0, 'random_state': 42, 'allow_writing_files': False}
    params_lgbm = {'n_estimators': 807, 'learning_rate': 0.014, 'num_leaves': 48, 'verbose': -1, 'random_state': 42}

    print("HistGB...")
    pred_hist = train_gbdt_cv(HistGradientBoostingRegressor, params_hist, train_feat, test_feat, sem_train, sem_test, emb_cols)

    print("GB...")
    pred_gb = train_gbdt_cv(GradientBoostingRegressor, params_gb, train_feat, test_feat, sem_train, sem_test, emb_cols)

    print("CatBoost...")
    pred_cat = train_gbdt_cv(CatBoostRegressor, params_cat, train_feat, test_feat, sem_train, sem_test, emb_cols)

    print("LightGBM...")
    pred_lgbm = train_gbdt_cv(LGBMRegressor, params_lgbm, train_feat, test_feat, sem_train, sem_test, emb_cols)

    siglip_pred = (pred_hist + pred_gb + pred_cat + pred_lgbm) / 4.0

    siglip_df = test_siglip.copy()
    siglip_df[inference_config.target_names] = siglip_pred
    siglip_df['Dry_Clover_g'] = 0.0
    siglip_df['GDM_g'] = siglip_df['Dry_Green_g']
    siglip_df['Dry_Total_g'] = siglip_df['GDM_g'] + siglip_df['Dry_Dead_g']

    # Create ensemble
    print("\n[5/6] Creating ensemble...")
    print(f"Weights: DINO={inference_config.w_dino}, SigLIP={inference_config.w_siglip}")

    ALL_TARGETS = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']

    final_df = test_wide.copy()

    for target in ALL_TARGETS:
        if target == 'Dry_Clover_g':
            final_df[target] = dino_df[target]
        else:
            final_df[target] = dino_df[target] * inference_config.w_dino + siglip_df[target] * inference_config.w_siglip

    final_df['Dry_Clover_g'] = final_df['Dry_Clover_g'].clip(lower=0.0)
    final_df['GDM_g'] = final_df['Dry_Green_g'] + final_df['Dry_Clover_g']
    final_df['Dry_Total_g'] = final_df['GDM_g'] + final_df['Dry_Dead_g']

    for col in ALL_TARGETS:
        final_df[col] = final_df[col].clip(lower=0.0)

    # Create submission
    print("\n[6/6] Creating submission...")

    submission_rows = []

    for _, row in final_df.iterrows():
        image_id = Path(row['image_path']).stem
        for target in inference_config.targets:
            submission_rows.append({
                'sample_id': f"{image_id}__{target}",
                'target': row[target]
            })

    submission = pd.DataFrame(submission_rows)
    submission.to_csv('submission.csv', index=False)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nSubmission saved: submission.csv")
    print(submission.head(10))
    print(f"\nStats:\n{submission['target'].describe()}")


if __name__ == "__main__":
    main()

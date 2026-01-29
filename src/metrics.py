"""
Evaluation metrics for CSIRO Biomass Prediction.

Author: Zulqarnain Ali
"""

import numpy as np

from src.config import train_config


def weighted_r2_score(y_true, y_pred, weights=None):
    """
    Calculate weighted R² score matching competition metric.

    Args:
        y_true: Ground truth values (N, 5)
        y_pred: Predicted values (N, 5)
        weights: Per-target weights. Defaults to competition weights.

    Returns:
        weighted_r2: Overall weighted R² score
        r2_scores: List of R² scores per target
    """
    if weights is None:
        weights = np.array(train_config.target_weights)

    # Log transform
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    r2_scores = []

    for i in range(y_true.shape[1]):
        yt = y_true_log[:, i]
        yp = y_pred_log[:, i]

        # Calculate R²
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)

        if ss_tot > 0:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = 0.0

        r2_scores.append(r2)

    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)

    return weighted_r2, r2_scores


def per_target_metrics(y_true, y_pred, target_names=None):
    """
    Calculate per-target metrics.

    Args:
        y_true: Ground truth values (N, 5)
        y_pred: Predicted values (N, 5)
        target_names: List of target names

    Returns:
        Dictionary of per-target metrics
    """
    if target_names is None:
        target_names = train_config.targets

    metrics = {}

    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    for i, name in enumerate(target_names):
        yt = y_true_log[:, i]
        yp = y_pred_log[:, i]

        # R²
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # RMSE
        rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))

        # MAE
        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))

        # Correlation
        corr = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]

        metrics[name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr
        }

    return metrics


def compute_metrics(y_true, y_pred, target_names=None, weights=None):
    """
    Compute all metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        target_names: Target names
        weights: Per-target weights

    Returns:
        Dictionary containing all metrics
    """
    weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred, weights)
    per_target = per_target_metrics(y_true, y_pred, target_names)

    return {
        'weighted_r2': weighted_r2,
        'per_target_r2': dict(zip(target_names, r2_scores)),
        'per_target_metrics': per_target
    }

"""
Loss functions for CSIRO Biomass Prediction.

Author: Zulqarnain Ali
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import train_config


class WeightedBiomassLoss(nn.Module):
    """
    Custom loss function matching the competition evaluation metric.

    Uses log-transformed Huber loss with per-target weights matching
    the competition's weighted R² metric.
    """

    def __init__(self, weights=None, beta: float = 5.0):
        """
        Initialize WeightedBiomassLoss.

        Args:
            weights: Tensor of shape (5,) with per-target weights.
                     Defaults to competition weights.
            beta: Huber loss beta parameter
        """
        super().__init__()

        if weights is None:
            weights = train_config.target_weights

        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.huber = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, preds, labels):
        """
        Calculate loss.

        Args:
            preds: Predicted values (B, 5)
            labels: Ground truth values (B, 5)

        Returns:
            Weighted loss value
        """
        # Log transform for better handling of biomass values
        preds_log = torch.log1p(preds)
        labels_log = torch.log1p(labels)

        # Huber loss
        loss = self.huber(preds_log, labels_log)

        # Apply weights
        weighted_loss = (loss * self.weights.to(loss.device)).mean()

        return weighted_loss


class MSELossWithWeights(nn.Module):
    """
    MSE loss with per-target weights.
    """

    def __init__(self, weights=None):
        """
        Initialize MSELossWithWeights.

        Args:
            weights: Per-target weights. Defaults to competition weights.
        """
        super().__init__()

        if weights is None:
            weights = train_config.target_weights

        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, preds, labels):
        """
        Calculate weighted MSE loss.

        Args:
            preds: Predicted values (B, 5)
            labels: Ground truth values (B, 5)

        Returns:
            Weighted MSE loss
        """
        mse = F.mse_loss(preds, labels, reduction='none')
        weighted_mse = (mse * self.weights.to(mse.device)).mean()
        return weighted_mse

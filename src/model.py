"""
Model architectures for CSIRO Biomass Prediction.

Author: Zulqarnain Ali
"""

import torch
import torch.nn as nn
import timm


class LocalMambaBlock(nn.Module):
    """
    Local Mamba Block for feature fusion.

    A lightweight attention-like mechanism that uses depthwise convolution
    for local pattern capture and a gating mechanism for information flow.
    """

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        """
        Initialize LocalMambaBlock.

        Args:
            dim: Feature dimension
            kernel_size: Kernel size for depthwise convolution
            dropout: Dropout probability
        """
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            Output tensor of shape (B, N, D)
        """
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)  # (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, N, D)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class BiomassModel(nn.Module):
    """
    Biomass prediction model using DINO v3 backbone.

    Processes left and right halves of images separately, then fuses
    features for multi-target prediction.
    """

    def __init__(self, model_name: str, pretrained: bool = True, dropout: float = 0.2):
        """
        Initialize BiomassModel.

        Args:
            model_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
        """
        super().__init__()
        self.model_name = model_name

        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        # Get number of features
        nf = self.backbone.num_features

        # Feature fusion
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=dropout),
            LocalMambaBlock(nf, kernel_size=5, dropout=dropout)
        )

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Prediction heads
        self.head_green = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nf // 2, 1),
            nn.Softplus()
        )

        self.head_dead = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nf // 2, 1),
            nn.Softplus()
        )

        self.head_clover = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nf // 2, 1),
            nn.Softplus()
        )

    def forward(self, left, right):
        """
        Forward pass.

        Args:
            left: Left image tensor (B, C, H, W)
            right: Right image tensor (B, C, H, W)

        Returns:
            Predictions for all 5 targets (B, 5)
        """
        # Extract features from both halves
        x_l = self.backbone(left)
        x_r = self.backbone(right)

        # Concatenate features
        x_cat = torch.cat([x_l, x_r], dim=1)

        # Fuse features
        x_fused = self.fusion(x_cat)

        # Pool features
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)

        # Predict individual components
        green = self.head_green(x_pool)
        dead = self.head_dead(x_pool)
        clover = self.head_clover(x_pool)

        # Calculate derived targets
        gdm = green + clover
        total = gdm + dead

        # Concatenate all predictions
        return torch.cat([green, dead, clover, gdm, total], dim=1)

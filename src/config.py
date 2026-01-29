"""
Configuration settings for CSIRO Biomass Prediction project.

Author: Zulqarnain Ali
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Paths
    data_path: str = "/kaggle/input/csiro1"
    output_dir: str = "/workspace/output"
    model_dir: str = "/workspace/output/models"

    # Model
    model_name: str = "vit_huge_plus_patch16_dinov3.lvd1689m"

    # Training
    seed: int = 42
    n_folds: int = 3
    folds_to_train: list = field(default_factory=lambda: [0, 1, 2])

    # Image dimensions
    img_height: int = 768
    img_width: int = 384
    batch_size: int = 4
    num_workers: int = 0

    # Training hyperparameters
    epochs: int = 180
    warmup_epochs: int = 6
    lr_backbone: float = 1e-5
    lr_head: float = 5e-4
    weight_decay: float = 1e-2

    # Regularization
    clip_grad_norm: float = 1.0
    dropout: float = 0.2
    early_stopping_patience: int = 30

    # Targets
    targets: list = field(
        default_factory=lambda: [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g"
        ]
    )
    target_cols: list = field(
        default_factory=lambda: [
            'Dry_Green_g',
            'Dry_Dead_g',
            'Dry_Clover_g',
            'GDM_g',
            'Dry_Total_g'
        ]
    )
    target_weights: list = field(
        default_factory=lambda: [0.1, 0.1, 0.1, 0.2, 0.5]
    )

    # Normalization
    mean: list = field(default_factory=lambda: [0.4417, 0.5036, 0.3057])
    std: list = field(default_factory=lambda: [0.2364, 0.2355, 0.2219])

    @property
    def device(self) -> str:
        """Get device string."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_train_csv(self) -> str:
        """Get training CSV path."""
        return os.path.join(self.data_path, "train.csv")

    def get_train_image_dir(self) -> str:
        """Get training images directory."""
        return os.path.join(self.data_path, "train")

    def get_test_csv(self) -> str:
        """Get test CSV path."""
        return os.path.join(self.data_path, "test.csv")

    def get_test_image_dir(self) -> str:
        """Get test images directory."""
        return os.path.join(self.data_path, "test")


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    # Paths
    data_path: str = "/kaggle/input/csiro-biomass"
    split_path: str = "/kaggle/input/csiro-datasplit/csiro_data_split.csv"
    models_dir: str = "/kaggle/input/dino-retrain-hu-2/models_trained"
    siglip_path: str = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"

    # Model
    model_name: str = "vit_huge_plus_patch16_dinov3.lvd1689m"

    # Settings
    seed: int = 42
    img_height: int = 768
    img_width: int = 384
    batch_size: int = 4
    n_folds: int = 3
    dropout: float = 0.2

    # Weights for ensemble
    w_dino: float = 0.75
    w_siglip: float = 0.25

    # Target names
    targets: list = field(
        default_factory=lambda: [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g"
        ]
    )
    target_names: list = field(
        default_factory=lambda: [
            'Dry_Clover_g',
            'Dry_Dead_g',
            'Dry_Green_g',
            'Dry_Total_g',
            'GDM_g'
        ]
    )

    # Target maximum values for normalization
    target_max: dict = field(
        default_factory=lambda: {
            "Dry_Clover_g": 71.7865,
            "Dry_Dead_g": 83.8407,
            "Dry_Green_g": 157.9836,
            "Dry_Total_g": 185.70,
            "GDM_g": 157.9836,
        }
    )

    # Normalization
    mean: list = field(default_factory=lambda: [0.4417, 0.5036, 0.3057])
    std: list = field(default_factory=lambda: [0.2364, 0.2355, 0.2219])

    @property
    def device(self) -> str:
        """Get device string."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


# Default configurations
train_config = TrainingConfig()
inference_config = InferenceConfig()

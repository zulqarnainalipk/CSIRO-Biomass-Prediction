"""
Data augmentation transforms for CSIRO Biomass Prediction.

Author: Zulqarnain Ali
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import train_config


def get_train_transforms(img_height=768, img_width=384, mean=None, std=None):
    """
    Get training augmentation transforms.

    Args:
        img_height: Target image height
        img_width: Target image width
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations Compose transform
    """
    if mean is None:
        mean = train_config.mean
    if std is None:
        std = train_config.std

    return A.Compose([
        A.Resize(img_height, img_width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_val_transforms(img_height=768, img_width=384, mean=None, std=None):
    """
    Get validation transforms (no augmentation).

    Args:
        img_height: Target image height
        img_width: Target image width
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations Compose transform
    """
    if mean is None:
        mean = train_config.mean
    if std is None:
        std = train_config.std

    return A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_tta_transforms(img_height=768, img_width=384, mean=None, std=None):
    """
    Get Test-Time Augmentation transforms.

    Args:
        img_height: Target image height
        img_width: Target image width
        mean: Normalization mean
        std: Normalization std

    Returns:
        List of transforms for TTA
    """
    if mean is None:
        mean = train_config.mean
    if std is None:
        std = train_config.std

    # Original
    transform_original = A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Horizontal flip
    transform_hflip = A.Compose([
        A.Resize(img_height, img_width),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Vertical flip
    transform_vflip = A.Compose([
        A.Resize(img_height, img_width),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Both flips
    transform_both = A.Compose([
        A.Resize(img_height, img_width),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return [
        transform_original,
        transform_hflip,
        transform_vflip,
        transform_both
    ]

"""
Dataset classes for CSIRO Biomass Prediction.

Author: Zulqarnain Ali
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from src.config import train_config


class BiomassDataset(Dataset):
    """
    Dataset for pasture biomass estimation.

    Splits each image vertically into two halves and processes them separately
    to extract more spatial features.
    """

    def __init__(
        self,
        df,
        img_dir,
        transform=None,
        target_cols=None,
        img_height=768,
        img_width=384
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame with image paths and labels
            img_dir: Directory containing images
            transform: Albumentations transform
            target_cols: List of target column names
            img_height: Target image height
            img_width: Target image width
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width

        if target_cols is None:
            target_cols = train_config.target_cols
        self.target_cols = target_cols

        self.paths = df['image_path'].values
        self.labels = df[self.target_cols].values.astype(np.float32)

    def __len__(self):
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            left: Transformed left image tensor
            right: Transformed right image tensor
            label: Target values tensor
        """
        img_name = os.path.basename(self.paths[idx])
        path = os.path.join(self.img_dir, img_name)

        # Read image
        img = cv2.imread(path)

        # Handle missing images
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split image vertically
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]

        # Apply transforms
        if self.transform:
            left = self.transform(image=left)['image']
            right = self.transform(image=right)['image']

        # Convert label to tensor
        label = torch.from_numpy(self.labels[idx])

        return left, right, label


class TestDataset(Dataset):
    """
    Dataset for test/inference.
    """

    def __init__(
        self,
        df,
        image_root,
        img_height=768,
        img_width=384,
        mean=None,
        std=None
    ):
        """
        Initialize the test dataset.

        Args:
            df: DataFrame with image paths
            image_root: Root directory for images
            img_height: Target image height
            img_width: Target image width
            mean: Normalization mean
            std: Normalization std
        """
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.img_height = img_height
        self.img_width = img_width

        if mean is None:
            mean = [0.4417, 0.5036, 0.3057]
        if std is None:
            std = [0.2364, 0.2355, 0.2219]

        self.mean = np.array(mean)
        self.std = np.array(std)

    def __len__(self):
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            left: Left image tensor
            right: Right image tensor
            info: Dictionary with sample info
        """
        row = self.df.iloc[idx]

        if isinstance(self.image_root, str):
            img_path = os.path.join(self.image_root, row["image_path"])
        else:
            img_path = self.image_root / row["image_path"]

        # Read image
        img = cv2.imread(str(img_path))

        # Handle missing images
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split image vertically
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]

        # Resize
        left = cv2.resize(left, (self.img_width, self.img_height))
        right = cv2.resize(right, (self.img_width, self.img_height))

        # Normalize
        left = left.astype(np.float32) / 255.0
        right = right.astype(np.float32) / 255.0
        left = (left - self.mean) / self.std
        right = (right - self.mean) / self.std

        # Convert to tensors
        left = torch.from_numpy(left.transpose(2, 0, 1)).float()
        right = torch.from_numpy(right.transpose(2, 0, 1)).float()

        return left, right, row.to_dict()


def collate_fn(batch):
    """
    Custom collate function for variable-length data.

    Args:
        batch: List of (left, right, info) tuples

    Returns:
        lefts: Stacked left images
        rights: Stacked right images
        infos: List of sample info dicts
    """
    lefts = torch.stack([b[0] for b in batch])
    rights = torch.stack([b[1] for b in batch])
    infos = [b[2] for b in batch]
    return lefts, rights, infos

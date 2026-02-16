"""
SkinVision — Dataset & DataLoaders
PyTorch Dataset for HAM10000 with augmentation support.
"""

import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

from src.config import (
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
    IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
)


def get_transforms(mode="train"):
    """
    Returns albumentations transforms for train/val/test.

    Train: augmentation + normalization
    Val/Test: just resize + normalization
    """
    if mode == "train":
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Supports images split across multiple directories (part_1, part_2).
    """

    def __init__(self, metadata_df, image_dirs, transform=None):
        """
        Args:
            metadata_df: DataFrame with columns ['image_id', 'dx']
            image_dirs: list of Path objects to folders containing .jpg images
            transform: albumentations transform
        """
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform

        # Encode labels as integers
        self.label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.df["label"] = self.df["dx"].map(self.label_map)

        # Build image path lookup from all directories
        self.image_path_map = {}
        if isinstance(image_dirs, (str, Path)):
            image_dirs = [Path(image_dirs)]
        for d in image_dirs:
            d = Path(d)
            if d.exists():
                for f in d.iterdir():
                    if f.suffix == ".jpg":
                        self.image_path_map[f.stem] = f

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]

        # Load image from path map
        img_path = self.image_path_map[image_id]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

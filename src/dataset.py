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

    Expects:
        - metadata_df: DataFrame with columns ['image_id', 'dx'] at minimum
        - image_dir: path to folder containing the .jpg images
        - transform: albumentations transform
    """

    def __init__(self, metadata_df, image_dir, transform=None):
        self.df = metadata_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Encode labels as integers
        self.label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.df["label"] = self.df["dx"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]

        # Load image
        img_path = self.image_dir / f"{image_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def prepare_dataloaders(metadata_path, image_dir, test_size=0.15, val_size=0.15):
    """
    Splits data into train/val/test with stratification and returns DataLoaders.

    Args:
        metadata_path: path to HAM10000_metadata.csv
        image_dir: path to folder with images
        test_size: fraction for test set
        val_size: fraction for validation set (from remaining after test)

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    df = pd.read_csv(metadata_path)

    # Stratified split: first split off test, then split remaining into train/val
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["dx"], random_state=SEED
    )
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_fraction, stratify=train_val_df["dx"], random_state=SEED
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Create datasets
    train_dataset = HAM10000Dataset(train_df, image_dir, transform=get_transforms("train"))
    val_dataset = HAM10000Dataset(val_df, image_dir, transform=get_transforms("val"))
    test_dataset = HAM10000Dataset(test_df, image_dir, transform=get_transforms("test"))

    # Compute class weights for imbalanced data
    label_counts = train_df["dx"].value_counts()
    total = len(train_df)
    class_weights = torch.tensor(
        [total / (len(CLASS_NAMES) * label_counts.get(c, 1)) for c in CLASS_NAMES],
        dtype=torch.float32,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights

"""
SkinVision — Preprocessing Utilities
Image loading, resizing, and preprocessing for inference.
"""

import numpy as np
from PIL import Image
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def load_and_preprocess(image_path):
    """
    Load an image from path and preprocess it for model inference.

    Args:
        image_path: path to the image file

    Returns:
        tensor: preprocessed image tensor (1, C, H, W) ready for model
        original: original image as numpy array (for visualization)
    """
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    transformed = transform(image=original)
    tensor = transformed["image"].unsqueeze(0)  # Add batch dimension

    return tensor, original


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: normalized image tensor (C, H, W)

    Returns:
        numpy array (H, W, C) with values 0-255
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    tensor = tensor.cpu().clone()
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)

    # Convert to numpy (H, W, C) and scale to 0-255
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    return image

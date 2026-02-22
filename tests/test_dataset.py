"""Tests for src/dataset.py — transforms and dataset class."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import torch

from src.dataset import get_transforms, HAM10000Dataset
from src.config import IMAGE_SIZE, CLASS_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image():
    """A random 100×100 RGB numpy array."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def image_dir(tmp_path):
    """A temp directory with a few dummy .jpg images per class."""
    for i, cls in enumerate(CLASS_NAMES):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(tmp_path / f"img_{cls}_{i}.jpg")
    return tmp_path


@pytest.fixture
def dummy_metadata(image_dir):
    """Metadata DataFrame referencing the dummy images."""
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        rows.append({"image_id": f"img_{cls}_{i}", "dx": cls})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

def test_train_transform_output_shape(dummy_image):
    transform = get_transforms("train")
    out = transform(image=dummy_image)["image"]
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_val_transform_output_shape(dummy_image):
    transform = get_transforms("val")
    out = transform(image=dummy_image)["image"]
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_test_transform_output_shape(dummy_image):
    transform = get_transforms("test")
    out = transform(image=dummy_image)["image"]
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_transform_returns_tensor(dummy_image):
    transform = get_transforms("val")
    out = transform(image=dummy_image)["image"]
    assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_dataset_length(dummy_metadata, image_dir):
    ds = HAM10000Dataset(dummy_metadata, [image_dir])
    assert len(ds) == len(CLASS_NAMES)


def test_dataset_item_shapes(dummy_metadata, image_dir):
    transform = get_transforms("val")
    ds = HAM10000Dataset(dummy_metadata, [image_dir], transform=transform)
    image, label = ds[0]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_dataset_label_is_int(dummy_metadata, image_dir):
    transform = get_transforms("val")
    ds = HAM10000Dataset(dummy_metadata, [image_dir], transform=transform)
    _, label = ds[0]
    assert isinstance(label, (int, np.integer))


def test_dataset_label_in_valid_range(dummy_metadata, image_dir):
    transform = get_transforms("val")
    ds = HAM10000Dataset(dummy_metadata, [image_dir], transform=transform)
    for i in range(len(ds)):
        _, label = ds[i]
        assert 0 <= label < len(CLASS_NAMES)

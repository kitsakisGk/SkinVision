"""Tests for src/preprocessing.py"""

import numpy as np
import torch
import pytest
from PIL import Image


from src.preprocessing import load_and_preprocess, denormalize
from src.config import IMAGE_SIZE


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary RGB image for testing."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


def test_load_and_preprocess_tensor_shape(sample_image_path):
    tensor, original = load_and_preprocess(sample_image_path)
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_load_and_preprocess_tensor_type(sample_image_path):
    tensor, _ = load_and_preprocess(sample_image_path)
    assert isinstance(tensor, torch.Tensor)


def test_load_and_preprocess_original_is_array(sample_image_path):
    _, original = load_and_preprocess(sample_image_path)
    assert isinstance(original, np.ndarray)
    assert original.ndim == 3  # (H, W, C)


def test_denormalize_output_shape():
    tensor = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)
    result = denormalize(tensor)
    assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_denormalize_output_dtype():
    tensor = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)
    result = denormalize(tensor)
    assert result.dtype == np.uint8


def test_denormalize_output_range():
    tensor = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)
    result = denormalize(tensor)
    assert result.min() >= 0
    assert result.max() <= 255

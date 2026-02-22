"""Tests for src/gradcam.py — GradCAM and overlay utilities."""

import numpy as np
import pytest
import torch

from src.gradcam import GradCAM, overlay_heatmap
from src.model import create_model
from src.config import IMAGE_SIZE, NUM_CLASSES


@pytest.fixture(scope="module")
def model():
    return create_model(pretrained=False)


@pytest.fixture(scope="module")
def gradcam(model):
    # EfficientNet-B0's last conv block accessible via timm internals
    target_layer = model.blocks[-1][-1].conv_pwl
    return GradCAM(model, target_layer)


@pytest.fixture
def dummy_input():
    return torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_gradcam_heatmap_shape(gradcam, dummy_input):
    heatmap, _, _ = gradcam.generate(dummy_input)
    assert heatmap.shape == (IMAGE_SIZE, IMAGE_SIZE)


def test_gradcam_heatmap_range(gradcam, dummy_input):
    heatmap, _, _ = gradcam.generate(dummy_input)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_predicted_class_valid(gradcam, dummy_input):
    _, predicted_class, _ = gradcam.generate(dummy_input)
    assert 0 <= predicted_class < NUM_CLASSES


def test_gradcam_confidence_range(gradcam, dummy_input):
    _, _, confidence = gradcam.generate(dummy_input)
    assert 0.0 <= confidence <= 1.0


def test_gradcam_target_class_override(gradcam, dummy_input):
    _, predicted_class, _ = gradcam.generate(dummy_input, target_class=0)
    assert predicted_class == 0


def test_overlay_heatmap_shape():
    image = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    heatmap = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    result = overlay_heatmap(image, heatmap)
    assert result.shape == image.shape


def test_overlay_heatmap_dtype():
    image = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    heatmap = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    result = overlay_heatmap(image, heatmap)
    assert result.dtype == np.uint8

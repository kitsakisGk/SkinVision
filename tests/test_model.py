"""Tests for src/model.py"""

import torch
import pytest

from src.model import create_model, freeze_base, unfreeze_all, get_model_summary
from src.config import NUM_CLASSES


@pytest.fixture(scope="module")
def model():
    return create_model(pretrained=False)


def test_create_model_returns_module(model):
    assert isinstance(model, torch.nn.Module)


def test_model_output_shape(model):
    dummy = torch.zeros(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, NUM_CLASSES)


def test_freeze_base_freezes_most_params(model):
    freeze_base(model)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert len(frozen) > 0, "Expected some frozen parameters after freeze_base()"


def test_freeze_base_keeps_head_trainable(model):
    freeze_base(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "Classifier head should remain trainable"


def test_unfreeze_all_makes_all_trainable(model):
    freeze_base(model)
    unfreeze_all(model)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert len(frozen) == 0, "All params should be trainable after unfreeze_all()"


def test_get_model_summary_keys(model):
    summary = get_model_summary(model)
    assert "total" in summary
    assert "trainable" in summary
    assert "frozen" in summary


def test_get_model_summary_totals(model):
    unfreeze_all(model)
    summary = get_model_summary(model)
    assert summary["total"] == summary["trainable"] + summary["frozen"]
    assert summary["total"] > 0

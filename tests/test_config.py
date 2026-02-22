"""Tests for src/config.py"""

from src.config import (
    CLASS_NAMES,
    CLASS_LABELS,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_NAME,
    SEED,
)


def test_class_names_length():
    assert len(CLASS_NAMES) == 7


def test_num_classes_matches_class_names():
    assert NUM_CLASSES == len(CLASS_NAMES)


def test_class_labels_cover_all_classes():
    for name in CLASS_NAMES:
        assert name in CLASS_LABELS, f"Missing label for class: {name}"


def test_class_labels_are_non_empty_strings():
    for key, label in CLASS_LABELS.items():
        assert isinstance(label, str) and len(label) > 0


def test_image_size_positive():
    assert IMAGE_SIZE > 0


def test_batch_size_positive():
    assert BATCH_SIZE > 0


def test_imagenet_stats_shape():
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3


def test_imagenet_mean_range():
    for v in IMAGENET_MEAN:
        assert 0.0 <= v <= 1.0


def test_imagenet_std_positive():
    for v in IMAGENET_STD:
        assert v > 0


def test_model_name_is_string():
    assert isinstance(MODEL_NAME, str) and len(MODEL_NAME) > 0


def test_seed_is_int():
    assert isinstance(SEED, int)

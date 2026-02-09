"""
SkinVision — Model
EfficientNet-based classifier using the timm library.
"""

import timm
import torch
import torch.nn as nn

from src.config import NUM_CLASSES, MODEL_NAME, PRETRAINED


def create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    """
    Creates a pretrained model and replaces the classification head.

    Args:
        model_name: timm model name (e.g., 'efficientnet_b3', 'resnet50')
        num_classes: number of output classes
        pretrained: whether to load ImageNet pretrained weights

    Returns:
        PyTorch model ready for fine-tuning
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_base(model):
    """Freeze all layers except the classification head."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head (timm models use different names)
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_all(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_model_summary(model):
    """Print a summary of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    print(f"Total parameters:     {total:>12,}")
    print(f"Trainable parameters: {trainable:>12,}")
    print(f"Frozen parameters:    {frozen:>12,}")

    return {"total": total, "trainable": trainable, "frozen": frozen}

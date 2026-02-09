"""
SkinVision — Evaluation & Visualization
Metrics, confusion matrix, ROC curves, and training history plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
)
from tqdm import tqdm

from src.config import CLASS_NAMES, CLASS_LABELS, RESULTS_DIR


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run model on all data and collect predictions + true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def print_classification_report(y_true, y_pred):
    """Print a formatted classification report."""
    target_names = [CLASS_LABELS[c] for c in CLASS_NAMES]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return report


def plot_confusion_matrix(y_true, y_pred, save=True):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = [CLASS_LABELS[c] for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
        print(f"Saved to {RESULTS_DIR / 'confusion_matrix.png'}")

    plt.show()
    return fig


def plot_roc_curves(y_true, y_probs, save=True):
    """Plot per-class ROC curves with AUC scores."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        label = CLASS_LABELS[class_name]
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
        print(f"Saved to {RESULTS_DIR / 'roc_curves.png'}")

    plt.show()
    return fig


def plot_training_history(history, save=True):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
        print(f"Saved to {RESULTS_DIR / 'training_curves.png'}")

    plt.show()
    return fig

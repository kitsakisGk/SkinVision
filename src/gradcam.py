"""
SkinVision — Grad-CAM
Visualize which regions of the image the model focuses on for its prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from src.config import CLASS_NAMES, CLASS_LABELS, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    Highlights the regions of an input image that are most important
    for the model's prediction.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: trained PyTorch model
            target_layer: the convolutional layer to extract gradients from
                          (typically the last conv layer before the classifier)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: preprocessed image tensor (1, C, H, W)
            target_class: class index to explain (None = predicted class)

        Returns:
            heatmap: numpy array (H, W) with values 0-1
            predicted_class: the class the model predicted
            confidence: softmax probability for the predicted class
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Forward pass
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # Only positive contributions
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        return cam, target_class, confidence


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay a Grad-CAM heatmap on an image.

    Args:
        image: original image as numpy array (H, W, 3), values 0-255
        heatmap: Grad-CAM heatmap (H, W), values 0-1
        alpha: transparency of the heatmap overlay

    Returns:
        overlaid image as numpy array
    """
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image
    heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))

    overlaid = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    return overlaid


def visualize_gradcam(image, heatmap, predicted_class, confidence, save_path=None):
    """
    Show original image, heatmap, and overlay side by side.

    Args:
        image: original image (H, W, 3), values 0-255
        heatmap: Grad-CAM heatmap (H, W), values 0-1
        predicted_class: class index
        confidence: prediction confidence
        save_path: optional path to save the figure
    """
    overlay = overlay_heatmap(image, heatmap)
    class_name = CLASS_LABELS[CLASS_NAMES[predicted_class]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {class_name}\nConfidence: {confidence:.1%}", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    return fig

"""
SkinVision — Gradio Demo App
Upload a dermatoscopic image → get prediction + Grad-CAM heatmap.
"""

import sys
from pathlib import Path

# Allow imports from project root
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import cv2
import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import (
    CLASS_NAMES, CLASS_LABELS, NUM_CLASSES,
    IMAGE_SIZE, MODEL_NAME, MODELS_DIR,
    IMAGENET_MEAN, IMAGENET_STD,
)
from src.model import create_model
from src.gradcam import GradCAM, overlay_heatmap

# ── Load model ──────────────────────────────────────────────
device = torch.device("cpu")
model = create_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
model.load_state_dict(
    torch.load(MODELS_DIR / "best_model.pth", map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()

# ── Grad-CAM setup ──────────────────────────────────────────
target_layer = model.bn2
grad_cam = GradCAM(model, target_layer)

# ── Preprocessing ───────────────────────────────────────────
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def predict(image):
    """Run prediction + Grad-CAM on an uploaded image."""
    if image is None:
        return None, {}

    # image comes as numpy array (H, W, 3) from Gradio
    original = np.array(image)
    original_resized = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

    # Preprocess
    transformed = transform(image=original)
    input_tensor = transformed["image"].unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor.to(device))
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    # Grad-CAM
    heatmap, pred_class, confidence = grad_cam.generate(input_tensor)
    overlay = overlay_heatmap(original_resized, heatmap, alpha=0.4)

    # Build label → confidence dict for Gradio
    confidences = {
        CLASS_LABELS[cls]: float(prob)
        for cls, prob in zip(CLASS_NAMES, probs)
    }

    return Image.fromarray(overlay), confidences


# ── Gradio Interface ────────────────────────────────────────
DESCRIPTION = """
# SkinVision — Skin Lesion Classifier

Upload a dermatoscopic image to classify it into one of **7 skin conditions** using EfficientNet-B0.

The model also shows a **Grad-CAM heatmap** highlighting where it's looking.

**Classes:** Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanoma, Melanocytic Nevus, Vascular Lesion

> **Disclaimer:** This is a demo project, NOT a medical diagnostic tool. Always consult a dermatologist.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Skin Lesion Image"),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM Overlay"),
        gr.Label(num_top_classes=7, label="Prediction"),
    ],
    title="SkinVision",
    description=DESCRIPTION,
)

if __name__ == "__main__":
    demo.launch()

---
title: SkinVision
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
short_description: Skin lesion classifier with Grad-CAM (EfficientNet-B0, HAM10000)
---

# SkinVision — Skin Lesion Classifier

Upload a dermatoscopic image to classify it into one of **7 skin conditions**
using EfficientNet-B0 fine-tuned on HAM10000.

The app also shows a **Grad-CAM heatmap** highlighting which regions the model
focuses on when making its prediction.

**Classes:** Actinic Keratosis · Basal Cell Carcinoma · Benign Keratosis ·
Dermatofibroma · Melanoma · Melanocytic Nevus · Vascular Lesion

> **Disclaimer:** This is an educational demo, NOT a medical diagnostic tool.
> Always consult a qualified dermatologist.

---

**Model:** EfficientNet-B0 · **Dataset:** HAM10000 (10,015 images) ·
**Test Accuracy:** 75.8% · **Mean AUC-ROC:** 0.957

[GitHub repo](https://github.com/kitsakisGk/SkinVision)

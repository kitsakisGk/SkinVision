# SkinVision — The Plan

## Inspiration

[VolleyVision](https://github.com/shukkkur/VolleyVision) showed us how computer vision can be applied to a real-world domain (volleyball) using models like YOLO, image classification, and segmentation. We're taking that same energy — **learning CV techniques by building something real** — and applying it to **health**: specifically, a **Skin Condition Detector**.

---

## What Are We Building?

A **Skin Condition Detection app** that takes an image of a skin area (mole, rash, lesion, etc.) and classifies it into possible conditions using deep learning image classification.

**Think of it like:** VolleyVision detects volleyballs and player actions → We detect skin conditions and their characteristics.

### What the User Does
1. Upload a photo of a skin area (or use webcam)
2. The model analyzes the image
3. Get a prediction: condition type, confidence score, and a recommendation to see a doctor if needed

> **Important disclaimer:** This is a **learning project**, NOT a medical device. Always consult a dermatologist for real diagnosis.

---

## CV Techniques We'll Learn (Mapped from VolleyVision)

| VolleyVision Technique | What We'll Learn Instead | How It Applies |
|------------------------|--------------------------|----------------|
| Image Classification (action recognition) | **Image Classification** (skin condition type) | Core of our project — classifying images into categories |
| Object Detection (YOLO) | **Object Detection / Localization** | Detecting and highlighting the lesion area in the image |
| Segmentation (court detection) | **Segmentation** | Segmenting the skin lesion from surrounding healthy skin |
| Data Augmentation | **Data Augmentation** | Critical for medical images — rotation, flipping, color jitter |
| Transfer Learning | **Transfer Learning** | Using pretrained models (ResNet, EfficientNet) on our dataset |
| Model Evaluation (mAP, precision, recall) | **Medical Metrics** (sensitivity, specificity, AUC-ROC) | Learning proper evaluation for health-related models |

---

## Skin Conditions to Detect

Using established dermatology datasets, we can classify these categories:

| Condition | Description |
|-----------|-------------|
| **Melanoma** | Most dangerous skin cancer — early detection saves lives |
| **Basal Cell Carcinoma** | Most common skin cancer — slow growing |
| **Benign Keratosis** | Non-cancerous growths (seborrheic keratosis, etc.) |
| **Dermatofibroma** | Harmless firm bumps |
| **Melanocytic Nevus** | Common moles — usually benign |
| **Vascular Lesion** | Blood vessel-related marks |
| **Actinic Keratosis** | Pre-cancerous rough patches from sun damage |

---

## Datasets We Can Use

| Dataset | Size | What It Has |
|---------|------|-------------|
| **ISIC 2019/2020** (International Skin Imaging Collaboration) | 25,000+ images | Dermoscopic images with labeled diagnoses — the gold standard |
| **HAM10000** | 10,015 images | 7 diagnostic categories, well-balanced, very popular for learning |
| **DermNet** | 23,000+ images | Clinical photos (not dermoscopic) — more like what a phone camera captures |
| **Fitzpatrick17k** | 16,577 images | Diverse skin tones — important for fairness/bias considerations |

**Recommendation:** Start with **HAM10000** — it's well-documented, widely used in tutorials, and has good class labels.

---

## Tech Stack

| Component | Tool / Library | Why |
|-----------|---------------|-----|
| **Deep Learning Framework** | PyTorch | Industry standard, flexible, great for learning |
| **Pretrained Models** | EfficientNet-B3 / ResNet50 | Transfer learning — don't train from scratch |
| **Image Processing** | OpenCV, Pillow, albumentations | Preprocessing, augmentation |
| **Data Handling** | Pandas, NumPy | Dataset management, label encoding |
| **Visualization** | Matplotlib, Seaborn, Plotly | Confusion matrices, training curves, interactive charts |
| **Explainability** | Grad-CAM | Show which part of the image the model is looking at |
| **UI / App** | Streamlit or Gradio | Upload image → get prediction (easy to build) |
| **Experiment Tracking** | Weights & Biases (wandb) | Track training runs, compare models |
| **Environment** | Local + Google Colab | Local for dev, Colab for GPU training |

---

## Architecture Overview

```
User uploads skin image
        │
        ▼
┌─────────────────────┐
│   Preprocessing      │
│  - Resize to 224x224 │
│  - Normalize pixels  │
│  - Augmentation      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Lesion Segmentation │  (Optional — Phase 3)
│  - U-Net / SAM       │
│  - Isolate lesion     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Classification Model │
│  - EfficientNet-B3    │
│  - Fine-tuned on      │
│    HAM10000           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Explainability       │
│  - Grad-CAM heatmap   │
│  - Confidence scores  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Output               │
│  - Predicted condition │
│  - Confidence %        │
│  - Heatmap overlay     │
│  - "See a doctor" flag │
└───────────────────────┘
```

---

## Project Structure

```
SkinVision/
├── README.md                        # The showpiece — banner, GIFs, results
├── The_plan.md                      # This file
├── requirements.txt                 # All dependencies
├── .gitignore                       # Ignore data/, models/, etc.
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA — class distribution, samples, stats
│   ├── 02_model_training.ipynb      # Transfer learning + training pipeline
│   ├── 03_evaluation.ipynb          # Metrics, confusion matrix, ROC curves
│   └── 04_gradcam.ipynb             # Explainability visualizations
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Paths, hyperparameters, class names
│   ├── dataset.py                   # PyTorch Dataset + DataLoader + augmentation
│   ├── model.py                     # Model architecture (EfficientNet wrapper)
│   ├── train.py                     # Training loop
│   ├── evaluate.py                  # Evaluation metrics + visualization
│   ├── gradcam.py                   # Grad-CAM implementation
│   └── preprocessing.py            # Image preprocessing utilities
│
├── app/
│   └── app.py                       # Streamlit / Gradio demo app
│
├── data/                            # Downloaded datasets (gitignored)
│   └── HAM10000/
│
├── models/                          # Saved model weights (gitignored)
│
└── results/                         # Generated outputs for README
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── roc_curves.png
    ├── class_distribution.png
    └── gradcam_samples/
```

---

## Implementation Roadmap

### Phase 1 — Data & EDA (Week 1–2)
- [ ] Set up Python environment (PyTorch, OpenCV, etc.)
- [ ] Download HAM10000 dataset
- [ ] **EDA notebook**: class distribution, image sizes, sample grids
- [ ] **Data pipeline**: PyTorch Dataset class with proper splits
- [ ] Understand train/val/test splits and stratification for medical data
- [ ] Read about skin condition types to understand what we're classifying

### Phase 2 — Build the Classification Model (Week 3–4)
- [ ] Implement data augmentation pipeline (albumentations)
- [ ] Load a pretrained EfficientNet-B3 (or ResNet50)
- [ ] Replace the final classification layer for our 7 classes
- [ ] Train with transfer learning (freeze base → train head → fine-tune all)
- [ ] Evaluate: accuracy, precision, recall, F1-score, confusion matrix
- [ ] Handle class imbalance (weighted loss, oversampling)
- [ ] Track experiments with wandb
- [ ] **Train multiple models** and create comparison table

### Phase 3 — Segmentation & Explainability (Week 5–6)
- [ ] Implement Grad-CAM to visualize what the model focuses on
- [ ] (Optional) Train a U-Net or use SAM for lesion segmentation
- [ ] Combine: segment lesion → classify → show heatmap of decision
- [ ] Evaluate segmentation quality (IoU, Dice score)

### Phase 4 — Build the App (Week 7–8)
- [ ] Build a Streamlit or Gradio web app
- [ ] Upload image → preprocess → run model → display results
- [ ] Show: prediction, confidence bar chart, Grad-CAM overlay
- [ ] Add disclaimer banner ("Not medical advice — see a dermatologist")
- [ ] Test with various image types and edge cases

### Phase 5 — Polish & Deploy (Week 9+)
- [ ] Test on DermNet dataset (clinical photos vs dermoscopic)
- [ ] Evaluate fairness across skin tones (Fitzpatrick17k)
- [ ] Deploy to Hugging Face Spaces or Streamlit Cloud (free hosting)
- [ ] Final README with all visuals, results, and demo link
- [ ] Docker container for reproducibility

---

## GitHub "Wow Factor" Checklist

Things that make people star a repo:

### Visual Impact
- [ ] **Banner image / logo** at the top of README
- [ ] **Live demo GIF** — upload image → prediction with heatmap appears
- [ ] **Before/after visuals** — original image vs Grad-CAM overlay
- [ ] **Architecture diagram** as a clean image (not just ASCII)
- [ ] **Sample prediction screenshots** showing the app in action

### Data Science Showcase
- [ ] **EDA section** in README with charts (class distribution, sample images)
- [ ] **Model comparison table** — EfficientNet vs ResNet vs VGG (accuracy, F1, AUC, speed)
- [ ] **Confusion matrix** — beautiful heatmap showing per-class performance
- [ ] **ROC curves** — one per class, showing AUC scores
- [ ] **Training curves** — loss and accuracy over epochs
- [ ] **Grad-CAM gallery** — grid of images with heatmap overlays

### Data Engineering Showcase
- [ ] **Clean data pipeline** — PyTorch Dataset class, DataLoaders, transforms
- [ ] **Reproducible environment** — requirements.txt, Docker, random seeds
- [ ] **Config-driven** — all hyperparameters in one config file
- [ ] **Proper train/val/test splits** with stratification
- [ ] **Data versioning** — document exactly which dataset version was used

### Professional Touches
- [ ] **Live demo link** — Hugging Face Spaces or Streamlit Cloud
- [ ] **Badges** in README (Python version, license, demo link, build status)
- [ ] **wandb report link** — public experiment tracking dashboard
- [ ] **Fairness analysis** — model performance across skin tones
- [ ] **Medical disclaimer** — shows responsibility and awareness
- [ ] **Contributing guide** + **License**
- [ ] **Clean commit history** — meaningful commit messages

---

## What You'll Learn From This Project

| Skill | Category | How You'll Learn It |
|-------|----------|---------------------|
| **Image Classification** | CV / ML | Core of the project — training CNNs on real data |
| **Transfer Learning** | CV / ML | Using pretrained models instead of training from scratch |
| **Data Augmentation** | CV / ML | Essential for small medical datasets |
| **Segmentation** | CV / ML | U-Net architecture for isolating regions of interest |
| **Explainable AI (XAI)** | CV / ML | Grad-CAM — critical for medical AI trust |
| **Class Imbalance Handling** | Data Science | Medical data is always imbalanced — learn to deal with it |
| **Model Evaluation** | Data Science | Precision, recall, F1, AUC-ROC, confusion matrices |
| **EDA & Visualization** | Data Science | Proper data exploration with publication-quality charts |
| **Data Pipelines** | Data Engineering | PyTorch Datasets, DataLoaders, transforms, splits |
| **Experiment Tracking** | MLOps | wandb — professional ML workflow |
| **Building ML Apps** | Engineering | Streamlit/Gradio — turning a model into a product |
| **Docker & Reproducibility** | Engineering | Containerizing ML applications |
| **Ethical AI** | Responsibility | Bias, fairness, and responsible disclaimers in health AI |

---

## Useful Resources

- HAM10000 dataset on Kaggle: search "HAM10000" or "Skin Cancer MNIST"
- ISIC Archive: the official skin imaging dataset collection
- PyTorch Transfer Learning tutorial (official docs)
- Grad-CAM paper and implementations on GitHub
- Streamlit docs for building the app
- Weights & Biases quickstart guide

---

**Let's learn some CV and build something cool!**

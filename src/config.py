"""
SkinVision — Configuration
All paths, hyperparameters, and class definitions in one place.
"""

import os
from pathlib import Path

# ============================================================
# Paths — Auto-detects Kaggle vs Local
# ============================================================
KAGGLE_DATA = Path("/kaggle/input/skin-cancer-mnist-ham10000")
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    # Running on Kaggle Notebooks
    PROJECT_ROOT = Path("/kaggle/working")
    DATA_DIR = KAGGLE_DATA
    IMAGE_DIR = DATA_DIR  # Kaggle puts all images flat in the input folder
else:
    # Running locally
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "HAM10000"
    IMAGE_DIR = DATA_DIR

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create dirs if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset
# ============================================================
# The 7 diagnostic categories in HAM10000
CLASS_NAMES = [
    "akiec",  # Actinic Keratosis
    "bcc",    # Basal Cell Carcinoma
    "bkl",    # Benign Keratosis
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic Nevus
    "vasc",   # Vascular Lesion
]

CLASS_LABELS = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# Hyperparameters
# ============================================================
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5

# ImageNet normalization (used by pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# Model
# ============================================================
MODEL_NAME = "efficientnet_b3"  # from timm library
PRETRAINED = True

# ============================================================
# Random Seed (reproducibility)
# ============================================================
SEED = 42

"""
SkinVision — Configuration
All paths, hyperparameters, and class definitions in one place.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "HAM10000"
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
# Hyperparameters (CPU-friendly defaults)
# ============================================================
IMAGE_SIZE = 128        # Smaller than 224 — faster on CPU
BATCH_SIZE = 16         # Smaller batches for CPU memory
NUM_WORKERS = 0         # 0 works best on Windows
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15         # Fewer epochs — fine-tuning doesn't need many
EARLY_STOPPING_PATIENCE = 4

# ImageNet normalization (used by pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# Model
# ============================================================
MODEL_NAME = "efficientnet_b0"  # b0 is much lighter than b3 — good for CPU
PRETRAINED = True

# ============================================================
# Random Seed (reproducibility)
# ============================================================
SEED = 42

"""
Configuration file for the Glaucoma Detection Project.
Contains all hyperparameters, paths, and settings.
"""
import os
import torch

# =========================
# DEVICE CONFIGURATION
# =========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# DATA PATHS
# =========================
# Root directory for datasets (will be created during download)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# ORIGA Dataset paths
ORIGA_ROOT = os.path.join(DATA_ROOT, 'ORIGA')
ORIGA_IMG_DIR = os.path.join(ORIGA_ROOT, 'Images_Square')
ORIGA_MASK_DIR = os.path.join(ORIGA_ROOT, 'Masks_Square')

# REFUGE Dataset paths
REFUGE_ROOT = os.path.join(DATA_ROOT, 'REFUGE')
REFUGE_IMG_DIR = os.path.join(REFUGE_ROOT, 'Images')
REFUGE_MASK_DIR = os.path.join(REFUGE_ROOT, 'Masks')

# =========================
# MODEL PATHS
# =========================
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'unet_origa_512.pth')

# =========================
# RESULTS PATHS
# =========================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
TASK3_RESULTS_DIR = os.path.join(RESULTS_DIR, 'task3_experiments')
TASK4_RESULTS_DIR = os.path.join(RESULTS_DIR, 'task4_refuge_evaluation')
TASK5_RESULTS_DIR = os.path.join(RESULTS_DIR, 'task5_cdr_analysis')

# =========================
# TRAINING HYPERPARAMETERS
# =========================
# Image processing
IMG_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 2

# Model training
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Validation split
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# =========================
# TASK 3: HYPERPARAMETER TUNING
# =========================
TASK3_EXPERIMENTS = [
    {
        'name': 'exp_1',
        'lr': 1e-3,
        'batch_size': 4,
        'augmentation': 'basic'
    },
    {
        'name': 'exp_2',
        'lr': 1e-4,
        'batch_size': 4,
        'augmentation': 'basic'
    },
    {
        'name': 'exp_3',
        'lr': 1e-4,
        'batch_size': 8,
        'augmentation': 'strong'
    }
]

# =========================
# TASK 5: CDR THRESHOLDS
# =========================
# Cup-to-Disc Ratio clinical thresholds
CDR_THRESHOLD_LOW = 0.5
CDR_THRESHOLD_MODERATE = 0.7

# Risk categories
RISK_LOW = 'low'
RISK_MODERATE = 'moderate'
RISK_HIGH = 'high'

# =========================
# VISUALIZATION
# =========================
FIGSIZE = (12, 8)
DPI = 100

# Project Structure Documentation

## Overview

This document provides detailed information about the project structure, module responsibilities, and data flow between components.

## Directory Structure

```
Final_project/
├── src/                          # Source code modules
├── scripts/                      # Executable scripts  
├── docs/                         # Documentation
├── results/                      # Output and results (gitignored)
├── models/                       # Trained models (gitignored)
├── data/                         # Datasets (gitignored)
├── README.md                     # Main documentation
├── requirements.txt              # Python dependencies
└── .gitignore                   # Git ignore rules
```

## Source Code Modules (`src/`)

### Core Infrastructure

#### `config.py`
Central configuration file containing all project settings:
- **Device Configuration**: CPU/GPU selection
- **Path Configuration**: Data, model, and results directories
- **Hyperparameters**: Learning rates, batch sizes, epochs
- **Task-specific Settings**: Experiment configs, CDR thresholds
- **Visualization Settings**: Figure sizes, DPI

#### `model.py`
U-Net architecture implementation:
- **`DoubleConv`**: Convolutional block (Conv2D → BatchNorm → ReLU) × 2
- **`UNet`**: Complete U-Net model with encoder-decoder architecture
- **`get_unet_model()`**: Factory function for model creation
- **`DiceLoss`**: Dice coefficient loss for segmentation
- **`CombinedLoss`**: Weighted combination of BCE and Dice losses

#### `dataset.py`
Dataset classes and data loading:
- **`ORIGADataset`**: Dataset class for ORIGA fundus images
- **`REFUGEDataset`**: Dataset class for REFUGE test images
- **`get_transforms()`**: Albumentations pipeline with augmentation options
- **`get_inference_transforms()`**: Transforms for inference (no augmentation)

#### `utils.py`
Utility functions and metrics:
- **Metrics**: `dice_coefficient()`, `calculate_iou()`, `calculate_cdr()`
- **Visualization**: `visualize_predictions()`, `plot_training_history()`, `plot_confusion_matrix()`
- **CDR Analysis**: `classify_glaucoma_risk()`
- **I/O**: `save_json()`, `load_json()`
- **Image Processing**: `create_overlay()`

### Task Modules

#### `data_acquisition.py`
Dataset download and verification:
- **`download_dataset()`**: Download from Kaggle using kagglehub
- **`verify_dataset_structure()`**: Verify ORIGA directory structure
- **`setup_data_directories()`**: Create project directories
- **Standalone execution**: Can be run directly with `python -m src.task1_data_acquisition`

#### `segmentation_model.py`
U-Net training pipeline:
- **`train_epoch()`**: Single epoch training loop
- **`validate_epoch()`**: Validation with metrics calculation
- **`train_segmentation_model()`**: Complete training workflow with:
  - Data loading and splitting
  - Model initialization
  - Training loop with early stopping
  - Best model checkpointing
  - History tracking

#### `hyperparameter_tuning.py`
Hyperparameter experimentation:
- **`run_experiment()`**: Execute single experiment with specific config
- **`run_all_experiments()`**: Run all experiment combinations
- **Tracks**: Learning rate, batch size, augmentation strategy
- **Outputs**: Per-experiment results, comparison table, recommendations

#### `refuge_testing.py`
Cross-dataset evaluation:
- **`test_on_refuge()`**: Evaluate model on REFUGE dataset
- **Metrics**: Dice coefficient, IoU score
- **Outputs**: Per-image metrics, statistical summary

#### `cdr_analysis.py`
Clinical CDR analysis and risk assessment:
- **`analyze_cdr()`**: Calculate CDR from segmentation masks
- **`create_cdr_visualizations()`**: Generate analysis plots
- **Risk Classification**: Low/Moderate/High based on thresholds
- **Statistical Analysis**: Pearson/Spearman correlation, Bland-Altman plots
- **Outputs**: CDR values, risk categories, confusion matrix

## Scripts (`scripts/`)

### `main.py`
Master execution script:
- **`run_task1()` - `run_task5()`**: Individual task runners
- **`main()`**: Orchestrates full pipeline or individual tasks
- **Command-line interface**: `--task N` to run specific task
- **Error handling**: Graceful failure with traceback

## Data Flow

```
1. Task 1: Download ORIGA dataset (650 images) → data/ORIGA/
   
2. Task 2: 
   - Split: 520 train + 130 validation
   - Train U-Net → models/unet_origa_512.pth
   - Output: Training history, curves
   
3. Task 3:
   - Run 3 experiments with different hyperparameters
   - Compare results → Best configuration
   - Output: Results table, recommendation
   
4. Task 4:
   - Load REFUGE dataset
   - Evaluate trained model
   - Output: Dice/IoU metrics
   
5. Task 5:
   - Predict segmentation masks
   - Calculate CDR from masks
   - Classify risk (Low/Moderate/High)
   - Output: CDR analysis, visualizations
```

## Configuration Management

The project uses a centralized configuration approach:

1. **Default Settings**: Defined in `src/config.py`
2. **Runtime Override**: Can be overridden when calling functions
3. **Path Resolution**: Automatic relative path resolution from project root

Example:
```python
from src import config

# Use default settings
print(config.LEARNING_RATE)  # 1e-4

# Override at runtime
train_segmentation_model(
    learning_rate=5e-5,  # Override default
    batch_size=8         # Override default
)
```

## Output Organization

### `results/`
Structured output directory:
```
results/
├── training_history.json
├── training_curves.png
├── task3_experiments/
│   ├── exp_1/
│   │   ├── best_model.pth
│   │   ├── history.json
│   │   └── summary.json
│   ├── exp_2/
│   ├── exp_3/
│   ├── results_table.csv
│   └── recommendation.txt
├── task4_refuge_evaluation/
│   ├── test_results.csv
│   └── test_summary.json
└── task5_cdr_analysis/
    ├── cdr_results.csv
    ├── summary.json
    ├── cdr_scatter.png
    ├── cdr_error_hist.png
    ├── bland_altman.png
    └── risk_confusion.png
```

## Module Dependencies

```
config.py (no dependencies)
    ↓
model.py → config
    ↓
dataset.py → config
    ↓
utils.py → config
    ↓
task1 → config
    ↓
task2 → config, model, dataset, utils
    ↓
task3 → config, model, dataset, task2, utils
    ↓
task4 → config, model, dataset, utils
    ↓
task5 → config, model, dataset, utils
    ↓
main.py → all task modules
```

## Development Guidelines

### Adding New Features

1. **New Task**: Create `src/taskN_description.py` following existing pattern
2. **New Utility**: Add to `src/utils.py` with proper documentation
3. **Configuration**: Add settings to `src/config.py`
4. **Documentation**: Update README.md and this file

### Code Style
- **Docstrings**: All functions must have docstrings
- **Type Hints**: Use where appropriate
- **Error Handling**: Graceful failure with informative messages
- **Logging**: Use print statements for progress updates

### Testing
Run individual modules to test:
```bash
python -m src.data_acquisition
python -m src.segmentation_model
```

## External Dependencies

See `requirements.txt` for complete list. Key dependencies:
- **PyTorch**: Deep learning framework
- **Albumentations**: Image augmentation
- **OpenCV**: Image processing
- **Pandas**: Data analysis
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Metrics and utilities

## Git Workflow

Files included in version control:
- ✅ Source code (`src/`, `scripts/`)
- ✅ Documentation (`README.md`, `docs/`)
- ✅ Configuration (`requirements.txt`, `.gitignore`)

Files excluded (via `.gitignore`):
- ❌ Data files (`data/`)
- ❌ Model weights (`models/`, `*.pth`)
- ❌ Results (`results/`)
- ❌ Notebooks (`*.ipynb`)
- ❌ Office files (`*.docx`, `*.pptx`)

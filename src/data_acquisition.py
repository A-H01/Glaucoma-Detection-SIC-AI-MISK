"""
Task 1: Data Acquisition
Download and setup the ORIGA glaucoma dataset from Kaggle.
"""
import os
import kagglehub
from pathlib import Path


def download_dataset():
    """
    Download the glaucoma datasets from Kaggle using kagglehub.
    
    Returns:
        Path to the downloaded dataset
    """
    print("Downloading glaucoma datasets from Kaggle...")
    print("This may take several minutes depending on your internet connection.")
    
    # Download latest version
    path = kagglehub.dataset_download("deathtrooper/glaucoma-datasets")
    
    print(f"Dataset downloaded successfully!")
    print(f"Path to dataset files: {path}")
    
    return path


def verify_dataset_structure(dataset_path):
    """
    Verify that the dataset has the expected structure.
    
    Args:
        dataset_path: Path to the downloaded dataset
    
    Returns:
        Dictionary with paths to ORIGA images and masks
    """
    origa_path = os.path.join(dataset_path, "ORIGA")
    
    if not os.path.exists(origa_path):
        raise FileNotFoundError(f"ORIGA directory not found at {origa_path}")
    
    # Check for images and masks directories
    img_dir = os.path.join(origa_path, "Images_Square")
    mask_dir = os.path.join(origa_path, "Masks_Square")
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Images directory not found at {img_dir}")
    
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Masks directory not found at {mask_dir}")
    
    # Count files
    import glob
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    masks = glob.glob(os.path.join(mask_dir, "*.png"))
    
    print(f"\nDataset verification:")
    print(f"  Images found: {len(images)}")
    print(f"  Masks found: {len(masks)}")
    
    if len(images) != len(masks):
        print("  WARNING: Number of images and masks don't match!")
    
    return {
        'origa_root': origa_path,
        'img_dir': img_dir,
        'mask_dir': mask_dir,
        'num_images': len(images),
        'num_masks': len(masks)
    }


def setup_data_directories(project_root):
    """
    Create necessary data directories in the project.
    
    Args:
        project_root: Root directory of the project
    """
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nCreated directories:")
    print(f"  {data_dir}")
    print(f"  {models_dir}")
    print(f"  {results_dir}")


if __name__ == "__main__":
    # Run data acquisition
    print("=" * 60)
    print("TASK 1: Data Acquisition")
    print("=" * 60)
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Verify structure
    dataset_info = verify_dataset_structure(dataset_path)
    
    print("\n" + "=" * 60)
    print("Task 1 completed successfully!")
    print("=" * 60)

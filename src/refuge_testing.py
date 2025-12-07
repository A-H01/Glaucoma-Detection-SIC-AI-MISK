"""
Task 4: REFUGE Dataset Testing
Evaluate the trained model on the REFUGE glaucoma dataset.
"""
import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from .model import get_unet_model
from .dataset import REFUGEDataset, get_inference_transforms
from .utils import dice_coefficient, calculate_iou, save_json
from . import config


def test_on_refuge(model_path, refuge_img_dir, refuge_mask_dir, batch_size=4):
    """
    Test the trained model on REFUGE dataset.
    
    Args:
        model_path: Path to trained model weights
        refuge_img_dir: Directory containing REFUGE images
        refuge_mask_dir: Directory containing REFUGE masks
        batch_size: Batch size for testing
    
    Returns:
        Dictionary with test results
    """
    print("=" * 60)
    print("TASK 4: REFUGE Dataset Evaluation")
    print("=" * 60)
    
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    model = get_unet_model(in_channels=3, out_channels=1, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load REFUGE dataset
    print("\nLoading REFUGE dataset...")
    images = sorted(glob.glob(os.path.join(refuge_img_dir, "*.jpg")))
    masks = sorted(glob.glob(os.path.join(refuge_mask_dir, "*.png")))
    
    print(f"Found {len(images)} images and {len(masks)} masks")
    
    # Create dataset and dataloader
    dataset = REFUGEDataset(images, masks, transform=get_inference_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=config.NUM_WORKERS)
    
    # Evaluation metrics
    all_dice_scores = []
    all_iou_scores = []
    
    print("\nEvaluating on REFUGE dataset...")
    
    with torch.no_grad():
        for images_batch, masks_batch in dataloader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)
            
            # Predict
            outputs = model(images_batch)
            preds = (outputs > 0.5).float()
            
            # Calculate metrics for each sample
            for pred, mask in zip(preds, masks_batch):
                dice = dice_coefficient(pred, mask)
                iou = calculate_iou(pred, mask)
                
                all_dice_scores.append(dice.item())
                all_iou_scores.append(iou.item())
    
    # Calculate statistics
    results = {
        'num_samples': len(all_dice_scores),
        'dice': {
            'mean': float(np.mean(all_dice_scores)),
            'std': float(np.std(all_dice_scores)),
            'min': float(np.min(all_dice_scores)),
            'max': float(np.max(all_dice_scores))
        },
        'iou': {
            'mean': float(np.mean(all_iou_scores)),
            'std': float(np.std(all_iou_scores)),
            'min': float(np.min(all_iou_scores)),
            'max': float(np.max(all_iou_scores))
        }
    }
    
    # Save detailed results
    df = pd.DataFrame({
        'image_idx': range(len(all_dice_scores)),
        'dice_score': all_dice_scores,
        'iou_score': all_iou_scores
    })
    
    os.makedirs(config.TASK4_RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.TASK4_RESULTS_DIR, 'test_results.csv'), index=False)
    save_json(results, os.path.join(config.TASK4_RESULTS_DIR, 'test_summary.json'))
    
    # Print results
    print("\n" + "=" * 60)
    print("REFUGE Test Results")
    print("=" * 60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"\nDice Coefficient:")
    print(f"  Mean: {results['dice']['mean']:.4f} ± {results['dice']['std']:.4f}")
    print(f"  Range: [{results['dice']['min']:.4f}, {results['dice']['max']:.4f}]")
    print(f"\nIoU Score:")
    print(f"  Mean: {results['iou']['mean']:.4f} ± {results['iou']['std']:.4f}")
    print(f"  Range: [{results['iou']['min']:.4f}, {results['iou']['max']:.4f}]")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = test_on_refuge(
        model_path=config.MODEL_SAVE_PATH,
        refuge_img_dir=config.REFUGE_IMG_DIR,
        refuge_mask_dir=config.REFUGE_MASK_DIR,
        batch_size=config.BATCH_SIZE
    )

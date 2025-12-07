"""
Task 5: Cup-to-Disc Ratio (CDR) Analysis
Calculate CDR from segmentation masks and classify glaucoma risk.
"""
import os
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from .model import get_unet_model
from .dataset import REFUGEDataset, get_inference_transforms
from .utils import calculate_cdr, classify_glaucoma_risk, save_json, plot_confusion_matrix
from . import config


def analyze_cdr(model_path, refuge_img_dir, refuge_mask_dir, batch_size=4):
    """
    Perform CDR analysis and glaucoma risk classification.
    
    Args:
        model_path: Path to trained model
        refuge_img_dir: Directory with REFUGE images
        refuge_mask_dir: Directory with REFUGE ground truth masks
        batch_size: Batch size for inference
    
    Returns:
        DataFrame with CDR analysis results
    """
    print("=" * 60)
    print("TASK 5: Cup-to-Disc Ratio (CDR) Analysis")
    print("=" * 60)
    
    device = config.DEVICE
    
    # Load model
    print("\nLoading model...")
    model = get_unet_model(in_channels=3, out_channels=1, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load dataset
    print("Loading REFUGE dataset...")
    images = sorted(glob.glob(os.path.join(refuge_img_dir, "*.jpg")))
    masks = sorted(glob.glob(os.path.join(refuge_mask_dir, "*.png")))
    
    dataset = REFUGEDataset(images, masks, transform=get_inference_transforms())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Store results
    results = []
    
    print("\nCalculating CDR for each image...")
    
    with torch.no_grad():
        for idx, (image, gt_mask) in enumerate(dataloader):
            image = image.to(device)
            
            # Predict mask
            output = model(image)
            pred_mask = (output > 0.5).float()
            
            # Convert to numpy
            pred_mask_np = pred_mask.cpu().squeeze().numpy()
            gt_mask_np = gt_mask.cpu().squeeze().numpy()
            
            # Calculate CDR
            pred_cdr = calculate_cdr(pred_mask_np)
            gt_cdr = calculate_cdr(gt_mask_np)
            
            # Classify risk
            pred_risk = classify_glaucoma_risk(pred_cdr, 
                                              config.CDR_THRESHOLD_LOW, 
                                              config.CDR_THRESHOLD_MODERATE)
            gt_risk = classify_glaucoma_risk(gt_cdr,
                                            config.CDR_THRESHOLD_LOW,
                                            config.CDR_THRESHOLD_MODERATE)
            
            results.append({
                'image_idx': idx,
                'image_path': images[idx],
                'pred_cdr': pred_cdr,
                'gt_cdr': gt_cdr,
                'cdr_error': abs(pred_cdr - gt_cdr),
                'pred_risk': pred_risk,
                'gt_risk': gt_risk
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("CDR Analysis Results")
    print("=" * 60)
    
    print(f"\nNumber of samples: {len(df)}")
    print(f"\nCDR Statistics:")
    print(f"  Predicted CDR - Mean: {df['pred_cdr'].mean():.4f} ± {df['pred_cdr'].std():.4f}")
    print(f"  Ground Truth CDR - Mean: {df['gt_cdr'].mean():.4f} ± {df['gt_cdr'].std():.4f}")
    print(f"  Mean Absolute Error: {df['cdr_error'].mean():.4f}")
    
    # Calculate correlation
    pearson_r, pearson_p = pearsonr(df['pred_cdr'], df['gt_cdr'])
    spearman_r, spearman_p = spearmanr(df['pred_cdr'], df['gt_cdr'])
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")
    
    # Risk classification accuracy
    risk_accuracy = (df['pred_risk'] == df['gt_risk']).mean()
    print(f"\nRisk Classification Accuracy: {risk_accuracy:.2%}")
    
    # Risk distribution
    print(f"\nRisk Distribution (Predicted):")
    print(df['pred_risk'].value_counts())
    
    print(f"\nRisk Distribution (Ground Truth):")
    print(df['gt_risk'].value_counts())
    
    # Save results
    os.makedirs(config.TASK5_RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.TASK5_RESULTS_DIR, 'cdr_results.csv'), index=False)
    
    summary = {
        'num_samples': len(df),
        'mean_pred_cdr': float(df['pred_cdr'].mean()),
        'mean_gt_cdr': float(df['gt_cdr'].mean()),
        'mean_abs_error': float(df['cdr_error'].mean()),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'risk_accuracy': float(risk_accuracy)
    }
    save_json(summary, os.path.join(config.TASK5_RESULTS_DIR, 'summary.json'))
    
    # Create visualizations
    create_cdr_visualizations(df)
    
    print("=" * 60)
    
    return df


def create_cdr_visualizations(df):
    """
    Create visualizations for CDR analysis.
    
    Args:
        df: DataFrame with CDR results
    """
    # Scatter plot: Predicted vs Ground Truth CDR
    plt.figure(figsize=(8, 6))
    plt.scatter(df['gt_cdr'], df['pred_cdr'], alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.xlabel('Ground Truth CDR')
    plt.ylabel('Predicted CDR')
    plt.title('CDR: Predicted vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.TASK5_RESULTS_DIR, 'cdr_scatter.png'), 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    # Error histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df['cdr_error'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('CDR Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of CDR Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.TASK5_RESULTS_DIR, 'cdr_error_hist.png'),
                dpi=100, bbox_inches='tight')
    plt.close()
    
    # Bland-Altman plot
    mean_cdr = (df['pred_cdr'] + df['gt_cdr']) / 2
    diff_cdr = df['pred_cdr'] - df['gt_cdr']
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_cdr, diff_cdr, alpha=0.6)
    plt.axhline(diff_cdr.mean(), color='r', linestyle='--', label='Mean difference')
    plt.axhline(diff_cdr.mean() + 1.96*diff_cdr.std(), color='g', linestyle='--', 
                label='±1.96 SD')
    plt.axhline(diff_cdr.mean() - 1.96*diff_cdr.std(), color='g', linestyle='--')
    plt.xlabel('Mean CDR')
    plt.ylabel('Difference (Predicted - Ground Truth)')
    plt.title('Bland-Altman Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.TASK5_RESULTS_DIR, 'bland_altman.png'),
                dpi=100, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix for risk classification
    plot_confusion_matrix(
        df['gt_risk'].tolist(),
        df['pred_risk'].tolist(),
        labels=[config.RISK_LOW, config.RISK_MODERATE, config.RISK_HIGH],
        save_path=os.path.join(config.TASK5_RESULTS_DIR, 'risk_confusion.png')
    )
    
    print("\n✓ Visualizations saved to results directory")


if __name__ == "__main__":
    df_results = analyze_cdr(
        model_path=config.MODEL_SAVE_PATH,
        refuge_img_dir=config.REFUGE_IMG_DIR,
        refuge_mask_dir=config.REFUGE_MASK_DIR,
        batch_size=1
    )

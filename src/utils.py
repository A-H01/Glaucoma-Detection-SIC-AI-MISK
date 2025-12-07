"""
Utility functions for visualization, metrics, and helper operations.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import cv2


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation evaluation.
    
    Args:
        pred: Predicted mask (binary)
        target: Ground truth mask (binary)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient value
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def calculate_iou(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) metric.
    
    Args:
        pred: Predicted mask (binary)
        target: Ground truth mask (binary)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU value
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def visualize_predictions(images, masks, predictions, num_samples=4, save_path=None):
    """
    Visualize original images, ground truth masks, and predictions.
    
    Args:
        images: Batch of images
        masks: Ground truth masks
        predictions: Predicted masks
        num_samples: Number of samples to visualize
        save_path: Path to save the figure (optional)
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Denormalize image for visualization
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        # Get masks
        gt_mask = masks[i].cpu().squeeze().numpy()
        pred_mask = predictions[i].cpu().squeeze().numpy()
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/metrics over epochs.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'val_dice'
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Dice coefficient
    axes[1].plot(history['val_dice'], label='Val Dice', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Validation Dice Coefficient')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def calculate_cdr(mask):
    """
    Calculate Cup-to-Disc Ratio from a segmentation mask.
    
    Args:
        mask: Segmentation mask with cup=1 and disc=2 (or combined)
    
    Returns:
        CDR value (float)
    """
    # Separate cup and disc
    cup = (mask == 1).astype(np.uint8)
    disc = ((mask == 1) | (mask == 2)).astype(np.uint8)
    
    # Calculate areas
    cup_area = np.sum(cup)
    disc_area = np.sum(disc)
    
    # Calculate CDR (handle division by zero)
    if disc_area == 0:
        return 0.0
    
    cdr = np.sqrt(cup_area / disc_area)
    return float(cdr)


def classify_glaucoma_risk(cdr, threshold_low=0.65, threshold_moderate=0.75):
    """
    Classify glaucoma risk based on CDR value.
    
    Args:
        cdr: Cup-to-Disc Ratio value
        threshold_low: Lower threshold for moderate risk
        threshold_moderate: Upper threshold for high risk
    
    Returns:
        Risk category: 'low', 'moderate', or 'high'
    """
    if cdr < threshold_low:
        return 'low'
    elif cdr < threshold_moderate:
        return 'moderate'
    else:
        return 'high'


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Risk')
    plt.xlabel('Predicted Risk')
    plt.title('Risk Classification Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()


def save_json(data, filepath):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_overlay(image, mask, alpha=0.4):
    """
    Create an overlay of the mask on the original image.
    
    Args:
        image: Original RGB image
        mask: Binary mask
        alpha: Transparency factor
    
    Returns:
        Overlay image
    """
    overlay = image.copy()
    
    # Create colored mask (green for visualization)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay

"""
Task 3: Hyperparameter Tuning
Experiment with different learning rates, batch sizes, and augmentation strategies.
"""
import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from .model import get_unet_model, CombinedLoss
from .dataset import ORIGADataset, get_transforms
from .task2_segmentation_model import train_epoch, validate_epoch
from .utils import save_json
from . import config


def run_experiment(exp_config, img_dir, mask_dir, num_epochs=20):
    """
    Run a single hyperparameter experiment.
    
    Args:
        exp_config: Dictionary with experiment configuration
        img_dir: Directory containing images
        mask_dir: Directory containing masks
        num_epochs: Number of epochs to train
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_config['name']}")
    print(f"Learning Rate: {exp_config['lr']}")
    print(f"Batch Size: {exp_config['batch_size']}")
    print(f"Augmentation: {exp_config['augmentation']}")
    print(f"{'='*60}")
    
    device = config.DEVICE
    
    # Load data
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED
    )
    
    # Create datasets with specified augmentation
    train_dataset = ORIGADataset(
        train_images, train_masks, 
        transform=get_transforms(exp_config['augmentation'])
    )
    val_dataset = ORIGADataset(
        val_images, val_masks, 
        transform=get_transforms('basic')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=exp_config['batch_size'], 
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=exp_config['batch_size'], 
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Initialize model
    model = get_unet_model(in_channels=3, out_channels=1, device=device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=exp_config['lr'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': []
    }
    
    best_dice = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Track best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            
            # Save best model for this experiment
            exp_dir = os.path.join(config.TASK3_RESULTS_DIR, exp_config['name'])
            os.makedirs(exp_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
    
    # Save experiment results
    exp_dir = os.path.join(config.TASK3_RESULTS_DIR, exp_config['name'])
    os.makedirs(exp_dir, exist_ok=True)
    
    save_json(history, os.path.join(exp_dir, 'history.json'))
    
    summary = {
        'name': exp_config['name'],
        'lr': exp_config['lr'],
        'batch_size': exp_config['batch_size'],
        'augmentation': exp_config['augmentation'],
        'best_val_dice': float(best_dice),
        'best_epoch': int(best_epoch)
    }
    save_json(summary, os.path.join(exp_dir, 'summary.json'))
    
    print(f"\n✓ Experiment completed! Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    
    return summary


def run_all_experiments(img_dir, mask_dir, num_epochs=20):
    """
    Run all hyperparameter tuning experiments.
    
    Args:
        img_dir: Directory containing images
        mask_dir: Directory containing masks
        num_epochs: Number of epochs per experiment
    
    Returns:
        List of experiment summaries
    """
    print("=" * 60)
    print("TASK 3: Hyperparameter Tuning")
    print("=" * 60)
    
    results = []
    
    for exp_config in config.TASK3_EXPERIMENTS:
        summary = run_experiment(exp_config, img_dir, mask_dir, num_epochs)
        results.append(summary)
    
    # Create comparison table
    df = pd.DataFrame(results)
    df = df.sort_values('best_val_dice', ascending=False)
    
    # Save results table
    os.makedirs(config.TASK3_RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.TASK3_RESULTS_DIR, 'results_table.csv'), index=False)
    
    # Find best configuration
    best_exp = df.iloc[0]
    
    # Save recommendation
    recommendation = f"""Recommended Configuration (Best Dice: {best_exp['best_val_dice']:.4f}):
- Experiment: {best_exp['name']}
- Learning Rate: {best_exp['lr']}
- Batch Size: {best_exp['batch_size']}
- Augmentation: {best_exp['augmentation']}
- Best Epoch: {best_exp['best_epoch']}
"""
    
    with open(os.path.join(config.TASK3_RESULTS_DIR, 'recommendation.txt'), 'w') as f:
        f.write(recommendation)
    
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Results")
    print("=" * 60)
    print(df.to_string(index=False))
    print("\n" + recommendation)
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_all_experiments(
        img_dir=config.ORIGA_IMG_DIR,
        mask_dir=config.ORIGA_MASK_DIR,
        num_epochs=config.NUM_EPOCHS
    )

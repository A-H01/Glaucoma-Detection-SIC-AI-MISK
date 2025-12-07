"""
Task 2: Optic Disc and Cup Segmentation using U-Net
Train a U-Net model for segmenting optic disc and cup from fundus images.
"""
import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from .model import get_unet_model, CombinedLoss
from .dataset import ORIGADataset, get_transforms
from .utils import dice_coefficient, plot_training_history, visualize_predictions, save_json
from . import config


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: U-Net model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model on validation set.
    
    Args:
        model: U-Net model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device (CPU/GPU)
    
    Returns:
        Average validation loss and Dice coefficient
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate Dice
            preds = (outputs > 0.5).float()
            dice = dice_coefficient(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
    
    return total_loss / len(dataloader), total_dice / len(dataloader)


def train_segmentation_model(img_dir, mask_dir, num_epochs=20, batch_size=4, 
                            learning_rate=1e-4, save_path=None):
    """
    Train the U-Net segmentation model.
    
    Args:
        img_dir: Directory containing training images
        mask_dir: Directory containing segmentation masks
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save the best model
    
    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print("TASK 2: Training U-Net Segmentation Model")
    print("=" * 60)
    
    # Get device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load image and mask paths
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    print(f"\nDataset size: {len(images)} images")
    
    # Split into train and validation
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Create datasets
    train_dataset = ORIGADataset(train_images, train_masks, transform=get_transforms('basic'))
    val_dataset = ORIGADataset(val_images, val_masks, transform=get_transforms('basic'))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=config.NUM_WORKERS)
    
    # Initialize model
    print("\nInitializing U-Net model...")
    model = get_unet_model(in_channels=3, out_channels=1, device=device)
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': []
    }
    
    best_dice = 0.0
    patience_counter = 0
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"✓ Best model saved with Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    # Train model
    model, history = train_segmentation_model(
        img_dir=config.ORIGA_IMG_DIR,
        mask_dir=config.ORIGA_MASK_DIR,
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        save_path=config.MODEL_SAVE_PATH
    )
    
    # Save training history
    save_json(history, os.path.join(config.RESULTS_DIR, 'training_history.json'))
    
    # Plot history
    plot_training_history(history, 
                         save_path=os.path.join(config.RESULTS_DIR, 'training_curves.png'))

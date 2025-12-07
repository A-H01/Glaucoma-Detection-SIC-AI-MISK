"""
Dataset classes for ORIGA and REFUGE datasets.
Handles image loading, preprocessing, and augmentation.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ORIGADataset(Dataset):
    """
    Dataset class for ORIGA fundus images and segmentation masks.
    
    Args:
        image_paths: List of paths to fundus images
        mask_paths: List of paths to segmentation masks
        transform: Albumentations transform pipeline
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # Add channel dimension
        
        return image, mask


class REFUGEDataset(Dataset):
    """
    Dataset class for REFUGE fundus images and segmentation masks.
    
    Args:
        image_paths: List of paths to fundus images
        mask_paths: List of paths to segmentation masks (optional)
        transform: Albumentations transform pipeline
    """
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        if self.mask_paths is not None:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
        
        # Apply transformations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask'].unsqueeze(0)
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
        
        if mask is not None:
            return image, mask
        else:
            return image


def get_transforms(augmentation='basic', img_size=512):
    """
    Get albumentations transform pipeline.
    
    Args:
        augmentation: 'basic' or 'strong'
        img_size: Target image size (default: 512)
    
    Returns:
        Albumentations Compose transform
    """
    if augmentation == 'strong':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
    else:  # basic
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])


def get_inference_transforms(img_size=512):
    """
    Get transform pipeline for inference (no augmentation).
    
    Args:
        img_size: Target image size (default: 512)
    
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])

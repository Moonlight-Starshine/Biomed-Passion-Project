"""
Data loading, preprocessing, and augmentation for disease detection.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path


class HistopathologyDataset(Dataset):
    """Load histopathology images with disease labels."""
    
    def __init__(self, image_paths, labels, transform=None, image_size=224):
        """
        Args:
            image_paths: List of image file paths
            labels: List of integer labels (0, 1, 2, ...)
            transform: Torchvision transforms
            image_size: Resize images to this size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Stain normalization (simple histogram equalization)
        image = self._stain_normalize(image)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Apply augmentations
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        label = self.labels[idx]
        
        return image, label
    
    @staticmethod
    def _stain_normalize(image):
        """Simple stain normalization using histogram equalization."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Equalize value channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2].astype(np.uint8))
        # Back to RGB
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return image


def load_data_from_directory(data_dir="data/raw", image_size=224):
    """
    Load images organized as: data/raw/{disease_name}/{image.jpg}
    
    Returns:
        - disease_names: List of disease class names
        - X_train, X_val, X_test: Image path lists
        - y_train, y_val, y_test: Label lists
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    image_paths = []
    labels = []
    disease_names = []
    label_map = {}
    
    # Scan directories
    for disease_idx, disease_dir in enumerate(sorted(data_path.iterdir())):
        if not disease_dir.is_dir():
            continue
        
        disease_name = disease_dir.name
        disease_names.append(disease_name)
        label_map[disease_name] = disease_idx
        
        # Find images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            for img_path in disease_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(disease_idx)
    
    if not image_paths:
        raise ValueError(
            f"No images found in {data_path}. "
            "Please organize images as: data/raw/{{disease_name}}/{{image.jpg}}"
        )
    
    print(f"Found {len(image_paths)} images across {len(disease_names)} diseases")
    print(f"Disease classes: {disease_names}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    return disease_names, (X_train, X_val, X_test), (y_train, y_val, y_test)


def get_dataloaders(
    data_dir="data/raw",
    batch_size=32,
    num_workers=4,
    image_size=224,
    augment=True
):
    """Create train/val/test dataloaders with augmentations."""
    
    disease_names, (X_train, X_val, X_test), (y_train, y_val, y_test) = \
        load_data_from_directory(data_dir, image_size)
    
    # Training augmentations
    if augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Datasets
    train_dataset = HistopathologyDataset(
        X_train, y_train, train_transform, image_size
    )
    val_dataset = HistopathologyDataset(
        X_val, y_val, val_transform, image_size
    )
    test_dataset = HistopathologyDataset(
        X_test, y_test, val_transform, image_size
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, disease_names

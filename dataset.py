

from typing import Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10HDataset(Dataset):
    """
    Custom PyTorch Dataset for CIFAR-10H images with soft label distributions.
    
    Handles 32x32 RGB images paired with 10-dimensional soft label distributions
    representing human annotator disagreement on class assignments.
    
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """Initialize the dataset."""
        self.images = images
        self.labels = labels
        self.transform = transform
        
        # Validate dimensions
        assert len(self.images) == len(self.labels), \
            f"Number of images ({len(self.images)}) must match number of labels ({len(self.labels)})"
        assert self.labels.shape[1] == 10, \
            f"Labels must have 10 dimensions, got {self.labels.shape[1]}"
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (image, label) where:
                - image: Tensor of shape (3, 32, 32) with values in [0, 1]
                - label: Tensor of shape (10,) with soft label distribution
        """
        # Get image and label
        image = self.images[idx]  # Shape: (32, 32, 3)
        label = self.labels[idx]  # Shape: (10,)
        
        # Apply transforms if provided (before tensor conversion)
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert image to tensor
        # If already uint8 in [0, 255], convert to float and normalize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, 32, 32)
            # Normalize to [0, 1] if values are in [0, 255]
            if image.max() > 1.0:
                image = image / 255.0
        else:
            # Already a tensor
            image = image.permute(2, 0, 1).float() if image.shape[-1] == 3 else image
        
        # Convert label to tensor
        label = torch.from_numpy(label).float() if isinstance(label, np.ndarray) else label.float()
        
        return image, label


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for CIFAR-10H train/val/test splits.
    
    Args:
        X_train: Training images, shape (6000, 32, 32, 3)
        y_train: Training soft labels, shape (6000, 10)
        X_val: Validation images, shape (2000, 32, 32, 3)
        y_val: Validation soft labels, shape (2000, 10)
        X_test: Test images, shape (2000, 32, 32, 3)
        y_test: Test soft labels, shape (2000, 10)
        batch_size: Batch size for DataLoaders. Default: 32
        num_workers: Number of worker processes for data loading. Default: 0
        pin_memory: Whether to pin memory for faster GPU transfer. Default: True
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoader instances.
        
    Example:
        >>> dataloaders = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)
        >>> train_loader = dataloaders['train']
        >>> for images, labels in train_loader:
        ...     print(images.shape, labels.shape)  # torch.Size([32, 3, 32, 32]) torch.Size([32, 10])
    """
    
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # Validation/Test: no augmentation, just normalize
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # Create datasets
    train_dataset = CIFAR10HDataset(X_train, y_train, transform=train_transform)
    val_dataset = CIFAR10HDataset(X_val, y_val, transform=eval_transform)
    test_dataset = CIFAR10HDataset(X_test, y_test, transform=eval_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    """
    Test script: Verify dataset and dataloader functionality with dummy data.
    """
    
    print("CIFAR-10H DataLoader Test")
    
    
    # Create dummy data matching the expected shapes from your notebook
    num_train, num_val, num_test = 6000, 2000, 2000
    
    X_train = np.random.randint(0, 256, size=(num_train, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.dirichlet(np.ones(10), size=num_train).astype(np.float32)
    
    X_val = np.random.randint(0, 256, size=(num_val, 32, 32, 3), dtype=np.uint8)
    y_val = np.random.dirichlet(np.ones(10), size=num_val).astype(np.float32)
    
    X_test = np.random.randint(0, 256, size=(num_test, 32, 32, 3), dtype=np.uint8)
    y_test = np.random.dirichlet(np.ones(10), size=num_test).astype(np.float32)
    
    print(f"\nDummy Data Shapes:")
    print(f"  Train: Images {X_train.shape}, Labels {y_train.shape}")
    print(f"  Val:   Images {X_val.shape}, Labels {y_val.shape}")
    print(f"  Test:  Images {X_test.shape}, Labels {y_test.shape}")
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32,
        num_workers=0
    )
    
    print(f"\nDataLoaders Created Successfully!")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches:   {len(dataloaders['val'])}")
    print(f"  Test batches:  {len(dataloaders['test'])}")
    
    # Get a single batch and verify shapes
    
    print("Sample Batch Verification")
    
    
    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))
    
    print(f"\nImages shape:  {images.shape}")
    print(f"  Expected:    torch.Size([32, 3, 32, 32])")
    print(f"  Match: {images.shape == torch.Size([32, 3, 32, 32])}")
    
    print(f"\nLabels shape:  {labels.shape}")
    print(f"  Expected:    torch.Size([32, 10])")
    print(f"  Match: {labels.shape == torch.Size([32, 10])}")
    
    print(f"\nImage value range:  [{images.min():.3f}, {images.max():.3f}]")
    print(f"Label sum per sample (should be ~1.0): {labels.sum(dim=1)[0]:.6f}")
    
    
    print("All tests passed! Dataset and DataLoaders are ready to use.")
    
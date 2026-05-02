"""
finetune.py — Day 4: Fine-tuning on CIFAR-10H Soft Labels
==========================================================
Author: Pavan
Purpose: Fine-tune pretrained backbone on 6,000 CIFAR-10H training images
         using three different loss functions and compare performance

Usage:
    python finetune.py

Output:
    best_model_kl.pth      — best model trained with KL divergence
    best_model_jsd.pth     — best model trained with Jensen-Shannon divergence
    best_model_custom.pth  — best model trained with custom composite loss
    training_logs.pkl      — training history for plotting
"""

import os
import json
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Import project modules ────────────────────────────────────────────────
from model import CIFAR10H_ResNet
from dataset import CIFAR10HDataset  # Assume this exists from Day 1-2
from losses import KLDivergenceLoss, JensenShannonDivergenceLoss, CustomCompositeEntropy


# ══════════════════════════════════════════════════════════════════════════
# 0. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════
# 1. HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
NUM_EPOCHS      = 200
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-5      # CHANGED: was 1e-4 (too high for fine-tuning)
WEIGHT_DECAY    = 1e-5
PATIENCE        = 50        # CHANGED: was 10 (too aggressive early stopping)
BACKBONE_PATH   = "pretrained_backbone.pth"
NUM_WORKERS     = 0            # Set to 0 for macOS to avoid multiprocessing issues

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ══════════════════════════════════════════════════════════════════════════
# 2. DEVICE SETUP
# ══════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[Device] Using: {device}")


# ══════════════════════════════════════════════════════════════════════════
# 3. DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════
print(f"[Data] Loading CIFAR-10H dataset...")

# Load CIFAR-10 images from pickle files
def unpickle(file):
    """Load CIFAR-10 pickle files"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load all CIFAR-10 training batches
train_images = []
train_labels_hard = []
for i in range(1, 6):
    batch = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
    train_images.append(batch[b'data'])
    train_labels_hard.append(batch[b'labels'])

train_images = np.concatenate(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (50000, 32, 32, 3)
train_labels_hard = np.concatenate(train_labels_hard)

# Load test batch
test_batch = unpickle('./data/cifar-10-batches-py/test_batch')
test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (10000, 32, 32, 3)
test_labels_hard = np.array(test_batch[b'labels'])

print(f"  CIFAR-10 images loaded: {train_images.shape} (train), {test_images.shape} (test)")

# Load CIFAR-10H soft labels (10,000 labels for test set)
cifar10h_probs = np.load('./cifar10h-probs.npy')  # Shape: (10000, 10)
print(f"  CIFAR-10H soft labels loaded: {cifar10h_probs.shape}")

# Use only the first 10,000 images from CIFAR-10 training set (matching CIFAR-10H)
# Split into train (6000), val (2000), test (2000)
X_train = train_images[:6000]
y_train = cifar10h_probs[:6000]

X_val = train_images[6000:8000]
y_val = cifar10h_probs[6000:8000]

X_test = train_images[8000:10000]
y_test = cifar10h_probs[8000:10000]

print(f"  Train: {X_train.shape}, {y_train.shape}")
print(f"  Val:   {X_val.shape}, {y_val.shape}")
print(f"  Test:  {X_test.shape}, {y_test.shape}")

# Create datasets with transforms
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),     # NEW: augmentation
    transforms.RandomCrop(32, padding=4),        # NEW: augmentation
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_dataset = CIFAR10HDataset(X_train, y_train, transform=train_transform)   # changed to train_transform
val_dataset = CIFAR10HDataset(X_val, y_val, transform=val_transform)
test_dataset = CIFAR10HDataset(X_test, y_test, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

print(f"[Data] DataLoaders created successfully")


# ══════════════════════════════════════════════════════════════════════════
# 4. MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════
def build_model(backbone_path, freeze_backbone=False):
    """
    Load backbone and set up for fine-tuning
    
    Args:
        backbone_path: Path to pretrained backbone weights
        freeze_backbone: If True, don't update backbone weights during fine-tuning
    
    Returns:
        model: CIFAR10H_ResNet model ready for fine-tuning
    """
    # Load backbone
    model = CIFAR10H_ResNet()
    
    if os.path.exists(backbone_path):
        state_dict = torch.load(backbone_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Model] Loaded pretrained backbone from {backbone_path}")
    else:
        print(f"[WARNING] Backbone file not found at {backbone_path}. Using random initialization.")
    
    # Decide on backbone freezing strategy
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print(f"[Model] Backbone FROZEN (weights will not be updated)")
    else:
        print(f"[Model] Backbone will be FINE-TUNED (weights will be updated)")
    
    model = model.to(device)
    return model


# ══════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
def run_epoch(loader, model, criterion, optimizer=None, training=False):
    """
    Run one epoch on a dataloader
    
    Returns:
        avg_loss: Average loss across batches
    """
    model.train() if training else model.eval()
    total_loss = 0.0
    
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, soft_labels in loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, soft_labels)
            
            # Backward pass (if training)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * images.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def train_with_loss(model, train_loader, val_loader, criterion, loss_name, save_path):
    """
    Train model with given loss function
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        loss_name: Name of loss (for logging)
        save_path: Where to save best checkpoint
    
    Returns:
        history: Dict with training/validation loss per epoch
    """
    # Filter for trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.Adam(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\n{'='*70}")
    print(f"Training with {loss_name}")
    print(f"{'='*70}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Best Val':>12} {'LR':>10}")
    print(f"{'-'*70}")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        
        train_loss = run_epoch(train_loader, model, criterion, optimizer, training=True)
        val_loss = run_epoch(val_loader, model, criterion, optimizer=None, training=False)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>12.4f} {best_val_loss:>12.4f} {current_lr:>10.6f} ({elapsed:.1f}s)")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"          ✓ New best val_loss={best_val_loss:.4f} → saved to {save_path}")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n[Early Stop] No improvement for {PATIENCE} epochs. Stopping at epoch {epoch}.")
            break
    
    print(f"{'-'*70}")
    print(f"Best validation loss: {best_val_loss:.4f}\n")
    
    return history


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING
# ══════════════════════════════════════════════════════════════════════════
def main():
    """Main training pipeline for all three loss functions"""
    
    # Define loss functions and save paths
    losses_config = [
        {
            'name': 'KL Divergence',
            'criterion': KLDivergenceLoss(),
            'save_path': 'best_model_kl.pth',
        },
        {
            'name': 'Jensen-Shannon Divergence',
            'criterion': JensenShannonDivergenceLoss(),
            'save_path': 'best_model_jsd.pth',
        },
        {
            'name': 'Custom Composite (KL + Entropy)',
            'criterion': CustomCompositeEntropy(lambda1=1.0, lambda2=0.5),
            'save_path': 'best_model_custom.pth',
        },
    ]
    
    all_histories = {}
    
    # Train with each loss function
    for config in losses_config:
        # Rebuild model for each loss (fresh start)
        model = build_model(BACKBONE_PATH, freeze_backbone=False)  # Fine-tune backbone on soft labels
        
        # Train
        history = train_with_loss(
            model,
            train_loader,
            val_loader,
            config['criterion'],
            config['name'],
            config['save_path'],
        )
        
        all_histories[config['name']] = history
    
    # Save training histories
    with open('training_logs.pkl', 'wb') as f:
        pickle.dump(all_histories, f)
    print(f"\n[Done] Saved training logs to training_logs.pkl")
    
    return all_histories


if __name__ == "__main__":
    histories = main()
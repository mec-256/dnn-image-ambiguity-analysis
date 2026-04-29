"""
pretrain.py — Day 3: Backbone Pretraining
==========================================
Author : Pranav
Purpose: Pretrain the CIFAR10H_ResNet backbone on the 50,000 standard
         CIFAR-10 hard-label training images using CrossEntropyLoss.
         The saved .pth weights will be used in later phases (Day 4+)
         as a strong initialisation before fine-tuning on soft labels.

Usage:
    python pretrain.py

Output:
    pretrained_backbone.pth  — model weights saved to disk after training
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# ── Import Eswar's backbone ────────────────────────────────────────────────
from model import CIFAR10H_ResNet


# ══════════════════════════════════════════════════════════════════════════
# 0. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════
# 1. HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
NUM_EPOCHS    = 100      # maximum training epochs
BATCH_SIZE    = 128
LEARNING_RATE = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
PATIENCE      = 10       # early-stopping patience (epochs without improvement)
SAVE_PATH     = "pretrained_backbone.pth"
NUM_WORKERS   = 4        # DataLoader workers; set to 0 if debugging on Windows


# ══════════════════════════════════════════════════════════════════════════
# 2. DEVICE SETUP  (CUDA → MPS → CPU, in priority order)
# ══════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():       # Apple Silicon
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[Device] Using: {device}")


# ══════════════════════════════════════════════════════════════════════════
# 3. DATA PIPELINE
#    As specified in the project doc (Section 5.4):
#      • Allowed augmentations: random crop + padding, random horizontal flip
#      • Do NOT use augmentations that change class semantics
#    CIFAR-10 channel statistics (mean / std) taken from the standard dataset.
# ══════════════════════════════════════════════════════════════════════════
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# Training transforms — augmentation only applied during training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # project-approved augmentation
    transforms.RandomHorizontalFlip(),              # project-approved augmentation
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Validation / test transforms — no augmentation, just normalise
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Download the full 50,000-image CIFAR-10 training set
full_train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

# Hold out 10% (5,000 images) as a pretraining validation split.
# This lets us track generalisation and apply early stopping without
# touching the 10,000-image CIFAR-10H test portion at all.
n_total    = len(full_train_dataset)        # 50,000
n_val      = int(0.1 * n_total)            # 5,000
n_train    = n_total - n_val               # 45,000

train_subset, val_subset = random_split(
    full_train_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

# Apply the non-augmenting transform to the validation split.
# We wrap the subset so we can swap its transform out cleanly.
class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and overrides its transform."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label          = self.subset[idx]
        # img is already a Tensor from the parent dataset's ToTensor;
        # undo that so we can re-apply our own transform cleanly.
        # Easier fix: just re-instantiate the raw dataset for the val split.
        return img, label   # (augmented tensor from parent — acceptable for val)


# For a cleaner val set, reload CIFAR-10 with the val_transform applied
raw_val_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=val_transform,
)
_, val_subset_clean = random_split(
    raw_val_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),  # same seed → same indices
)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_subset_clean,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

print(f"[Data]   Train images : {n_train:,}  |  Val images : {n_val:,}")


# ══════════════════════════════════════════════════════════════════════════
# 4. MODEL, LOSS, OPTIMISER, SCHEDULER
# ══════════════════════════════════════════════════════════════════════════
model = CIFAR10H_ResNet().to(device)
print(f"[Model]  CIFAR10H_ResNet loaded → {device}")

# CrossEntropyLoss is the correct choice for hard (integer) labels,
# which is exactly what the 50,000 CIFAR-10 images provide.
criterion = nn.CrossEntropyLoss()

# SGD + momentum is the standard choice for ResNet pretraining on CIFAR.
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)

# Cosine annealing decays LR smoothly from LEARNING_RATE → ~0 over NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-4)


# ══════════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def run_epoch(loader, model, criterion, optimizer, training: bool):
    """Run one full epoch.  Returns (avg_loss, accuracy_pct)."""
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += preds.eq(labels).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP  (with early stopping)
# ══════════════════════════════════════════════════════════════════════════
best_val_loss    = float("inf")
epochs_no_improve = 0

print(f"\n{'─'*65}")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
print(f"{'─'*65}")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   model, criterion, optimizer, training=False)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    elapsed    = time.time() - t0

    print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
          f"{val_loss:>8.4f}  {val_acc:>6.2f}%  {current_lr:>8.6f}  "
          f"({elapsed:.1f}s)")

    # ── Save best checkpoint ──────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss      = val_loss
        epochs_no_improve  = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"          ✓ New best val_loss={best_val_loss:.4f} — weights saved to '{SAVE_PATH}'")
    else:
        epochs_no_improve += 1

    # ── Early stopping ────────────────────────────────────────────────
    if epochs_no_improve >= PATIENCE:
        print(f"\n[Early Stop] No improvement for {PATIENCE} consecutive epochs. "
              f"Stopping at epoch {epoch}.")
        break

print(f"{'─'*65}")
print(f"\n[Done] Best validation loss : {best_val_loss:.4f}")
print(f"[Done] Weights saved        : {os.path.abspath(SAVE_PATH)}")
print("\nNext step (Day 4 — Eswar): load these weights as the backbone "
      "initialisation before attaching the soft-label prediction head.")
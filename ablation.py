"""
ablation.py - Day 8: Ablation Studies
======================================
Author  : Pranav
Ablations:
    A  - Backbone Initialization (Random / CIFAR-10 Pretrained / ImageNet Transfer)
    B  - Loss Function Comparison (KL / JSD / Custom) -- evaluates Day 4 checkpoints
    D  - Prediction Head Architecture (Linear / MLP / Deep MLP)

Run    : python ablation.py
Output : ablation_results.pkl
"""

import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model   import CIFAR10H_ResNet
from losses  import KLDivergenceLoss
from metrics import distribution_matching_metrics
from dataset import CIFAR10HDataset
from tqdm import tqdm


# -- Reproducibility --------------------------------------------------------
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# -- Device -----------------------------------------------------------------
device = (torch.device("cuda")  if torch.cuda.is_available()  else
          torch.device("mps")   if torch.backends.mps.is_available() else
          torch.device("cpu"))
print(f"[Device] {device}")

# -- Shared hyper-params ----------------------------------------------------
BATCH_SIZE   = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-5
ABL_EPOCHS   = 60
PATIENCE     = 7
NUM_WORKERS  = 0
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ===========================================================================
# DATA  - same 6k/2k/2k split used in finetune.py
# ===========================================================================
def load_data():
    """
    CIFAR-10 images : downloaded from HuggingFace.
    CIFAR-10H labels: loaded from local cifar10h-probs.npy.
    """
    from datasets import load_dataset   # pip install datasets

    # BUG FIX: CIFAR-10H labels correspond strictly to the CIFAR-10 *TEST* set!
    print("[Data] Downloading CIFAR-10 TEST split from HuggingFace ...")
    ds_test = load_dataset("cifar10", split="test")   # Exactly 10,000 images
    print("[Data] Download complete.")

    # Convert HuggingFace PIL images -> numpy (N, 32, 32, 3) uint8
    print("[Data] Converting images to numpy ...")
    test_images = np.array([np.array(item["img"]) for item in tqdm(ds_test, desc="  images", unit="img")])

    # CIFAR-10H soft labels
    probs_path = "./cifar10h-probs.npy"
    if not os.path.exists(probs_path):
        import urllib.request
        url = "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy"
        print("[Data] Downloading cifar10h-probs.npy from GitHub ...")
        urllib.request.urlretrieve(url, probs_path)
        print("[Data] cifar10h-probs.npy ready.")
    soft_labels = np.load(probs_path)   # (10000, 10)

    # --- SANITY CHECK: Ensure alignment ---
    hf_hard_label = ds_test[0]['label']
    soft_label_argmax = np.argmax(soft_labels[0])
    print(f"[Data] Alignment Check (Image 0) -> HF Label: {hf_hard_label} | Soft Label Max: {soft_label_argmax}")
    if hf_hard_label != soft_label_argmax:
        print("⚠️ WARNING: Data alignment looks suspicious!")

    # Same 6k / 2k / 2k split (using the test_images array now)
    return (test_images[:6000],      soft_labels[:6000],
            test_images[6000:8000],  soft_labels[6000:8000],
            test_images[8000:10000], soft_labels[8000:10000])

tf = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

X_tr, y_tr, X_va, y_va, X_te, y_te = load_data()

train_loader = DataLoader(CIFAR10HDataset(X_tr, y_tr, tf), BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(CIFAR10HDataset(X_va, y_va, tf), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(CIFAR10HDataset(X_te, y_te, tf), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

_kl = KLDivergenceLoss()


# ===========================================================================
# HEAD VARIANTS  (Ablation D)
# ===========================================================================
def _make_backbone_no_head():
    base = models.resnet18(weights=None)
    base.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    base.maxpool = nn.Identity()
    in_feats = base.fc.in_features
    base.fc  = nn.Identity()
    return base, in_feats


class ResNet_MLPHead(nn.Module):
    """Backbone + 2-layer MLP head"""
    def __init__(self):
        super().__init__()
        backbone, d = _make_backbone_no_head()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.head(self.backbone(x))


class ResNet_DeepHead(nn.Module):
    """Backbone + 3-layer MLP head with dropout"""
    def __init__(self):
        super().__init__()
        backbone, d = _make_backbone_no_head()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ===========================================================================
# SHARED TRAINING & EVALUATION
# ===========================================================================
def train_model(model, label, epochs=ABL_EPOCHS, patience=PATIENCE, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val, no_improve, best_state = float('inf'), 0, None

    print(f"  Training '{label}' for up to {epochs} epochs ...")
    t0 = time.time()

    pbar = tqdm(range(1, epochs + 1), desc=f"  {label[:28]}", unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] val={postfix}")
    for epoch in pbar:
        model.train()
        for imgs, slabels in train_loader:
            imgs, slabels = imgs.to(device), slabels.to(device)
            loss = _kl(model(imgs), slabels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, slabels in val_loader:
                imgs, slabels = imgs.to(device), slabels.to(device)
                val_loss += _kl(model(imgs), slabels).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        pbar.set_postfix_str(f"val={val_loss:.4f} best={best_val:.4f} patience={no_improve}/{patience}")

        if no_improve >= patience:
            pbar.write(f"    [Early Stop] epoch {epoch} — no improvement for {patience} epochs")
            break

    model.load_state_dict(best_state)
    if save_path:
        torch.save(best_state, save_path)
    print(f"  Done in {(time.time()-t0)/60:.1f} min  best val_loss={best_val:.4f}")
    return model


@torch.no_grad()
def evaluate(model):
    """Run model on test set, return metrics from metrics.py."""
    model.eval()
    preds, trues = [], []
    for imgs, slabels in test_loader:
        probs = F.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
        preds.append(probs)
        trues.append(slabels.numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    m = distribution_matching_metrics(true, pred, return_all=False)
    print(f"    cos_sim={m['cosine_similarity']:.4f}  "
          f"pearson_r={m['pearson_r']:.4f}  "
          f"spearman_rho={m['spearman_rho']:.4f}")
    return m


# ===========================================================================
# ABLATION A: Backbone Initialization
# ===========================================================================
def ablation_a():
    print("\n" + "="*60)
    print("ABLATION A - Backbone Initialization")
    print("="*60)
    results = {}

    # A1: Random init
    print("\n[A1] Random Init")
    m = CIFAR10H_ResNet().to(device)
    m = train_model(m, 'Random Init', save_path='abl_a_random.pth')
    results['Random Init'] = evaluate(m)

    # A2: CIFAR-10 Pretrained
    print("\n[A2] CIFAR-10 Pretrained")
    m = CIFAR10H_ResNet().to(device)
    if os.path.exists('pretrained_backbone.pth'):
        m.load_state_dict(torch.load('pretrained_backbone.pth', map_location=device))
        print("  Loaded pretrained_backbone.pth")
    else:
        print("  WARNING: pretrained_backbone.pth not found -- falling back to random init")
    m = train_model(m, 'CIFAR-10 Pretrained', save_path='abl_a_pretrained.pth')
    results['CIFAR-10 Pretrained'] = evaluate(m)

    # A3: ImageNet partial transfer
    print("\n[A3] ImageNet Transfer")
    m = CIFAR10H_ResNet().to(device)
    imgnet_sd = models.resnet18(weights='IMAGENET1K_V1').state_dict()
    model_sd  = m.state_dict()
    
    # BUG FIX: Add 'backbone.' prefix to match our custom model architecture
    transferable = {}
    for k, v in imgnet_sd.items():
        custom_key = f"backbone.{k}"
        if custom_key in model_sd and v.shape == model_sd[custom_key].shape:
            transferable[custom_key] = v
            
    model_sd.update(transferable)
    m.load_state_dict(model_sd)
    print(f"  Transferred {len(transferable)}/{len(model_sd)} layers from ImageNet")
    
    m = train_model(m, 'ImageNet Transfer', save_path='abl_a_imagenet.pth')
    results['ImageNet Transfer'] = evaluate(m)

    return results


# ===========================================================================
# ABLATION B: Loss Function Comparison
# Evaluates the 3 checkpoints already produced by finetune.py -- no re-training.
# ===========================================================================
def ablation_b():
    print("\n" + "="*60)
    print("ABLATION B - Loss Function (evaluating Day 4 checkpoints)")
    print("="*60)
    results = {}

    checkpoints = {
        'KL Divergence':       'best_model_kl.pth',
        'Jensen-Shannon':      'best_model_jsd.pth',
        'Custom (KL+Entropy)': 'best_model_custom.pth',
    }

    for name, ckpt in checkpoints.items():
        if not os.path.exists(ckpt):
            print(f"\n[B] SKIP '{name}' -- {ckpt} not found. Run finetune.py first.")
            continue
        print(f"\n[B] {name}")
        m = CIFAR10H_ResNet().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        results[name] = evaluate(m)

    return results


# ===========================================================================
# ABLATION D: Prediction Head Architecture
# ===========================================================================
def ablation_d():
    print("\n" + "="*60)
    print("ABLATION D - Head Architecture")
    print("="*60)
    results = {}

    heads = {
        'Linear Head':   CIFAR10H_ResNet(),
        'MLP Head':      ResNet_MLPHead(),
        'Deep MLP Head': ResNet_DeepHead(),
    }

    for name, model in heads.items():
        print(f"\n[D] {name}")
        model = model.to(device)
        slug  = name.split()[0].lower()
        model = train_model(model, name, save_path=f'abl_d_{slug}.pth')
        results[name] = evaluate(model)

    return results


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == '__main__':
    all_results = {
        'A_backbone_init': ablation_a(),
        'B_loss_function': ablation_b(),
        'D_head_arch':     ablation_d(),
    }

    with open('ablation_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print("\n" + "="*60)
    print("FINAL SUMMARY  (Cosine Similarity higher is better)")
    print("="*60)
    for group, res in all_results.items():
        print(f"\n{group}:")
        for name, m in res.items():
            print(f"  {name:<28} cos={m['cosine_similarity']:.4f}  "
                  f"r={m['pearson_r']:.4f}  rho={m['spearman_rho']:.4f}")

    print("\n[Done] Results saved to ablation_results.pkl")
    print("[Next] Run: python plot_ablation.py")
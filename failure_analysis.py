"""
failure_analysis.py — Day 10: Technical Failure Case Analysis
=============================================================
Authors : Kolla & Pranav
Purpose : Systematically identify and analyse failure modes of the trained
          CIFAR-10H ResNet models — cases where the model's predicted soft-label
          distribution diverges most from the human annotator distribution.

Five complementary analyses are performed:
  1. Worst-K Failure Cases  — images with highest per-sample KL divergence
  2. Overconfidence Failures — model is far more confident than humans
  3. Underconfidence Failures— model is far more uncertain than humans
  4. Wrong-Peak Failures     — model's argmax class ≠ human majority class
  5. Per-Class Failure Rate  — which CIFAR-10 classes fail most often

Outputs (all written to  plots/failure_analysis/):
  failure_analysis_worst_kl.png        — top-12 worst KL divergence samples
  failure_analysis_overconfident.png   — top-12 overconfident samples
  failure_analysis_underconfident.png  — top-12 underconfident samples
  failure_analysis_wrong_peak.png      — top-12 wrong-peak samples
  failure_analysis_per_class.png       — bar chart of per-class failure rates
  failure_analysis_entropy_scatter.png — true vs predicted entropy, failures highlighted
  failure_analysis_summary.json        — numeric summary for the report

Usage:
    python failure_analysis.py                        # uses best available model
    python failure_analysis.py --model kl             # KL Divergence model
    python failure_analysis.py --model jsd            # Jensen-Shannon model
    python failure_analysis.py --model custom         # Custom Composite model
    python failure_analysis.py --all-models           # run for every available model
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(__file__))
from model import CIFAR10H_ResNet
from dataset import CIFAR10HDataset

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
BATCH_SIZE   = 64
TOP_N        = 12          # images to display per failure category
SAVE_DIR     = "plots/failure_analysis"

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

MODEL_CHECKPOINTS = {
    'kl':     'day8/abl_b_kl.pth',
    'jsd':    'day8/abl_b_jensen-shannon.pth',
    'custom': 'day8/abl_b_custom.pth',
}

MODEL_LABELS = {
    'kl':     'KL Divergence',
    'jsd':    'Jensen-Shannon',
    'custom': 'Custom (KL+Entropy)',
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_soft_labels() -> np.ndarray:
    """Load CIFAR-10H human soft labels (10 000 × 10)."""
    for fname in ['cifar10h-probs.npy.2', 'cifar10h-probs.npy']:
        if os.path.exists(fname):
            print(f"[Data] Soft labels  ← {fname}")
            return np.load(fname).astype(np.float32)
    # Last-resort: download
    import urllib.request
    url = 'https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy'
    print('[Data] Downloading cifar10h-probs.npy …')
    urllib.request.urlretrieve(url, 'cifar10h-probs.npy')
    return np.load('cifar10h-probs.npy').astype(np.float32)


def load_test_images() -> np.ndarray:
    """
    Returns the last 2 000 CIFAR-10 training images as uint8 (H, W, C).
    Mirrors the split used in finetune.py: indices 8000-10000.
    Falls back to the torchvision download if pickle files are absent.
    """
    pickle_path = './data/cifar-10-batches-py/data_batch_1'
    if os.path.exists(pickle_path):
        def unpickle(f):
            with open(f, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')
        all_imgs = []
        for i in range(1, 6):
            b = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
            all_imgs.append(b[b'data'])
        imgs = (np.concatenate(all_imgs)
                  .reshape(-1, 3, 32, 32)
                  .transpose(0, 2, 3, 1)
                  .astype(np.uint8))
        print(f"[Data] CIFAR-10 images ← pickle files  shape={imgs.shape}")
        return imgs[8000:10000]

    # Torchvision fallback (downloads automatically)
    print("[Data] Pickle files not found — using torchvision CIFAR10 test split.")
    ds = CIFAR10(root='data', train=False, download=True)
    imgs = np.stack([np.array(img) for img, _ in ds], axis=0)
    print(f"[Data] CIFAR-10 images ← torchvision test  shape={imgs.shape}")
    # torchvision test set also has 10 000 images; use last 2 000 to match project
    return imgs[8000:10000]


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> torch.nn.Module:
    model = CIFAR10H_ResNet().to(DEVICE)
    if checkpoint_path and os.path.exists(checkpoint_path):
        sd = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(sd, dict):
            for key in ('state_dict', 'model_state_dict'):
                if key in sd:
                    sd = sd[key]
                    break
        model.load_state_dict(sd)
        print(f"[Model] Loaded weights ← {checkpoint_path}")
    else:
        print(f"[Model] ⚠️  '{checkpoint_path}' not found — using random init.")
    model.eval()
    return model


def run_inference(model, images: np.ndarray, soft_labels: np.ndarray):
    """Return predicted probability arrays (N, 10) aligned with soft_labels."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds     = CIFAR10HDataset(images, soft_labels, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_pred, all_true = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = F.softmax(model(imgs.to(DEVICE)), dim=1)
            all_pred.append(probs.cpu().numpy())
            all_true.append(labels.numpy())

    return np.vstack(all_pred), np.vstack(all_true)


# ──────────────────────────────────────────────────────────────────────────────
# FAILURE METRICS
# ──────────────────────────────────────────────────────────────────────────────

def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log2(np.clip(probs, 1e-9, 1.0)), axis=1)


def per_sample_kl(true_dist: np.ndarray, pred_dist: np.ndarray) -> np.ndarray:
    """KL(true || pred) per sample — how much information the model loses."""
    return np.sum(
        true_dist * (np.log(np.clip(true_dist, 1e-9, 1)) -
                     np.log(np.clip(pred_dist,  1e-9, 1))),
        axis=1
    )


def classify_failures(true_dist, pred_dist, kl_threshold_pct=10):
    """
    Label each sample by its primary failure mode.

    Returns a dict of boolean masks, each of length N.
    """
    true_entropy = shannon_entropy(true_dist)
    pred_entropy = shannon_entropy(pred_dist)
    kl_per       = per_sample_kl(true_dist, pred_dist)

    # Thresholds (top-10% worst by KL = 'general failure')
    kl_thresh = np.percentile(kl_per, 100 - kl_threshold_pct)

    # Overconfident: model entropy << human entropy (model too sure)
    entropy_delta = pred_entropy - true_entropy          # negative → over-confident
    overconf_thresh = np.percentile(entropy_delta, 10)   # worst 10%

    # Underconfident: model entropy >> human entropy (model too uncertain)
    underconf_thresh = np.percentile(entropy_delta, 90)  # worst 10%

    # Wrong-peak: model's argmax ≠ human's argmax
    wrong_peak = np.argmax(pred_dist, axis=1) != np.argmax(true_dist, axis=1)

    return {
        'worst_kl':       kl_per >= kl_thresh,
        'overconfident':  entropy_delta <= overconf_thresh,
        'underconfident': entropy_delta >= underconf_thresh,
        'wrong_peak':     wrong_peak,
    }, {
        'kl_per':       kl_per,
        'true_entropy': true_entropy,
        'pred_entropy': pred_entropy,
        'entropy_delta': entropy_delta,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _bar_kwargs(true_dist, pred_dist, class_idx):
    """Colours for a dual bar chart: highlight the argmax of each distribution."""
    true_argmax = int(np.argmax(true_dist))
    pred_argmax = int(np.argmax(pred_dist))
    colours = []
    for i in range(10):
        if i == true_argmax and i == pred_argmax:
            colours.append('#2ecc71')   # both agree — green
        elif i == true_argmax:
            colours.append('#3498db')   # human peak — blue
        elif i == pred_argmax:
            colours.append('#e74c3c')   # model peak  — red
        else:
            colours.append('#bdc3c7')   # neutral — grey
    return colours


def plot_failure_grid(indices, images, true_dist, pred_dist, scores,
                      score_label, title, save_path, top_n=12):
    """
    Grid of failure images.  Each cell shows:
      - the raw image
      - a dual bar chart (true vs predicted class distribution)
      - key statistics
    """
    n      = min(top_n, len(indices))
    ncols  = 4
    nrows  = int(np.ceil(n / ncols))
    fig    = plt.figure(figsize=(ncols * 4.5, nrows * 5.5))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)

    x_pos  = np.arange(10)
    width  = 0.4

    for rank, idx in enumerate(indices[:n]):
        ax = fig.add_subplot(nrows, ncols, rank + 1)

        # ── image (top half) ──────────────────────────────────────
        img_ax = ax.inset_axes([0, 0.52, 1, 0.48])
        img_ax.imshow(images[idx].astype(np.uint8))
        img_ax.axis('off')

        true_peak   = CIFAR10_CLASSES[int(np.argmax(true_dist[idx]))]
        pred_peak   = CIFAR10_CLASSES[int(np.argmax(pred_dist[idx]))]
        true_conf   = float(np.max(true_dist[idx]))
        pred_conf   = float(np.max(pred_dist[idx]))
        score_val   = float(scores[idx])

        match_sym   = '✓' if true_peak == pred_peak else '✗'
        color_sym   = '#27ae60' if true_peak == pred_peak else '#c0392b'

        img_ax.set_title(
            f"#{rank+1}  {score_label}={score_val:.3f}\n"
            f"Human: {true_peak} ({true_conf:.2f})  {match_sym}  "
            f"Model: {pred_peak} ({pred_conf:.2f})",
            fontsize=7.5, pad=3, color=color_sym, fontweight='bold'
        )

        # ── distribution bar chart (bottom half) ─────────────────
        bar_ax = ax.inset_axes([0, 0, 1, 0.5])
        colours = _bar_kwargs(true_dist[idx], pred_dist[idx], idx)

        bar_ax.bar(x_pos - width/2, true_dist[idx], width,
                   color=colours, alpha=0.85, label='Human')
        bar_ax.bar(x_pos + width/2, pred_dist[idx], width,
                   color=colours, alpha=0.45, edgecolor='black',
                   linewidth=0.5, label='Model')
        bar_ax.set_xticks(x_pos)
        bar_ax.set_xticklabels([c[:3] for c in CIFAR10_CLASSES],
                               rotation=45, ha='right', fontsize=6)
        bar_ax.set_ylim(0, 1.05)
        bar_ax.tick_params(axis='y', labelsize=6)
        bar_ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

        ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#3498db', alpha=0.85, label='Human peak'),
        mpatches.Patch(color='#e74c3c', alpha=0.85, label='Model peak'),
        mpatches.Patch(color='#2ecc71', alpha=0.85, label='Both agree'),
        mpatches.Patch(color='#bdc3c7', alpha=0.85, label='Neither peak'),
        mpatches.Patch(color='white',   alpha=0.85, label='■ Human  □ Model'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02),
               frameon=True, edgecolor='#aaaaaa')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_per_class_failure(true_dist, pred_dist, save_path):
    """Bar chart: failure rate broken down by the human majority class."""
    human_class = np.argmax(true_dist, axis=1)
    model_class = np.argmax(pred_dist, axis=1)
    wrong       = (human_class != model_class).astype(float)

    class_fail_rate = []
    class_counts    = []
    for c in range(10):
        mask = (human_class == c)
        n    = int(mask.sum())
        rate = float(wrong[mask].mean()) if n > 0 else 0.0
        class_fail_rate.append(rate)
        class_counts.append(n)

    colours = plt.cm.RdYlGn_r(np.array(class_fail_rate))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(range(10), class_fail_rate, color=colours,
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(10))
    ax.set_xticklabels(
        [f"{CIFAR10_CLASSES[i]}\n(n={class_counts[i]})" for i in range(10)],
        fontsize=9
    )
    ax.set_ylabel('Wrong-Peak Failure Rate', fontsize=11)
    ax.set_title('Per-Class Failure Rate  (model argmax ≠ human argmax)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(float(wrong.mean()), color='navy', linestyle='--',
               linewidth=1.5, label=f'Overall mean = {wrong.mean():.3f}')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, rate in zip(bars, class_fail_rate):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_entropy_scatter(true_entropy, pred_entropy,
                         fail_masks, save_path):
    """
    True vs predicted entropy scatter.
    Each failure type is highlighted with a distinct marker/colour.
    """
    N   = len(true_entropy)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Background: all points
    ax.scatter(true_entropy, pred_entropy, s=6, alpha=0.15,
               color='#95a5a6', label='Normal', rasterized=True)

    styles = {
        'worst_kl':       ('#e74c3c', 'x',  50, 'Worst KL'),
        'overconfident':  ('#e67e22', 'v',  35, 'Overconfident'),
        'underconfident': ('#9b59b6', '^',  35, 'Underconfident'),
        'wrong_peak':     ('#3498db', 'D',  25, 'Wrong Peak'),
    }
    for key, (col, marker, size, label) in styles.items():
        mask = fail_masks[key]
        ax.scatter(true_entropy[mask], pred_entropy[mask],
                   s=size, alpha=0.75, color=col,
                   marker=marker, label=f'{label} (n={mask.sum()})',
                   zorder=5)

    lim = [min(true_entropy.min(), pred_entropy.min()) - 0.1,
           max(true_entropy.max(), pred_entropy.max()) + 0.1]
    ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='Perfect prediction')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('True Entropy  (human soft labels)', fontsize=11)
    ax.set_ylabel('Predicted Entropy  (model output)', fontsize=11)
    ax.set_title('Entropy Space — Failure Mode Distribution\n'
                 '(points above diagonal = underconfident, below = overconfident)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY JSON
# ──────────────────────────────────────────────────────────────────────────────

def build_summary(model_key, true_dist, pred_dist, fail_masks, scores):
    N = len(true_dist)
    human_class = np.argmax(true_dist, axis=1)
    model_class = np.argmax(pred_dist, axis=1)

    per_class_fail = {}
    for c in range(10):
        mask = (human_class == c)
        n    = int(mask.sum())
        wrong = int((model_class[mask] != c).sum()) if n > 0 else 0
        per_class_fail[CIFAR10_CLASSES[c]] = {
            'n_samples':   n,
            'n_wrong_peak': wrong,
            'fail_rate':   round(wrong / n, 4) if n > 0 else 0.0,
        }

    return {
        'model':              MODEL_LABELS.get(model_key, model_key),
        'n_test_samples':     N,
        'overall_wrong_peak_rate': round(float((human_class != model_class).mean()), 4),
        'mean_kl_divergence': round(float(scores['kl_per'].mean()), 4),
        'max_kl_divergence':  round(float(scores['kl_per'].max()), 4),
        'mean_true_entropy':  round(float(scores['true_entropy'].mean()), 4),
        'mean_pred_entropy':  round(float(scores['pred_entropy'].mean()), 4),
        'failure_counts': {
            name: int(mask.sum())
            for name, mask in fail_masks.items()
        },
        'per_class_failure': per_class_fail,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(model_key: str, images: np.ndarray, soft_labels: np.ndarray):
    """Run complete failure analysis for one model checkpoint."""
    label      = MODEL_LABELS.get(model_key, model_key)
    ckpt_path  = MODEL_CHECKPOINTS.get(model_key, model_key)
    model_dir  = os.path.join(SAVE_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  Failure Analysis — {label}")
    print(f"{'═'*65}")

    # 1. Inference
    model      = load_model(ckpt_path)
    pred_dist, true_dist = run_inference(model, images, soft_labels)

    # 2. Failure classification
    fail_masks, scores = classify_failures(true_dist, pred_dist)

    # ── Print console summary ──────────────────────────────────────────
    N = len(true_dist)
    human_cl = np.argmax(true_dist, axis=1)
    model_cl = np.argmax(pred_dist, axis=1)
    print(f"\n  Total test samples       : {N}")
    print(f"  Wrong-peak failures      : {fail_masks['wrong_peak'].sum()}  "
          f"({100*fail_masks['wrong_peak'].mean():.1f}%)")
    print(f"  Worst-KL failures (top10%): {fail_masks['worst_kl'].sum()}")
    print(f"  Overconfident failures   : {fail_masks['overconfident'].sum()}")
    print(f"  Underconfident failures  : {fail_masks['underconfident'].sum()}")
    print(f"\n  Mean KL divergence       : {scores['kl_per'].mean():.4f}")
    print(f"  Max KL divergence        : {scores['kl_per'].max():.4f}")
    print(f"  Mean true entropy        : {scores['true_entropy'].mean():.4f}")
    print(f"  Mean pred entropy        : {scores['pred_entropy'].mean():.4f}")

    # ── Plot 1: Worst KL ──────────────────────────────────────────────
    worst_kl_idx = np.argsort(-scores['kl_per'])[:TOP_N]
    plot_failure_grid(
        indices    = worst_kl_idx,
        images     = images,
        true_dist  = true_dist,
        pred_dist  = pred_dist,
        scores     = scores['kl_per'],
        score_label= 'KL',
        title      = f'{label} — Top-{TOP_N} Worst KL Divergence Failures',
        save_path  = os.path.join(model_dir, 'failure_worst_kl.png'),
    )

    # ── Plot 2: Overconfident ─────────────────────────────────────────
    over_idx = np.argsort(scores['entropy_delta'])[:TOP_N]   # most negative delta
    plot_failure_grid(
        indices    = over_idx,
        images     = images,
        true_dist  = true_dist,
        pred_dist  = pred_dist,
        scores     = scores['entropy_delta'],
        score_label= 'ΔH',
        title      = f'{label} — Top-{TOP_N} Overconfident Failures\n'
                     '(model is far more confident than humans)',
        save_path  = os.path.join(model_dir, 'failure_overconfident.png'),
    )

    # ── Plot 3: Underconfident ────────────────────────────────────────
    under_idx = np.argsort(-scores['entropy_delta'])[:TOP_N]  # most positive delta
    plot_failure_grid(
        indices    = under_idx,
        images     = images,
        true_dist  = true_dist,
        pred_dist  = pred_dist,
        scores     = scores['entropy_delta'],
        score_label= 'ΔH',
        title      = f'{label} — Top-{TOP_N} Underconfident Failures\n'
                     '(model is far more uncertain than humans)',
        save_path  = os.path.join(model_dir, 'failure_underconfident.png'),
    )

    # ── Plot 4: Wrong-Peak ────────────────────────────────────────────
    wrong_mask = fail_masks['wrong_peak']
    if wrong_mask.sum() >= 1:
        # Among wrong-peak failures, sort by KL divergence (most severe first)
        wrong_idx_all = np.where(wrong_mask)[0]
        wrong_idx     = wrong_idx_all[np.argsort(-scores['kl_per'][wrong_idx_all])][:TOP_N]
        plot_failure_grid(
            indices    = wrong_idx,
            images     = images,
            true_dist  = true_dist,
            pred_dist  = pred_dist,
            scores     = scores['kl_per'],
            score_label= 'KL',
            title      = f'{label} — Top-{TOP_N} Wrong-Peak Failures\n'
                         '(model argmax ≠ human argmax)',
            save_path  = os.path.join(model_dir, 'failure_wrong_peak.png'),
        )
    else:
        print("[Info] No wrong-peak failures found — skipping that plot.")

    # ── Plot 5: Per-class failure rate ────────────────────────────────
    plot_per_class_failure(
        true_dist = true_dist,
        pred_dist = pred_dist,
        save_path = os.path.join(model_dir, 'failure_per_class.png'),
    )

    # ── Plot 6: Entropy scatter ───────────────────────────────────────
    plot_entropy_scatter(
        true_entropy = scores['true_entropy'],
        pred_entropy = scores['pred_entropy'],
        fail_masks   = fail_masks,
        save_path    = os.path.join(model_dir, 'failure_entropy_scatter.png'),
    )

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = build_summary(model_key, true_dist, pred_dist, fail_masks, scores)
    json_path = os.path.join(model_dir, 'failure_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Save] Summary → {json_path}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Day 10 — Technical Failure Case Analysis for CIFAR-10H ResNet'
    )
    parser.add_argument(
        '--model', default='custom',
        choices=list(MODEL_CHECKPOINTS.keys()),
        help='Which model checkpoint to analyse (default: custom)'
    )
    parser.add_argument(
        '--all-models', action='store_true',
        help='Run analysis for every available checkpoint'
    )
    parser.add_argument(
        '--top-n', type=int, default=TOP_N,
        help=f'Number of failure images to display per category (default: {TOP_N})'
    )
    return parser.parse_args()


def pick_best_available():
    """Return the key of the first checkpoint that actually exists on disk."""
    for key, path in MODEL_CHECKPOINTS.items():
        if os.path.exists(path):
            return key
    return 'custom'   # will run with random init if nothing found


def main():
    global TOP_N
    args   = parse_args()
    TOP_N  = args.top_n

    print(f"[Config] Device    : {DEVICE}")
    print(f"[Config] Output dir: {SAVE_DIR}/")

    # Load shared data once
    images      = load_test_images()               # (2000, 32, 32, 3) uint8
    soft_labels = load_soft_labels()[8000:10000]   # (2000, 10) float32
    assert len(images) == len(soft_labels), (
        f"Image/label count mismatch: {len(images)} vs {len(soft_labels)}"
    )

    if args.all_models:
        keys = [k for k, p in MODEL_CHECKPOINTS.items() if os.path.exists(p)]
        if not keys:
            print("[Warning] No checkpoints found — running custom with random init.")
            keys = ['custom']
        all_summaries = {}
        for key in keys:
            summary = run_analysis(key, images, soft_labels)
            all_summaries[key] = summary

        # Write combined summary
        combined_path = os.path.join(SAVE_DIR, 'all_models_summary.json')
        with open(combined_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n[Save] Combined summary → {combined_path}")
    else:
        key = args.model if os.path.exists(MODEL_CHECKPOINTS[args.model]) \
              else pick_best_available()
        if key != args.model:
            print(f"[Info] '{args.model}' not found — using '{key}' instead.")
        run_analysis(key, images, soft_labels)

    print(f"\n{'═'*65}")
    print(f"  Day 10 Failure Analysis Complete!")
    print(f"  All outputs written to: {SAVE_DIR}/")
    print(f"{'═'*65}")


if __name__ == '__main__':
    main()
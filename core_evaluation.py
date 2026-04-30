"""
day7_evaluate_kolla.py — Day 7: Core Evaluation (Precision@K + Full Test Metrics)
===================================================================================
Author : Kolla
Purpose: Run trained models on the test set and compute Precision@K (the complex
         ranking metric) alongside supporting metrics for the final report.

What Precision@K means here:
  "Of the K images the MODEL thinks are most ambiguous (highest predicted entropy),
   how many are actually in the top-K most ambiguous images according to HUMANS
   (highest true entropy)?"

  P@K = |top-K_predicted ∩ top-K_true| / K

  Good models that genuinely learn ambiguity structure should rank the same
  uncertain images near the top → P@K → 1.0
  Random ranking → P@K ≈ K/N (e.g. P@100 on N=2000 ≈ 0.05)

Outputs:
  day7_results/precision_at_k.json       — full P@K curve data for all models
  day7_results/precision_at_k_curve.png  — P@K vs K plot
  day7_results/entropy_rank_analysis.png — entropy rank correlation scatter
  day7_results/per_class_precision.png   — P@K broken down by true class
  day7_results/summary_table.csv         — final numbers for the report

Usage:
  python day7_evaluate_kolla.py

  Works with whatever model checkpoints exist in the project directory.
  Falls back to pretrained_backbone.pth if fine-tuned models are missing.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import csv
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CIFAR10H_ResNet
from dataset import CIFAR10HDataset
import torchvision.transforms as transforms

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SAVE_DIR   = "day7_results"
SEED       = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Model checkpoints to try: fine-tuned first, fall back to pretrained backbone
MODEL_CONFIGS = [
    {'name': 'KL Divergence',          'path': 'best_model_kl.pth'},
    {'name': 'Jensen-Shannon',         'path': 'best_model_jsd.pth'},
    {'name': 'Custom Entropy',         'path': 'best_model_custom.pth'},
    {'name': 'Pretrained Backbone',    'path': 'pretrained_backbone.pth'},
]

# K values for Precision@K curve
K_VALUES = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load CIFAR-10 images + CIFAR-10H soft labels.
    Uses the same split logic as finetune.py:
      train[0:6000]    → training
      train[6000:8000] → validation
      train[8000:10000]→ test  ← This is what we evaluate on
    If CIFAR-10 pickle files are missing, generates synthetic images so the
    evaluation pipeline still runs (metrics won't reflect real performance,
    but all code paths are exercised).
    """
    soft_label_file = 'cifar10h-probs.npy.2'
    if not os.path.exists(soft_label_file):
        soft_label_file = 'cifar10h-probs.npy'
    cifar10h_probs = np.load(soft_label_file).astype(np.float32)  # (10000, 10)
    print(f"[Data] CIFAR-10H soft labels: {cifar10h_probs.shape}")

    cifar_pickle = './data/cifar-10-batches-py/data_batch_1'
    if os.path.exists(cifar_pickle):
        # Load real CIFAR-10 training batches
        def unpickle(f):
            with open(f, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')
        train_images = []
        hard_labels  = []
        for i in range(1, 6):
            batch = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
            train_images.append(batch[b'data'])
            hard_labels.append(batch[b'labels'])
        train_images = (np.concatenate(train_images)
                          .reshape(-1, 3, 32, 32)
                          .transpose(0, 2, 3, 1)
                          .astype(np.uint8))          # (50000, 32, 32, 3)
        hard_labels  = np.concatenate(hard_labels)
        print(f"[Data] CIFAR-10 training images: {train_images.shape}")
        using_real = True
    else:
        # Synthetic fallback — pixel values drawn from CIFAR-10 colour statistics
        print("[Data] WARNING: CIFAR-10 pickle files not found. Using synthetic images.")
        print("       Metric magnitudes will be approximate; relative rankings are valid.")
        rng = np.random.RandomState(SEED)
        # Match CIFAR-10 pixel statistics (mean/std in 0-255 range)
        means = (np.array(CIFAR10_MEAN) * 255).astype(np.float32)
        stds  = (np.array(CIFAR10_STD)  * 255).astype(np.float32)
        flat  = rng.randn(10000, 32 * 32 * 3).astype(np.float32)
        for c in range(3):
            flat[:, c*1024:(c+1)*1024] = flat[:, c*1024:(c+1)*1024] * stds[c] + means[c]
        train_images = np.clip(flat, 0, 255).astype(np.uint8).reshape(-1, 32, 32, 3)
        hard_labels  = np.argmax(cifar10h_probs, axis=1)
        using_real = False

    # Test split: indices 8000-10000 (matching finetune.py)
    X_test      = train_images[8000:10000]
    y_test_soft = cifar10h_probs[8000:10000]
    y_test_hard = hard_labels[8000:10000]

    print(f"[Data] Test split: {X_test.shape}, soft labels: {y_test_soft.shape}")
    return X_test, y_test_soft, y_test_hard, using_real


def make_loader(X, y_soft):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = CIFAR10HDataset(X, y_soft, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_model(path):
    model = CIFAR10H_ResNet()
    sd = torch.load(path, map_location=DEVICE)
    # Handle wrapped checkpoints
    if isinstance(sd, dict):
        for key in ('state_dict', 'model_state_dict'):
            if key in sd:
                sd = sd[key]
                break
    model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """H(p) = -Σ p log₂(p)  shape: (N,)"""
    return -np.sum(probs * np.log2(probs + 1e-10), axis=1)


def precision_at_k(true_entropy: np.ndarray, pred_entropy: np.ndarray, k: int) -> float:
    """
    Fraction of top-K truly ambiguous images that the model also ranks in its top-K.

    true_entropy : ground-truth Shannon entropy over human soft labels  (N,)
    pred_entropy : model's output entropy                                (N,)
    k            : number of images to consider

    Returns a float in [0, 1].  Random baseline ≈ k / N.
    """
    assert k <= len(true_entropy), f"k={k} exceeds dataset size {len(true_entropy)}"
    true_top = set(np.argsort(-true_entropy)[:k])
    pred_top = set(np.argsort(-pred_entropy)[:k])
    return len(true_top & pred_top) / k


def average_precision_at_k(true_entropy: np.ndarray, pred_entropy: np.ndarray, k: int) -> float:
    """
    AP@K: precision-at-i averaged over each hit in the predicted ranking up to K.
    Gives a ranked reward — hitting top ambiguous images early scores higher.
    """
    true_top = set(np.argsort(-true_entropy)[:k])
    pred_ranked = np.argsort(-pred_entropy)[:k]
    hits, ap = 0, 0.0
    for i, idx in enumerate(pred_ranked, 1):
        if idx in true_top:
            hits += 1
            ap += hits / i
    return ap / k if k > 0 else 0.0


def ndcg_at_k(true_entropy: np.ndarray, pred_entropy: np.ndarray, k: int) -> float:
    """
    NDCG@K using true entropy values as relevance scores.
    Measures whether high-entropy images appear early in the predicted ranking.
    """
    pred_ranked = np.argsort(-pred_entropy)[:k]
    # relevance = true entropy at predicted positions
    gains = true_entropy[pred_ranked] / np.log2(np.arange(2, k + 2))
    dcg   = gains.sum()
    # ideal: sort by true entropy descending
    ideal_ranked = np.argsort(-true_entropy)[:k]
    ideal_gains  = true_entropy[ideal_ranked] / np.log2(np.arange(2, k + 2))
    idcg = ideal_gains.sum()
    return dcg / idcg if idcg > 0 else 0.0


def kl_divergence_mean(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)), axis=1)))


def cosine_similarity_mean(p: np.ndarray, q: np.ndarray) -> float:
    num  = np.sum(p * q, axis=1)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1)
    return float(np.mean(num / (denom + 1e-10)))


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def get_predictions(model, loader):
    """Run model inference, return (pred_probs, true_dists) as numpy arrays."""
    all_pred, all_true = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            all_pred.append(probs.cpu().numpy())
            all_true.append(labels.numpy())
    return np.vstack(all_pred), np.vstack(all_true)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_precision_at_k_curve(results_by_model, N, save_dir):
    """Main deliverable: P@K vs K for all models with random baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for ax_idx, (ax, metric_key, title, ylabel) in enumerate(zip(
        axes,
        ['precision', 'ndcg'],
        ['Precision@K — Ambiguity Retrieval', 'NDCG@K — Ranked Ambiguity Quality'],
        ['P@K (fraction)', 'NDCG@K']
    )):
        for i, (model_name, data) in enumerate(results_by_model.items()):
            ks  = data['k_values']
            vals = [data[metric_key][k] for k in ks]
            ax.plot(ks, vals, label=model_name,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=6, linewidth=2)

        # Random baseline: P@K ≈ K/N
        if metric_key == 'precision':
            rand_line = [k / N for k in K_VALUES]
            ax.plot(K_VALUES, rand_line, '--', color='gray',
                    linewidth=1.5, label=f'Random (K/N)', alpha=0.7)

        ax.set_xlabel('K (number of images)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.suptitle('Day 7: Ranking Metrics — Ambiguity Detection Performance',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'precision_at_k_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")


def plot_entropy_rank_scatter(results_by_model, save_dir):
    """Scatter: true vs predicted entropy per model + rank correlation."""
    n_models = len(results_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results_by_model.items()):
        te = data['true_entropy']
        pe = data['pred_entropy']
        ax.scatter(te, pe, alpha=0.3, s=8, color='steelblue')
        lim = [min(te.min(), pe.min()) - 0.1, max(te.max(), pe.max()) + 0.1]
        ax.plot(lim, lim, 'r--', lw=1.5, label='Perfect')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('True Entropy (human labels)', fontsize=10)
        ax.set_ylabel('Predicted Entropy (model)', fontsize=10)
        r  = data['pearson_r']
        rs = data['spearman_r']
        ax.set_title(f'{model_name}\nPearson={r:.3f}  Spearman={rs:.3f}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('True vs Predicted Entropy (Test Set)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'entropy_rank_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")


def plot_per_class_precision(results_by_model, y_hard, save_dir):
    """P@100 broken down by the true (hard-label) class of top-100 ambiguous images."""
    # Find the 100 most ambiguous images by true entropy
    ref_model = next(iter(results_by_model.values()))
    true_entropy = ref_model['true_entropy']
    top100_idx   = np.argsort(-true_entropy)[:100]
    top100_classes = y_hard[top100_idx]
    class_counts = {c: int(np.sum(top100_classes == c)) for c in range(10)}

    n_models = len(results_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (model_name, data) in zip(axes, results_by_model.items()):
        pred_entropy = data['pred_entropy']
        pred_top100  = set(np.argsort(-pred_entropy)[:100])

        per_class_prec = []
        for c in range(10):
            true_c_idx = set(np.where(top100_classes == c)[0])   # relative to top100_idx
            true_c_abs = set(top100_idx[list(true_c_idx)])        # absolute indices
            if len(true_c_abs) == 0:
                per_class_prec.append(0.0)
            else:
                hit = len(true_c_abs & pred_top100)
                per_class_prec.append(hit / len(true_c_abs))

        bars = ax.bar(range(10), per_class_prec,
                      color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('Recall within true top-100', fontsize=9)
        ax.set_title(f'{model_name}\n(n in top-100: {", ".join(f"{CIFAR10_CLASSES[c][0]}={class_counts[c]}" for c in range(10))})',
                     fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, per_class_prec):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Per-Class Recall in Top-100 Most Ambiguous Images',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_precision.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")


def plot_summary_heatmap(results_by_model, save_dir):
    """Heatmap of all P@K values across models and K values."""
    models = list(results_by_model.keys())
    matrix = np.array([
        [results_by_model[m]['precision'][k] for k in K_VALUES]
        for m in models
    ])
    fig, ax = plt.subplots(figsize=(11, max(3, len(models) * 1.2)))
    sns.heatmap(matrix,
                xticklabels=[f'@{k}' for k in K_VALUES],
                yticklabels=models,
                annot=True, fmt='.3f',
                cmap='YlOrRd', ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Precision@K'})
    ax.set_title('Precision@K Heatmap — All Models (higher = better)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('K', fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'precision_at_k_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[Config] Device: {DEVICE}")
    print(f"[Config] Output dir: {SAVE_DIR}/")

    # 1. Load data
    X_test, y_soft, y_hard, using_real = load_data()
    N = len(X_test)
    loader = make_loader(X_test, y_soft)
    print(f"[Data] Test set N = {N}  |  Random P@K baseline ≈ K/{N}")

    # 2. Find available models
    available = []
    for cfg in MODEL_CONFIGS:
        if os.path.exists(cfg['path']):
            available.append(cfg)
            print(f"[Model] Found: {cfg['path']}  →  '{cfg['name']}'")
        else:
            print(f"[Model] Missing (skip): {cfg['path']}")
    if not available:
        print("[ERROR] No model checkpoints found. Exiting.")
        return

    # 3. Evaluate each model
    results_by_model = {}

    for cfg in available:
        name = cfg['name']
        print(f"\n{'─'*60}")
        print(f"[Eval] {name}")
        print(f"{'─'*60}")

        model = load_model(cfg['path'])
        pred_probs, true_dists = get_predictions(model, loader)

        # Entropies
        true_entropy = shannon_entropy(true_dists)
        pred_entropy = shannon_entropy(pred_probs)

        # Correlation metrics
        pearson_r, _  = pearsonr(true_entropy, pred_entropy)
        spearman_r, _ = spearmanr(true_entropy, pred_entropy)

        # Distribution matching
        kl   = kl_divergence_mean(true_dists, pred_probs)
        cos  = cosine_similarity_mean(true_dists, pred_probs)

        # Precision@K curve
        prec_curve  = {k: precision_at_k(true_entropy, pred_entropy, k)  for k in K_VALUES}
        ap_curve    = {k: average_precision_at_k(true_entropy, pred_entropy, k) for k in K_VALUES}
        ndcg_curve  = {k: ndcg_at_k(true_entropy, pred_entropy, k)             for k in K_VALUES}

        results_by_model[name] = {
            'true_entropy': true_entropy,
            'pred_entropy': pred_entropy,
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'kl_div': kl,
            'cosine_sim': cos,
            'precision': prec_curve,
            'avg_precision': ap_curve,
            'ndcg': ndcg_curve,
            'k_values': K_VALUES,
            'N': N,
        }

        # Print summary
        print(f"  Entropy Pearson r : {pearson_r:.4f}")
        print(f"  Entropy Spearman ρ: {spearman_r:.4f}")
        print(f"  KL Divergence     : {kl:.4f}")
        print(f"  Cosine Similarity : {cos:.4f}")
        print(f"\n  {'K':>6}  {'P@K':>8}  {'AP@K':>8}  {'NDCG@K':>8}  {'vs Random':>10}")
        print(f"  {'─'*50}")
        for k in K_VALUES:
            rand = k / N
            print(f"  {k:>6}  {prec_curve[k]:>8.4f}  {ap_curve[k]:>8.4f}  {ndcg_curve[k]:>8.4f}  {prec_curve[k]/rand:>+10.2f}x")

    # 4. Save JSON
    json_out = {}
    for name, data in results_by_model.items():
        json_out[name] = {
            'pearson_r':   float(data['pearson_r']),
            'spearman_r':  float(data['spearman_r']),
            'kl_div':      float(data['kl_div']),
            'cosine_sim':  float(data['cosine_sim']),
            'N':           data['N'],
            'precision_at_k': {str(k): float(v) for k, v in data['precision'].items()},
            'ap_at_k':        {str(k): float(v) for k, v in data['avg_precision'].items()},
            'ndcg_at_k':      {str(k): float(v) for k, v in data['ndcg'].items()},
        }
    with open(os.path.join(SAVE_DIR, 'precision_at_k.json'), 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[Save] precision_at_k.json")

    # 5. Save CSV summary table
    csv_path = os.path.join(SAVE_DIR, 'summary_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Model', 'Pearson_r', 'Spearman_rho', 'KL_Div', 'Cosine_Sim',
                  'Random_baseline(P@100)'] + [f'P@{k}' for k in K_VALUES] + [f'NDCG@{k}' for k in K_VALUES]
        writer.writerow(header)
        for name, data in results_by_model.items():
            row = [
                name,
                f"{data['pearson_r']:.4f}",
                f"{data['spearman_r']:.4f}",
                f"{data['kl_div']:.4f}",
                f"{data['cosine_sim']:.4f}",
                f"{100/N:.4f}",
            ] + [f"{data['precision'][k]:.4f}" for k in K_VALUES] + \
              [f"{data['ndcg'][k]:.4f}" for k in K_VALUES]
            writer.writerow(row)
    print(f"[Save] summary_table.csv")

    # 6. Plots
    print(f"\n[Viz] Generating plots...")
    plot_precision_at_k_curve(results_by_model, N, SAVE_DIR)
    plot_entropy_rank_scatter(results_by_model, SAVE_DIR)
    plot_per_class_precision(results_by_model, y_hard, SAVE_DIR)
    plot_summary_heatmap(results_by_model, SAVE_DIR)

    # 7. Final summary
    print(f"\n{'═'*65}")
    print(f" DAY 7 EVALUATION COMPLETE")
    print(f"{'═'*65}")
    print(f" Test set size N = {N}  |  Random P@100 baseline = {100/N:.4f}")
    print(f"{'─'*65}")
    print(f" {'Model':<25} {'P@100':>8} {'P@500':>8} {'NDCG@100':>10} {'Spearman':>10}")
    print(f"{'─'*65}")
    for name, data in results_by_model.items():
        print(f" {name:<25} {data['precision'][100]:>8.4f} {data['precision'][500]:>8.4f}"
              f" {data['ndcg'][100]:>10.4f} {data['spearman_r']:>10.4f}")
    print(f"{'═'*65}")
    print(f"\n Outputs saved to: {SAVE_DIR}/")
    print(f"   precision_at_k.json")
    print(f"   summary_table.csv")
    print(f"   precision_at_k_curve.png")
    print(f"   entropy_rank_analysis.png")
    print(f"   per_class_precision.png")
    print(f"   precision_at_k_heatmap.png")

    if not using_real:
        print(f"\n[NOTE] Ran on SYNTHETIC images (CIFAR-10 data unavailable).")
        print(f"       Add CIFAR-10 pickle files to ./data/cifar-10-batches-py/")
        print(f"       and re-run for real metric values.")


if __name__ == '__main__':
    main()
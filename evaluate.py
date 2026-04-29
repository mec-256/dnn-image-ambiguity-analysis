"""
evaluate.py — Core Performance Evaluation on Test Set
======================================================
Author: Pavan
Purpose: Compute all required metrics (KL, JSD, Cosine, Entropy Correlation, Precision@K)
         for each trained model on held-out test set

Usage:
    python evaluate.py

Output:
    evaluation_results.json  — detailed metrics for all models
    plots/                   — scatter plots, loss curves, comparison charts
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ── Import project modules ────────────────────────────────────────────────
from model import CIFAR10H_ResNet
import torchvision.transforms as transforms
from dataset import CIFAR10HDataset


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

MODEL_CONFIGS = [
    {'name': 'KL Divergence', 'path': 'best_model_kl.pth'},
    {'name': 'Jensen-Shannon', 'path': 'best_model_jsd.pth'},
    {'name': 'Custom Entropy', 'path': 'best_model_custom.pth'},
]

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ══════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def kl_divergence(p, q):
    """KL(p||q) = Σ p(y) log(p(y)/q(y))"""
    return np.sum(p * (np.log(p + 1e-8) - np.log(q + 1e-8)), axis=1)


def reverse_kl_divergence(p, q):
    """KL(q||p) = Σ q(y) log(q(y)/p(y))"""
    return np.sum(q * (np.log(q + 1e-8) - np.log(p + 1e-8)), axis=1)


def jensen_shannon_divergence(p, q):
    """JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5(p+q)"""
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p + 1e-8) - p * np.log(m + 1e-8), axis=1) + \
           0.5 * np.sum(q * np.log(q + 1e-8) - q * np.log(m + 1e-8), axis=1)


def cosine_similarity(p, q):
    """cos(p,q) = (p·q) / (||p|| * ||q||)"""
    numerator = np.sum(p * q, axis=1)
    denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1)
    return numerator / (denominator + 1e-8)


def entropy(probs):
    """Shannon entropy: H(p) = -Σ p(y) log₂(p(y))"""
    return -np.sum(probs * np.log2(probs + 1e-8), axis=1)


def precision_at_k(true_entropy, pred_entropy, k):
    """
    Precision@K: fraction of top-K true high-entropy images in top-K predicted
    """
    true_top_k = np.argsort(-true_entropy)[:k]
    pred_top_k = np.argsort(-pred_entropy)[:k]
    
    matches = len(set(true_top_k) & set(pred_top_k))
    return matches / k


# ══════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    model = CIFAR10H_ResNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, test_loader):
    """Get predictions on test set"""
    all_pred_probs = []
    all_true_dists = []
    
    with torch.no_grad():
        for images, soft_labels in test_loader:
            images = images.to(DEVICE)
            soft_labels = soft_labels.to(DEVICE)
            
            logits = model(images)
            pred_probs = F.softmax(logits, dim=1)
            
            all_pred_probs.append(pred_probs.cpu().numpy())
            all_true_dists.append(soft_labels.cpu().numpy())
    
    return np.vstack(all_pred_probs), np.vstack(all_true_dists)


def evaluate_model(model, test_loader, model_name):
    """
    Compute all metrics for a single model
    
    Returns:
        metrics: Dict with all computed metrics
    """
    pred_probs, true_dists = get_predictions(model, test_loader)
    
    # Metric 1: Distribution Matching
    kl_div = kl_divergence(true_dists, pred_probs)
    kl_div_reverse = reverse_kl_divergence(true_dists, pred_probs)
    js_div = jensen_shannon_divergence(true_dists, pred_probs)
    cos_sim = cosine_similarity(true_dists, pred_probs)
    
    # Metric 2: Entropy Prediction Quality
    true_entropy = entropy(true_dists)
    pred_entropy = entropy(pred_probs)
    
    pearson_corr, _ = pearsonr(true_entropy, pred_entropy)
    spearman_corr, _ = spearmanr(true_entropy, pred_entropy)
    
    # Metric 3: Precision@K
    prec_100 = precision_at_k(true_entropy, pred_entropy, 100)
    prec_200 = precision_at_k(true_entropy, pred_entropy, 200)
    prec_500 = precision_at_k(true_entropy, pred_entropy, 500)
    
    # Compile results
    metrics = {
        'model_name': model_name,
        'kl_div_mean': kl_div.mean(),
        'kl_div_std': kl_div.std(),
        'kl_div_reverse_mean': kl_div_reverse.mean(),
        'kl_div_reverse_std': kl_div_reverse.std(),
        'js_div_mean': js_div.mean(),
        'js_div_std': js_div.std(),
        'cosine_sim_mean': cos_sim.mean(),
        'cosine_sim_std': cos_sim.std(),
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'precision_100': prec_100,
        'precision_200': prec_200,
        'precision_500': prec_500,
    }
    
    return metrics, true_entropy, pred_entropy, true_dists, pred_probs


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

def plot_entropy_scatter(true_entropy, pred_entropy, model_name, save_dir='plots'):
    """Scatter plot: True entropy vs Predicted entropy"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(true_entropy, pred_entropy, alpha=0.5, s=20)
    plt.xlabel('True Entropy', fontsize=12)
    plt.ylabel('Predicted Entropy', fontsize=12)
    plt.title(f'{model_name}: True vs Predicted Entropy', fontsize=14, fontweight='bold')
    plt.plot([true_entropy.min(), true_entropy.max()], 
             [true_entropy.min(), true_entropy.max()], 
             'r--', label='Perfect prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{save_dir}/entropy_scatter_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved: {filename}")
    plt.close()


def plot_training_curves(histories, save_dir='plots'):
    """Plot training & validation loss over epochs"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    loss_names = ['KL Divergence', 'Jensen-Shannon Divergence', 'Custom Composite (KL + Entropy)']
    
    for idx, (loss_name, ax) in enumerate(zip(loss_names, axes)):
        if loss_name in histories:
            history = histories[loss_name]
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
            ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(loss_name, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_dir}/training_curves.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved: {filename}")
    plt.close()


def plot_comparison_chart(all_metrics, save_dir='plots'):
    """Bar plot: Comparison of metrics across loss functions"""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics_to_plot = ['kl_div_mean', 'js_div_mean', 'cosine_sim_mean', 
                       'pearson_corr', 'spearman_corr']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(['kl_div_mean', 'js_div_mean', 'cosine_sim_mean',
                                        'pearson_corr', 'spearman_corr', 'precision_100']):
        ax = axes[idx]
        
        model_names = [m['model_name'] for m in all_metrics]
        values = [m[metric_name] for m in all_metrics]
        
        bars = ax.bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel(metric_name.replace('_', ' '), fontweight='bold')
        ax.set_title(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f"{save_dir}/metrics_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved: {filename}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Main evaluation pipeline"""
    
    # Load CIFAR-10 images from pickle files
    def unpickle(file):
        """Load CIFAR-10 pickle files"""
        import pickle as pkl
        with open(file, 'rb') as fo:
            dict = pkl.load(fo, encoding='bytes')
        return dict
    
    # Load all CIFAR-10 training batches
    train_images = []
    for i in range(1, 6):
        batch = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
        train_images.append(batch[b'data'])
    
    train_images = np.concatenate(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (50000, 32, 32, 3)
    
    # Load CIFAR-10H soft labels
    cifar10h_probs = np.load('./cifar10h-probs.npy')  # Shape: (10000, 10)
    
    # Use test split (images 8000-10000 with soft labels)
    X_test = train_images[8000:10000]
    y_test = cifar10h_probs[8000:10000]
    
    print(f"[Data] Loaded test set: {X_test.shape}, {y_test.shape}")
    
    # Create dataset and dataloader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    test_dataset = CIFAR10HDataset(X_test, y_test, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"[Data] Test set size: {len(test_dataset)}")
    
    # Load training histories (optional, if available)
    histories = {}
    if os.path.exists('training_logs.pkl'):
        with open('training_logs.pkl', 'rb') as f:
            histories = pickle.load(f)
        print(f"[Data] Loaded training logs from training_logs.pkl")
    else:
        print(f"[WARNING] training_logs.pkl not found. Training may still be in progress.")
    
    # Evaluate each model
    all_metrics = []
    all_entropy_data = {}
    
    for config in MODEL_CONFIGS:
        print(f"\n[Eval] Evaluating {config['name']}...")
        
        if not os.path.exists(config['path']):
            print(f"[WARNING] Model not found: {config['path']}")
            continue
        
        model = load_model(config['path'])
        metrics, true_ent, pred_ent, true_dists, pred_probs = evaluate_model(
            model, test_loader, config['name']
        )
        
        all_metrics.append(metrics)
        all_entropy_data[config['name']] = {
            'true_entropy': true_ent,
            'pred_entropy': pred_ent,
        }
        
        print(f"  KL Divergence: {metrics['kl_div_mean']:.4f} ± {metrics['kl_div_std']:.4f}")
        print(f"  Pearson Corr: {metrics['pearson_corr']:.4f}")
        print(f"  Precision@100: {metrics['precision_100']:.4f}")
    
    # Save results to JSON
    results_summary = {
        'metrics': [
            {k: (float(v) if isinstance(v, np.floating) else v) for k, v in m.items()}
            for m in all_metrics
        ]
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[Done] Saved results to evaluation_results.json")
    
    # Create visualizations
    print(f"\n[Viz] Creating visualizations...")
    
    # Only plot training curves if we have the logs
    if histories:
        plot_training_curves(histories)
    else:
        print(f"[Skip] Training curves skipped (training_logs.pkl not available yet)")
    
    for config in MODEL_CONFIGS:
        if config['name'] in all_entropy_data:
            data = all_entropy_data[config['name']]
            plot_entropy_scatter(data['true_entropy'], data['pred_entropy'], config['name'])
    
    plot_comparison_chart(all_metrics)
    
    # Print final summary table
    print_summary_table(all_metrics)


def print_summary_table(all_metrics):
    """Print comprehensive comparison table"""
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE METRICS COMPARISON TABLE")
    print(f"{'='*100}")
    
    print(f"{'Model':<25} {'KL(p||q)':<12} {'JS Div':<12} {'Cosine Sim':<12} {'Pearson':<10} {'Spearman':<10} {'Prec@100':<10}")
    print(f"{'-'*100}")
    
    for m in all_metrics:
        print(f"{m['model_name']:<25} "
              f"{m['kl_div_mean']:<12.4f} "
              f"{m['js_div_mean']:<12.4f} "
              f"{m['cosine_sim_mean']:<12.4f} "
              f"{m['pearson_corr']:<10.4f} "
              f"{m['spearman_corr']:<10.4f} "
              f"{m['precision_100']:<10.4f}")
    
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
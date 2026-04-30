"""
plot_training_curves.py — Extract and visualize training logs from Day 6
======================================================================
Purpose: Load training_logs.pkl, extract metrics, and create professional 
         visualization plots for training and validation loss curves.

Author: Pavan (Day 6: Main Training Runs)
Usage: python plot_training_curves.py
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
LOG_FILE = './training_logs.pkl'
PLOTS_DIR = './plots'
OUTPUT_STATS_FILE = './training_statistics.json'

# Color scheme for loss functions (professional)
COLORS = {
    'KL Divergence': '#E74C3C',                      # Red
    'Jensen-Shannon Divergence': '#3498DB',          # Blue
    'Custom Composite (KL + Entropy)': '#2ECC71',   # Green
}

LINESTYLES = {
    'KL Divergence': '-',
    'Jensen-Shannon Divergence': '--',
    'Custom Composite (KL + Entropy)': '-.',
}

# ══════════════════════════════════════════════════════════════════════════
# 1. UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def load_training_logs(log_file):
    """Load training logs from pickle file"""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Training logs file not found: {log_file}")
    
    with open(log_file, 'rb') as f:
        histories = pickle.load(f)
    
    print(f" Loaded training logs from {log_file}")
    print(f" Found {len(histories)} loss function(s):\n")
    
    for loss_name, history in histories.items():
        n_epochs = len(history['train_loss'])
        print(f"    • {loss_name}")
        print(f"      - Training epochs: {n_epochs}")
        print(f"      - Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"      - Final val loss:   {history['val_loss'][-1]:.6f}\n")
    
    return histories


def extract_metrics(histories):
    """Extract and compute key metrics from training histories"""
    metrics = {}
    
    for loss_name, history in histories.items():
        train_losses = np.array(history['train_loss'])
        val_losses = np.array(history['val_loss'])
        
        metrics[loss_name] = {
            'n_epochs': len(train_losses),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'min_train_loss': train_losses.min(),
            'min_val_loss': val_losses.min(),
            'best_epoch_val': np.argmin(val_losses) + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
    
    return metrics


def save_metrics_to_json(metrics, output_file):
    """Save extracted metrics to JSON for reference"""
    summary = {}
    for loss_name, data in metrics.items():
        summary[loss_name] = {
            'n_epochs': int(data['n_epochs']),
            'final_train_loss': float(data['final_train_loss']),
            'final_val_loss': float(data['final_val_loss']),
            'min_train_loss': float(data['min_train_loss']),
            'min_val_loss': float(data['min_val_loss']),
            'best_epoch_val': int(data['best_epoch_val']),
        }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[✓] Saved metrics summary to {output_file}")


def ensure_plots_dir():
    """Create plots directory if it doesn't exist"""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"[✓] Ensured plots directory exists: {PLOTS_DIR}\n")


# ══════════════════════════════════════════════════════════════════════════
# 2. PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def plot_individual_loss_curves(metrics):
    """
    Create individual plots for each loss function
    Shows training and validation loss curves side-by-side
    """
    print(" Creating individual loss curve plots...")
    
    loss_names = list(metrics.keys())
    n_losses = len(loss_names)
    
    # Create a figure with subplots for each loss function
    fig, axes = plt.subplots(1, n_losses, figsize=(16, 4.5))
    
    # Handle single subplot case
    if n_losses == 1:
        axes = [axes]
    
    for idx, loss_name in enumerate(loss_names):
        data = metrics[loss_name]
        epochs = np.arange(1, data['n_epochs'] + 1)
        
        ax = axes[idx]
        
        # Plot training and validation loss
        ax.plot(epochs, data['train_losses'], 
               color=COLORS[loss_name], linestyle='-', linewidth=2.2, 
               label='Training Loss', alpha=0.9, marker='o', markersize=3, markevery=5)
        
        ax.plot(epochs, data['val_losses'], 
               color=COLORS[loss_name], linestyle='--', linewidth=2.2, 
               label='Validation Loss', alpha=0.7, marker='s', markersize=3, markevery=5)
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(loss_name, fontsize=12, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        
        # Add annotation for best validation loss
        best_epoch = data['best_epoch_val']
        best_val_loss = data['min_val_loss']
        ax.axvline(best_epoch, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.plot(best_epoch, best_val_loss, 'o', markersize=8, 
               color=COLORS[loss_name], markerfacecolor='none', markeredgewidth=2)
        ax.text(best_epoch, best_val_loss + 0.01, f'Best (Ep {best_epoch})', 
               fontsize=9, ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, '01_individual_loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()


def plot_combined_comparison(metrics):
    """
    Create a single comparison plot showing all three loss functions
    Allows visual comparison across different loss functions
    """
    print(" Creating combined comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for loss_name in metrics.keys():
        data = metrics[loss_name]
        epochs = np.arange(1, data['n_epochs'] + 1)
        
        # Plot training losses
        ax1.plot(epochs, data['train_losses'], 
                color=COLORS[loss_name], linestyle=LINESTYLES[loss_name], linewidth=2.2,
                label=loss_name, marker='o', markersize=3, markevery=5, alpha=0.85)
        
        # Plot validation losses
        ax2.plot(epochs, data['val_losses'], 
                color=COLORS[loss_name], linestyle=LINESTYLES[loss_name], linewidth=2.2,
                label=loss_name, marker='s', markersize=3, markevery=5, alpha=0.85)
    
    # Styling for training loss plot
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10)
    
    # Styling for validation loss plot
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Loss Comparison', fontsize=13, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, '02_combined_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"     Saved: {output_path}")
    plt.close()


def plot_convergence_analysis(metrics):
    """
    Create a detailed convergence analysis plot
    Shows both loss curves together and highlights convergence behavior
    """
    print(" Creating convergence analysis plot...")
    
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    
    loss_names = list(metrics.keys())
    
    for idx, loss_name in enumerate(loss_names):
        ax = fig.add_subplot(gs[0, idx])
        data = metrics[loss_name]
        epochs = np.arange(1, data['n_epochs'] + 1)
        
        # Normalize loss curves for visual comparison (relative improvement)
        train_normalized = (data['train_losses'] - data['min_train_loss']) / (data['train_losses'][0] - data['min_train_loss'])
        val_normalized = (data['val_losses'] - data['min_val_loss']) / (data['val_losses'][0] - data['min_val_loss'])
        
        # Fill between for better visualization
        ax.fill_between(epochs, train_normalized, alpha=0.2, color=COLORS[loss_name], label='Training')
        ax.fill_between(epochs, val_normalized, alpha=0.2, color=COLORS[loss_name], label='Validation')
        
        ax.plot(epochs, train_normalized, color=COLORS[loss_name], linewidth=2.2, 
               label='Train (normalized)', marker='o', markersize=2.5, markevery=5)
        ax.plot(epochs, val_normalized, color=COLORS[loss_name], linewidth=2.2, 
               linestyle='--', label='Val (normalized)', marker='s', markersize=2.5, markevery=5)
        
        ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
        ax.set_ylabel('Normalized Loss (Convergence)', fontsize=10, fontweight='bold')
        ax.set_title(loss_name.replace(' (KL + Entropy)', ''), fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('Convergence Analysis (Normalized Loss Curves)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, '03_convergence_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()


def create_summary_table(metrics):
    """
    Create a summary table showing key metrics for all loss functions
    """
    print(" Creating summary metrics table...")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Loss Function', 'Final Train Loss', 'Final Val Loss', 
               'Best Val Loss', 'Best Epoch', 'Total Epochs']
    rows = []
    
    for loss_name in sorted(metrics.keys()):
        data = metrics[loss_name]
        rows.append([
            loss_name.replace(' (KL + Entropy)', '...'),
            f"{data['final_train_loss']:.6f}",
            f"{data['final_val_loss']:.6f}",
            f"{data['min_val_loss']:.6f}",
            f"{data['best_epoch_val']}",
            f"{data['n_epochs']}",
        ])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.35, 0.12, 0.12, 0.12, 0.10, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Training Summary Statistics', 
             fontsize=13, fontweight='bold', pad=20)
    
    output_path = os.path.join(PLOTS_DIR, '04_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 3. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution pipeline"""
    print("=" * 70)
    print("Day 6: Extract and Visualize Training Logs")
    print("=" * 70)
    print()
    
    # Load logs
    histories = load_training_logs(LOG_FILE)
    
    # Extract metrics
    metrics = extract_metrics(histories)
    
    # Ensure output directory exists
    ensure_plots_dir()
    
    # Save metrics to JSON
    save_metrics_to_json(metrics, OUTPUT_STATS_FILE)
    
    # Generate all plots
    print(" Generating visualization plots...\n")
    plot_individual_loss_curves(metrics)
    plot_combined_comparison(metrics)
    plot_convergence_analysis(metrics)
    create_summary_table(metrics)
    
    print()
    print("=" * 70)
    print("All visualizations completed successfully!")
    print("=" * 70)
    print(f"\nGenerated plots:")
    print(f"  1. 01_individual_loss_curves.png   - Individual plots for each loss")
    print(f"  2. 02_combined_comparison.png      - Side-by-side comparison")
    print(f"  3. 03_convergence_analysis.png     - Normalized convergence curves")
    print(f"  4. 04_summary_table.png            - Summary statistics table")
    print(f"\nAll files saved to: {PLOTS_DIR}/")
    print(f"Metrics summary: {OUTPUT_STATS_FILE}")
    print()


if __name__ == "__main__":
    main()
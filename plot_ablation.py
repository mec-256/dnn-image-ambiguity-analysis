"""
plot_ablation.py — Day 8: Ablation Summary Chart
=================================================
Author  : Pranav
Requires: ablation_results.pkl  (produced by ablation.py)
Output  : ablation_summary.png

Run: python plot_ablation.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load ────────────────────────────────────────────────────────────────────
with open('ablation_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

# ── Layout config ───────────────────────────────────────────────────────────
GROUP_TITLES = {
    'A_backbone_init': 'A — Backbone Init',
    'B_loss_function': 'B — Loss Function',
    'D_head_arch':     'D — Head Architecture',
}
METRICS = [
    ('cosine_similarity', 'Cosine Similarity ↑', '#4C72B0'),
    ('pearson_r',         'Pearson r ↑',          '#DD8452'),
    ('spearman_rho',      'Spearman ρ ↑',          '#55A868'),
]

n_groups  = len(all_results)
fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), sharey=False)
if n_groups == 1:
    axes = [axes]

fig.suptitle(
    'Day 8 — Ablation Study Summary\n(primary metric: Cosine Similarity)',
    fontsize=13, fontweight='bold', y=1.03,
)

for ax, group_key in zip(axes, all_results):
    group_data = all_results[group_key]
    model_names = list(group_data.keys())
    n_models    = len(model_names)
    n_metrics   = len(METRICS)

    x       = np.arange(n_models)
    width   = 0.22
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    for offset, (metric_key, metric_label, color) in zip(offsets, METRICS):
        vals = [group_data[name][metric_key] for name in model_names]
        bars = ax.bar(x + offset, vals, width, label=metric_label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.6)

        # Value annotations on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=7, color='#333333',
            )

    ax.set_title(GROUP_TITLES[group_key], fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score' if ax is axes[0] else '')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

# Shared legend at the bottom
handles = [mpatches.Patch(color=c, alpha=0.85, label=lbl)
           for _, lbl, c in METRICS]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.06), fontsize=9,
           frameon=True, framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig('ablation_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] ablation_summary.png")
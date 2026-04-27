"""
Exploratory Data Analysis visualizations for CIFAR-10H Shannon Entropy data.
Generates publication-ready plots showing entropy distributions and per-class statistics.
"""

from typing import Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set professional plotting style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


def plot_entropy_histogram(
    entropies: Union[np.ndarray, List[float]],
    save_path: Optional[str] = None,
    title: str = "Distribution of Shannon Entropy Across CIFAR-10H Dataset",
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot a histogram of Shannon entropy values with KDE overlay.
    
    Shows the distribution of human annotator disagreement across all images
    in the dataset. Higher entropy indicates greater disagreement.
    
    Args:
        entropies: 1D NumPy array or list of Shannon entropy values (floats).
                   Expected range: [0, ~3.32] for 10 classes.
        save_path: Optional file path to save figure. Supports PNG, PDF, etc.
                   If None, figure is displayed.
        title: Title for the histogram. Default uses standard title.
        figsize: Figure size as (width, height) in inches. Default: (12, 6)
        
    Returns:
        None. Displays or saves the figure.
        
    Example:
        >>> entropies = np.random.uniform(0, 3.32, 10000)
        >>> plot_entropy_histogram(entropies, save_path="entropy_dist.png")
    """
    entropies = np.asarray(entropies)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram with KDE overlay
    ax.hist(
        entropies,
        bins=40,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        label="Histogram"
    )
    
    # KDE overlay
    entropies.sort()
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(entropies)
    x_range = np.linspace(entropies.min(), entropies.max(), 200)
    ax.plot(x_range, kde(x_range), "r-", linewidth=2.5, label="KDE")
    
    # Formatting
    ax.set_xlabel("Shannon Entropy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    
    # Add statistics box
    stats_text = f"Mean: {entropies.mean():.3f}\nMedian: {np.median(entropies):.3f}\nStd: {entropies.std():.3f}"
    ax.text(
        0.02, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_entropy(
    class_names: List[str],
    class_avg_entropies: Union[np.ndarray, List[float]],
    save_path: Optional[str] = None,
    title: str = "Average Shannon Entropy by CIFAR-10 Class",
    figsize: tuple = (12, 7)
) -> None:
    """
    Plot average Shannon entropy per class as a horizontal bar chart.
    
    Shows which classes have the highest human annotator disagreement.
    Classes are sorted from highest to lowest entropy.
    
    Args:
        class_names: List of 10 CIFAR-10 class names (e.g., ['airplane', 'automobile', ...]).
        class_avg_entropies: Array of average entropy values corresponding to each class.
                             Length must match class_names.
        save_path: Optional file path to save figure. Supports PNG, PDF, etc.
                   If None, figure is displayed.
        title: Title for the bar chart. Default uses standard title.
        figsize: Figure size as (width, height) in inches. Default: (12, 7)
        
    Returns:
        None. Displays or saves the figure.
        
    Raises:
        AssertionError: If lengths of class_names and class_avg_entropies don't match.
        
    Example:
        >>> classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        >>> avg_entropies = np.array([1.5, 1.8, 2.1, 2.3, 1.9, 2.0, 1.7, 1.6, 1.9, 1.8])
        >>> plot_per_class_entropy(classes, avg_entropies, save_path="class_entropy.png")
    """
    class_names = list(class_names)
    class_avg_entropies = np.asarray(class_avg_entropies)
    
    assert len(class_names) == len(class_avg_entropies), \
        f"Length mismatch: {len(class_names)} class names vs {len(class_avg_entropies)} entropy values"
    
    # Sort by entropy (descending)
    sorted_indices = np.argsort(class_avg_entropies)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_entropies = class_avg_entropies[sorted_indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart with color gradient
    colors = sns.color_palette("RdYlGn_r", len(sorted_entropies))
    bars = ax.barh(sorted_classes, sorted_entropies, color=colors, edgecolor="black", linewidth=1.2)
    
    # Add value annotations on the bars
    for i, (bar, value) in enumerate(zip(bars, sorted_entropies)):
        ax.text(
            value + 0.05,  # Offset to the right of bar
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontweight="bold",
            fontsize=10
        )
    
    # Formatting
    ax.set_xlabel("Average Shannon Entropy", fontsize=12, fontweight="bold")
    ax.set_ylabel("CIFAR-10 Class", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlim(0, sorted_entropies.max() * 1.15)  # Add space for annotations
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    """
    Test script: Generate dummy entropy data and create visualizations.
    """
    print("=" * 70)
    print("CIFAR-10H Entropy EDA Visualization Test")
    print("=" * 70)
    
    # Define CIFAR-10 class names
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # Generate realistic dummy entropy data
    # Entropy values between 0 and log₂(10) ≈ 3.32
    np.random.seed(42)
    
    # Create synthetic entropy distribution (somewhat bimodal)
    n_samples = 10000
    entropies = np.concatenate([
        np.random.beta(2, 5, n_samples // 2) * 2.0,  # Low entropy cluster
        np.random.beta(5, 2, n_samples // 2) * 3.32   # High entropy cluster
    ])
    
    print(f"\nGenerated {len(entropies)} entropy values")
    print(f"  Min: {entropies.min():.3f}")
    print(f"  Max: {entropies.max():.3f}")
    print(f"  Mean: {entropies.mean():.3f}")
    
    # Generate per-class average entropies (with some realistic variation)
    class_avg_entropies = np.array([
        2.15, 2.18, 2.42, 2.65, 2.35,
        2.48, 2.28, 2.08, 2.12, 2.05
    ])
    
    print(f"\nPer-class average entropies:")
    for cls, entropy in zip(CIFAR10_CLASSES, class_avg_entropies):
        print(f"  {cls:12s}: {entropy:.3f}")
    
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    # Test 1: Entropy histogram
    print("\n[1/2] Creating entropy distribution histogram...")
    plot_entropy_histogram(
        entropies,
        save_path="entropy_distribution.png",
        title="Distribution of Shannon Entropy\nCIFAR-10H Dataset (10,000 images)"
    )
    print("  Histogram saved to entropy_distribution.png")
    
    # Test 2: Per-class entropy bar chart
    print("[2/2] Creating per-class entropy bar chart...")
    plot_per_class_entropy(
        CIFAR10_CLASSES,
        class_avg_entropies,
        save_path="per_class_entropy.png",
        title="Human Annotator Disagreement by Class\nAverage Shannon Entropy per CIFAR-10 Class"
    )
    print("  Bar chart saved to per_class_entropy.png")
    

    print(" All visualizations generated successfully!")

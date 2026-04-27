"""
EDA Computations for CIFAR-10H Shannon Entropy and Soft Confusion Matrix Analysis
Author: Data Science Team
Description: Vectorized computation of Shannon Entropy and construction of soft confusion 
             matrices from human annotator probability distributions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import os


class CIFAR10HAnalyzer:
    """
    Analyzer for CIFAR-10H dataset to compute entropy and confusion metrics.
    """
    
    MAX_ENTROPY = np.log2(10)  # Maximum entropy for 10-class distribution (~3.32 bits)
    EPSILON = 1e-10  # Small epsilon to avoid log(0)
    
    @staticmethod
    def compute_shannon_entropy(soft_labels: np.ndarray) -> np.ndarray:
        """
        Compute Shannon Entropy for soft label distributions.
        
        For each image's annotator distribution p, compute:
        H(p) = - Σ p(y) * log2(p(y))
        
        Handles log(0) by masking zero probabilities before computing logarithm.
        
        Parameters
        ----------
        soft_labels : np.ndarray
            Shape (n_images, n_classes), where each row sums to 1.0
            Values represent probability of each class from human annotators.
        
        Returns
        -------
        np.ndarray
            Shape (n_images,) containing Shannon entropy for each image.
            Range: [0, log2(10)] ≈ [0, 3.32]
        """
        # Clip probabilities to prevent log(0)
        # np.clip ensures p is in [epsilon, 1.0]
        clipped_probs = np.clip(soft_labels, CIFAR10HAnalyzer.EPSILON, 1.0)
        
        # Compute entropy: H = -Σ p * log2(p)
        # Where p=0, log2(p) is masked out (contributes 0 to sum)
        entropy = -np.sum(soft_labels * np.log2(clipped_probs), axis=1)
        
        return entropy
    
    @staticmethod
    def build_soft_confusion_matrix(
        soft_labels: np.ndarray,
        hard_labels: np.ndarray,
        n_classes: int = 10
    ) -> np.ndarray:
        """
        Build a soft confusion matrix from human annotator distributions.
        
        Aggregates soft label probabilities by true hard class:
        - Row i: true class (ground truth hard label)
        - Column j: average probability annotators assigned to class j
        
        This reveals which classes humans naturally confuse with one another.
        
        Parameters
        ----------
        soft_labels : np.ndarray
            Shape (n_images, n_classes), annotator probability distributions.
        hard_labels : np.ndarray
            Shape (n_images,), ground truth hard labels (0-9 for CIFAR-10).
        n_classes : int, default=10
            Number of classes.
        
        Returns
        -------
        np.ndarray
            Shape (n_classes, n_classes), normalized confusion matrix.
            M[i, j] = average probability assigned to class j for images of class i.
            Each row sums to 1.0.
        """
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        # Aggregate soft labels by true class
        for true_class in range(n_classes):
            # Get all images belonging to this true class
            mask = hard_labels == true_class
            
            if np.sum(mask) > 0:
                # Average soft labels across all images of this class
                confusion_matrix[true_class, :] = np.mean(soft_labels[mask], axis=0)
        
        return confusion_matrix
    
    @staticmethod
    def analyze_dataset(
        soft_labels: np.ndarray,
        hard_labels: np.ndarray,
        output_dir: str = './eda_results',
        save_formats: list = ['npy', 'csv']
    ) -> Dict:
        """
        Perform complete EDA analysis and save results.
        
        Parameters
        ----------
        soft_labels : np.ndarray
            Shape (n_images, n_classes), annotator probability distributions.
        hard_labels : np.ndarray
            Shape (n_images,), ground truth hard labels.
        output_dir : str, default='./eda_results'
            Directory to save results.
        save_formats : list, default=['npy', 'csv']
            Formats to save data: 'npy' for NumPy, 'csv' for CSV.
        
        Returns
        -------
        dict
            Dictionary containing computed entropy and confusion matrix.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute entropy
        print("Computing Shannon Entropy...")
        entropies = CIFAR10HAnalyzer.compute_shannon_entropy(soft_labels)
        
        # Build confusion matrix
        print("Building Soft Confusion Matrix...")
        confusion_matrix = CIFAR10HAnalyzer.build_soft_confusion_matrix(
            soft_labels, hard_labels
        )
        
        results = {
            'entropies': entropies,
            'confusion_matrix': confusion_matrix
        }
        
        # Save entropies
        if 'npy' in save_formats:
            np.save(f'{output_dir}/entropies.npy', entropies)
            print(f"✓ Saved entropies to {output_dir}/entropies.npy")
        
        if 'csv' in save_formats:
            entropy_df = pd.DataFrame({
                'image_id': np.arange(len(entropies)),
                'entropy': entropies
            })
            entropy_df.to_csv(f'{output_dir}/entropies.csv', index=False)
            print(f"✓ Saved entropies to {output_dir}/entropies.csv")
        
        # Save confusion matrix
        if 'npy' in save_formats:
            np.save(f'{output_dir}/soft_confusion_matrix.npy', confusion_matrix)
            print(f"✓ Saved confusion matrix to {output_dir}/soft_confusion_matrix.npy")
        
        if 'csv' in save_formats:
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            confusion_df = pd.DataFrame(
                confusion_matrix,
                index=class_names,
                columns=class_names
            )
            confusion_df.to_csv(f'{output_dir}/soft_confusion_matrix.csv')
            print(f"✓ Saved confusion matrix to {output_dir}/soft_confusion_matrix.csv")
        
        # Print statistics
        print("\n" + "=" * 70)
        print("ENTROPY STATISTICS")
        print("=" * 70)
        print(f"Mean Entropy:     {np.mean(entropies):.4f} bits")
        print(f"Median Entropy:   {np.median(entropies):.4f} bits")
        print(f"Std Dev:          {np.std(entropies):.4f} bits")
        print(f"Min Entropy:      {np.min(entropies):.4f} bits")
        print(f"Max Entropy:      {np.max(entropies):.4f} bits")
        print(f"Max Possible:     {CIFAR10HAnalyzer.MAX_ENTROPY:.4f} bits")
        
        return results


if __name__ == "__main__":
    """
    Example usage with dummy CIFAR-10H data.
    """
    print("=" * 70)
    print("EDA Computations: Example Usage")
    print("=" * 70)
    
    # Generate dummy data (same shape as CIFAR-10H)
    np.random.seed(42)
    n_images = 10000
    n_classes = 10
    
    # Create random soft labels that sum to 1
    soft_labels_dummy = np.random.dirichlet(np.ones(n_classes), n_images)
    
    # Create dummy hard labels (true class)
    hard_labels_dummy = np.random.randint(0, n_classes, n_images)
    
    print(f"\nDataset shape: {n_images} images × {n_classes} classes")
    print(f"Soft labels shape: {soft_labels_dummy.shape}")
    print(f"Hard labels shape: {hard_labels_dummy.shape}")
    print(f"Soft labels sum check: min={soft_labels_dummy.sum(axis=1).min():.6f}, max={soft_labels_dummy.sum(axis=1).max():.6f}")
    
    # Run analysis
    print("\n" + "-" * 70)
    results = CIFAR10HAnalyzer.analyze_dataset(
        soft_labels_dummy,
        hard_labels_dummy,
        output_dir='./eda_results',
        save_formats=['npy', 'csv']
    )
    
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (first 5x5 preview)")
    print("=" * 70)
    print(results['confusion_matrix'][:5, :5])
    print("\n✓ Analysis complete!")
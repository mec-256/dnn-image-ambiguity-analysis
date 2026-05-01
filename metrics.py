

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from typing import Tuple, Union


# ═══════════════════════════════════════════════════════════════════════════
# COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(
    p: np.ndarray,
    q: np.ndarray,
    reduction: str = 'mean'
) -> Union[float, np.ndarray]:
    
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Ensure same shape
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p.shape={p.shape}, q.shape={q.shape}")
    
    # 1D case: single pair
    if p.ndim == 1:
        # scipy.spatial.distance.cosine returns distance (1 - similarity)
        distance = cosine(p, q)
        similarity = 1.0 - distance
        return float(similarity)
    
    # 2D case: batch of distributions
    elif p.ndim == 2:
        # Pairwise cosine similarity
        # sklearn returns (N, 1) matrix when comparing two arrays row-wise
        similarities = np.array([1.0 - cosine(p[i], q[i]) for i in range(len(p))])
        
        if reduction == 'mean':
            return float(np.mean(similarities))
        elif reduction == 'none':
            return similarities
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {p.shape}")


def cosine_similarity_batch(
    p: np.ndarray,
    q: np.ndarray
) -> np.ndarray:
   
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays")
    
    if p.shape[1] != q.shape[1]:
        raise ValueError(f"Dimension mismatch: p.shape[1]={p.shape[1]}, q.shape[1]={q.shape[1]}")
    
    # sklearn's cosine_similarity expects (n_samples, n_features)
    # Returns (n_samples_p, n_samples_q)
    return sklearn_cosine_similarity(p, q)


# ═══════════════════════════════════════════════════════════════════════════
# PEARSON CORRELATION
# ═══════════════════════════════════════════════════════════════════════════

def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    return_pvalue: bool = False
) -> Union[float, Tuple[float, float]]:
    
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 samples to compute correlation")
    
    r, p_value = pearsonr(x, y)
    
    if return_pvalue:
        return float(r), float(p_value)
    else:
        return float(r)


# ═══════════════════════════════════════════════════════════════════════════
# SPEARMAN CORRELATION
# ═══════════════════════════════════════════════════════════════════════════

def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
    return_pvalue: bool = False
) -> Union[float, Tuple[float, float]]:
    
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 samples to compute correlation")
    
    rho, p_value = spearmanr(x, y)
    
    if return_pvalue:
        return float(rho), float(p_value)
    else:
        return float(rho)


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED METRIC FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def distribution_matching_metrics(
    true_dist: np.ndarray,
    pred_dist: np.ndarray,
    return_all: bool = True
) -> dict:
    
    true_dist = np.asarray(true_dist, dtype=np.float64)
    pred_dist = np.asarray(pred_dist, dtype=np.float64)
    
    if true_dist.shape != pred_dist.shape:
        raise ValueError(f"Shape mismatch: true_dist.shape={true_dist.shape}, pred_dist.shape={pred_dist.shape}")
    
    # Compute entropies
    from scipy.stats import entropy as scipy_entropy
    true_entropy = np.array([scipy_entropy(p) for p in true_dist])
    pred_entropy = np.array([scipy_entropy(p) for p in pred_dist])
    
    # Main metrics
    cos_sim = cosine_similarity(true_dist, pred_dist, reduction='mean')
    pearson_r, pearson_p = pearson_correlation(true_entropy, pred_entropy, return_pvalue=True)
    spearman_rho, spearman_p = spearman_correlation(true_entropy, pred_entropy, return_pvalue=True)
    
    result = {
        'cosine_similarity': cos_sim,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho,
    }
    
    if return_all:
        result.update({
            'cosine_sim_per_sample': cosine_similarity(true_dist, pred_dist, reduction='none'),
            'pearson_pvalue': pearson_p,
            'spearman_pvalue': spearman_p,
            'true_entropy': true_entropy,
            'pred_entropy': pred_entropy,
        })
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════

def format_metrics(metrics: dict, prefix: str = "") -> str:
   
    lines = []
    lines.append(f"{prefix}Cosine Similarity     : {metrics['cosine_similarity']:.4f}")
    lines.append(f"{prefix}Pearson r (entropy)   : {metrics['pearson_r']:.4f}")
    lines.append(f"{prefix}Spearman ρ (entropy)  : {metrics['spearman_rho']:.4f}")
    
    if 'pearson_pvalue' in metrics:
        lines.append(f"{prefix}Pearson p-value       : {metrics['pearson_pvalue']:.6f}")
        lines.append(f"{prefix}Spearman p-value      : {metrics['spearman_pvalue']:.6f}")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    # Simple test
    print("Testing metrics.py wrapper functions...\n")
    
    # Test data
    np.random.seed(42)
    true_entropy = np.array([0.5, 1.2, 2.1, 0.3, 1.8, 2.5, 0.1, 1.4])
    pred_entropy = np.array([0.6, 1.1, 2.2, 0.4, 1.7, 2.4, 0.2, 1.5])
    
    # Test Pearson
    r = pearson_correlation(true_entropy, pred_entropy)
    r_pval, p = pearson_correlation(true_entropy, pred_entropy, return_pvalue=True)
    print(f"✓ Pearson correlation: r={r:.4f}")
    
    # Test Spearman
    rho = spearman_correlation(true_entropy, pred_entropy)
    print(f"✓ Spearman correlation: ρ={rho:.4f}")
    
    # Test Cosine Similarity 1D
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    sim = cosine_similarity(p, q)
    print(f"✓ Cosine similarity (1D): {sim:.4f}")
    
    # Test Cosine Similarity 2D
    true_dist = np.random.dirichlet(np.ones(10), 100)
    pred_dist = true_dist + np.random.normal(0, 0.05, true_dist.shape)
    pred_dist = np.clip(pred_dist, 0, 1)
    pred_dist /= pred_dist.sum(axis=1, keepdims=True)
    
    sim_batch = cosine_similarity(true_dist, pred_dist, reduction='mean')
    print(f"✓ Cosine similarity (2D batch, mean): {sim_batch:.4f}")
    
    # Test combined metrics
    metrics = distribution_matching_metrics(true_dist, pred_dist, return_all=True)
    print(f"\n✓ Combined metrics:")
    print(format_metrics(metrics, prefix="  "))
    
    print(f"\n All tests passed!")
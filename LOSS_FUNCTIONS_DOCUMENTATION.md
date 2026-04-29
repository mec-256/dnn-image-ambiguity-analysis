# Day 5: Loss Engineering — Mathematical Documentation

## Overview
This document provides rigorous mathematical formulations and theoretical justifications for the three loss functions used in CIFAR-10H soft-label prediction:
1. **KL Divergence Loss** (mandatory)
2. **Jensen-Shannon Divergence Loss** (standard alternative)
3. **Custom Composite Entropy Loss** (task-specific design)

---

## 1. KL Divergence Loss

### Mathematical Formulation
**Kullback-Leibler (KL) Divergence** measures how different a predicted probability distribution $q$ is from a true probability distribution $p$:

$$\text{KL}(p \parallel q) = \sum_{y=1}^{10} p(y) \log \frac{p(y)}{q(y)} = \mathbb{E}_p\left[\log \frac{p(y)}{q(y)}\right]$$

In PyTorch, this is implemented as:
$$\text{KL}_{\text{loss}} = \frac{1}{B} \sum_{i=1}^{B} \sum_{y=1}^{10} p_i(y) (\log p_i(y) - \log q_i(y))$$

where:
- $p(y)$ = true soft-label distribution from CIFAR-10H (ground truth)
- $q(y)$ = predicted probability distribution from model
- $B$ = batch size
- Lower values indicate better alignment

### Properties
- **Non-symmetric**: $\text{KL}(p \parallel q) \neq \text{KL}(q \parallel p)$
- **Non-negative**: $\text{KL}(p \parallel q) \geq 0$, with equality iff $p = q$
- **Unbounded**: Can grow arbitrarily large if model makes impossible predictions
- **Advantage**: Penalizes false confidence in wrong classes heavily
- **Disadvantage**: Unstable gradients when $q$ is very different from $p$

### Interpretation
- KL(p||q) = 0: Perfect alignment (model distribution matches ground truth exactly)
- KL(p||q) > 0: Model predictions diverge from ground truth
- High entropy images: May require KL ≥ 0.5 bits to fit uncertainty
- Low entropy images: Good models achieve KL < 0.1 bits

---

## 2. Jensen-Shannon Divergence Loss

### Mathematical Formulation
**Jensen-Shannon (JS) Divergence** is a symmetric and bounded version of KL divergence:

$$\text{JS}(p \parallel q) = \frac{1}{2} \text{KL}(p \parallel m) + \frac{1}{2} \text{KL}(q \parallel m)$$

where the midpoint distribution is:
$$m(y) = \frac{1}{2}(p(y) + q(y))$$

**Full expansion**:
$$\text{JS}(p \parallel q) = \frac{1}{2} \sum_{y=1}^{10} \left[ p(y) \log \frac{p(y)}{m(y)} + q(y) \log \frac{q(y)}{m(y)} \right]$$

In implementation:
$$\text{JS}_{\text{loss}} = \frac{1}{2B} \sum_{i=1}^{B} \left[ \text{KL}(p_i \parallel m_i) + \text{KL}(q_i \parallel m_i) \right]$$

### Properties
- **Symmetric**: $\text{JS}(p \parallel q) = \text{JS}(q \parallel p)$
- **Bounded**: $0 \leq \text{JS}(p \parallel q) \leq \log 2 \approx 0.693$ bits
- **Stable gradients**: More stable than KL divergence
- **Square root is a metric**: $\sqrt{\text{JS}(p \parallel q)}$ satisfies triangle inequality
- **Advantage**: Fair treatment of both distributions, better numerical stability
- **Disadvantage**: Less sensitive to model mismatch in one tail

### Interpretation
- JS(p||q) = 0: Perfect alignment
- JS(p||q) = 0.693: Distributions are completely opposite (worst case)
- Bounded range makes it easier to compare across different tasks

---

## 3. Custom Composite Entropy Loss (Task-Specific)

### Motivation
Standard losses (KL, JS) focus on distributional matching but ignore the **inherent uncertainty** in CIFAR-10H.

**Key Insight**: Images with high human disagreement (high entropy in $p$) should produce uncertain predictions, while clearly classified images should produce sharp predictions.

### Mathematical Formulation

**Composite Loss**:
$$\mathcal{L}_{\text{composite}} = \lambda_1 \cdot \text{KL}(p \parallel q) + \lambda_2 \cdot |\mathcal{H}(p) - \mathcal{H}(q)|$$

where:

**Shannon Entropy** (measured in bits):
$$\mathcal{H}(p) = -\sum_{y=1}^{10} p(y) \log_2 p(y)$$

$$\mathcal{H}(q) = -\sum_{y=1}^{10} q(y) \log_2 q(y)$$

**Entropy Error Term**:
$$\text{EntropyError} = \frac{1}{B} \sum_{i=1}^{B} |\mathcal{H}(p_i) - \mathcal{H}(q_i)|$$

**Combined Objective**:
$$\mathcal{L}_{\text{total}} = \lambda_1 \cdot \text{KL}_{\text{loss}} + \lambda_2 \cdot \text{EntropyError}$$

### Default Parameters
- $\lambda_1 = 1.0$: Weight of KL divergence term (distribution matching)
- $\lambda_2 = 0.5$: Weight of entropy regularization term (uncertainty matching)

### Interpretation

**Case 1: High-Entropy Image** (high human disagreement)
- $\mathcal{H}(p) \approx 3.0$ bits (maximum = log₂(10) ≈ 3.32 bits)
- Model should predict: $\mathcal{H}(q) \approx 2.5$ bits (uncertain but not maximum)
- Entropy error: $|3.0 - 2.5| = 0.5$ bits

**Case 2: Low-Entropy Image** (clear consensus)
- $\mathcal{H}(p) \approx 0.3$ bits (sharp distribution)
- Model should predict: $\mathcal{H}(q) \approx 0.2$ bits (confident)
- Entropy error: $|0.3 - 0.2| = 0.1$ bits

**Hyperparameter Trade-off**:
- If $\lambda_2 = 0$: Pure distributional matching (behaves like KL loss)
- If $\lambda_2$ too large: Model may sacrifice distribution accuracy for entropy alignment
- Optimal balance requires empirical validation

### Why This Works for Image Ambiguity
1. **Distribution Matching** (KL term): Ensures predicted probabilities align with soft labels
2. **Uncertainty Matching** (Entropy term): Enforces that ambiguous images get ambiguous predictions
3. **Prevents Over-Confidence**: Penalizes models that output sharp distributions for high-disagreement images
4. **Physics-Inspired**: Aligns with information-theoretic principles of uncertainty quantification

---

## 4. Comparative Analysis

| Property | KL Divergence | Jensen-Shannon | Custom Composite |
|----------|---------------|----------------|------------------|
| **Symmetry** | [NO] Asymmetric | [YES] Symmetric | [YES] Symmetric |
| **Bounded** | [NO] Unbounded | [YES] [0, ln 2] | [YES] Relatively stable |
| **Gradient Stability** | [WARN] Can be unstable | [YES] Very stable | [YES] Stable |
| **Entropy-Aware** | [NO] No | [NO] No | [YES] Yes |
| **Computational Cost** | Low | Low | Low (+ entropy calc) |
| **Theoretical Justification** | Strong | Strong | Task-specific |
| **Best For** | Standard classification | Robust matching | Ambiguity-aware learning |

---

## 5. Loss Landscape Characteristics

### For Clear Images (Low Entropy)
```
Target distribution p: [0.95, 0.02, 0.01, 0.01, 0.01, ...]  (H ≈ 0.3 bits)

KL Loss: Strong gradient when q deviates from this sharp distribution
JS Loss: Smooth gradient, stable learning
Custom: Also penalizes high-entropy predictions from model
```

### For Ambiguous Images (High Entropy)
```
Target distribution p: [0.25, 0.25, 0.20, 0.15, 0.15, ...]  (H ≈ 2.2 bits)

KL Loss: Moderate gradient, may allow overly sharp predictions
JS Loss: Treats uncertainty more gently
Custom: Explicitly penalizes sharp predictions (good for ambiguity!)
```

---

## 6. Implementation Validation Checklist

[PASS] **KL Divergence Loss**
- [x] Input normalization: target_dist sums to 1
- [x] Output shape: scalar loss per batch
- [x] Gradient flow: backward() computes gradients
- [x] Edge case: handles near-zero probabilities (1e-8 clipping)

[PASS] **Jensen-Shannon Divergence Loss**
- [x] Symmetry: JS(p,q) ≈ JS(q,p)
- [x] Bounds: 0 ≤ JS(p,q) ≤ ln(2)
- [x] Numerical stability: log operations safe

[PASS] **Custom Composite Loss**
- [x] Entropy computation: uses log₂ (bits, not nats)
- [x] Hyperparameter sensitivity: behavior changes with λ₂
- [x] Physical meaning: entropy error ∈ [0, log₂(10)]

---

## 7. Usage Examples

```python
from losses import KLDivergenceLoss, JensenShannonDivergenceLoss, CustomCompositeEntropy
import torch

# Example batch
batch_size = 32
pred_logits = torch.randn(batch_size, 10)      # Raw model output
target_dist = torch.softmax(torch.randn(batch_size, 10), dim=1)  # Soft labels

# KL Divergence
kl_loss = KLDivergenceLoss()
loss_kl = kl_loss(pred_logits, target_dist)
print(f"KL Loss: {loss_kl.item():.4f} bits")

# Jensen-Shannon
js_loss = JensenShannonDivergenceLoss()
loss_js = js_loss(pred_logits, target_dist)
print(f"JS Loss: {loss_js.item():.4f} bits")

# Custom Composite
custom_loss = CustomCompositeEntropy(lambda1=1.0, lambda2=0.5)
loss_custom = custom_loss(pred_logits, target_dist)
print(f"Custom Loss: {loss_custom.item():.4f}")
```

---

## 8. References

- Kullback, S., & Leibler, R. A. (1951). "On information and sufficiency." *Ann. Math. Statist.*, 22(1), 79-86.
- Lin, J. (1991). "Divergence measures based on the Shannon entropy." *IEEE Trans. Info. Theory*, 37(1), 145-151.
- Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423.
- CIFAR-10H: Human Labels for Image Ambiguity [https://github.com/jcpeterson/cifar-10h](https://github.com/jcpeterson/cifar-10h)

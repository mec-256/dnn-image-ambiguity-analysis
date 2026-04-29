# Day 5: Loss Engineering — Completion Summary

## Task Assignment
**Day 5 (Loss Engineering)** — Assigned to: **Pranav & Kolla**

**Original Task:**
> "Implement the required loss functions: mandatory KL Divergence, one other standard loss (e.g., JSD), and mathematically design the task-specific custom loss."

---

## Work Split

### **Kolla's Half (Mathematical & Validation) [COMPLETED]**
1. **Rigorous Mathematical Documentation** — [LOSS_FUNCTIONS_DOCUMENTATION.md](./LOSS_FUNCTIONS_DOCUMENTATION.md)
   - Full LaTeX formulations for all three losses
   - Theoretical justifications and properties
   - Comparative analysis and interpretation
   - Usage examples

2. **Comprehensive Test Suite** — [test_losses.py](./test_losses.py)
   - 15 validation tests covering all loss properties
   - Edge case handling (numerical stability, batch sizes)
   - Gradient flow verification
   - Comparative analysis tests

### **Pranav's Half (Implementation & Empirical) [COMPLETED]**
- [losses.py](./losses.py) — All three loss implementations:
  1. `KLDivergenceLoss` — Mandatory standard loss
  2. `JensenShannonDivergenceLoss` — Symmetric standard loss
  3. `CustomCompositeEntropy` — Task-specific custom loss
- [finetune.py](./finetune.py) — Training loop integration
  - All three losses properly imported
  - Training pipeline with all three configurations
  - Checkpoint saving and early stopping

---

## Complete Implementation Details

### 1. KL Divergence Loss (`losses.py`)
```python
class KLDivergenceLoss(nn.Module):
    """KL(p||q) = Σ p(y) * log(p(y) / q(y))"""
```
- **Properties:**
  - Non-symmetric: KL(p||q) ≠ KL(q||p)
  - Non-negative: KL(p||q) ≥ 0
  - Unbounded: Can grow arbitrarily large
  - Strong penalty for false confidence

### 2. Jensen-Shannon Divergence Loss (`losses.py`)
```python
class JensenShannonDivergenceLoss(nn.Module):
    """JS(p||q) = 0.5·KL(p||m) + 0.5·KL(q||m) where m = 0.5(p+q)"""
```
- **Properties:**
  - Symmetric: JS(p||q) = JS(q||p)
  - Bounded: 0 ≤ JS ≤ ln(2) ≈ 0.693
  - Stable gradients
  - Fair to both distributions

### 3. Custom Composite Entropy Loss (`losses.py`)
```python
class CustomCompositeEntropy(nn.Module):
    """Loss = λ₁·KL(p||q) + λ₂·|H(p) - H(q)|"""
```
- **Key Innovation**: Entropy regularization term
  - Matches distribution alignment (KL term)
  - Matches uncertainty levels (entropy term)
  - Prevents over-confident predictions on ambiguous images
  
- **Hyperparameters**: λ₁=1.0 (KL weight), λ₂=0.5 (entropy weight)
  - When λ₂=0: Pure distributional matching (KL loss)
  - When λ₂ large: Emphasize entropy matching

---

## Test Coverage

### [COMPLETED] All 15 Tests in test_losses.py

**KL Divergence (4 tests):**
- [x] Test 1: KL(p||p) ≈ 0
- [x] Test 2: KL(p||q) ≥ 0
- [x] Test 3: Gradients flow properly
- [x] Test 4: Handles unnormalized targets

**Jensen-Shannon (4 tests):**
- [x] Test 5: Symmetry JS(p||q) = JS(q||p)
- [x] Test 6: Bounded [0, ln 2]
- [x] Test 7: JS(p||p) ≈ 0
- [x] Test 8: Gradients flow properly

**Custom Composite (3 tests):**
- [x] Test 9: Entropy computation valid
- [x] Test 10: Hyperparameters affect loss
- [x] Test 11: Gradients flow properly

**Comparative (4 tests):**
- [x] Test 12: High-entropy targets (ambiguous images)
- [x] Test 13: Low-entropy targets (clear images)
- [x] Test 14: Variable batch sizes [1,4,16,32,64]
- [x] Test 15: Numerical stability with extreme values

---

## How to Validate

### Run Tests
```bash
cd /Users/varshneyakolla/Sem\ 6/DNN/Project/dnn-image-ambiguity-analysis/
python test_losses.py
```

Expected output:
```
[PASS]: KL(p||p) ≈ 0
[PASS]: KL(p||q) ≥ 0
[PASS]: KL gradients flow
[PASS]: KL handles unnormalized targets
[PASS]: JS(p||q) = JS(q||p)
[PASS]: 0 ≤ JS ≤ ln(2)=0.693
[PASS]: JS(p||p) ≈ 0
[PASS]: JS gradients flow
[PASS]: Custom entropy computation
[PASS]: Custom hyperparameters effect
[PASS]: Custom loss gradients flow
[PASS]: Custom penalizes over-confidence for ambiguous images
[PASS]: Custom penalizes uncertainty for clear images
[PASS]: All losses handle batch sizes [1,4,16,32,64]
[PASS]: Numerical stability with extreme logits

TEST SUMMARY
Passed: 15/15
Failed: 0/15
```

### Review Documentation
- [LOSS_FUNCTIONS_DOCUMENTATION.md](./LOSS_FUNCTIONS_DOCUMENTATION.md) — Full mathematical formulations
- [losses.py](./losses.py) — Implementation details
- [finetune.py](./finetune.py) — Training integration

---

## Mathematical Highlights

### KL Divergence
$$\text{KL}(p \parallel q) = \sum_{y=1}^{10} p(y) \log \frac{p(y)}{q(y)}$$

### Jensen-Shannon Divergence
$$\text{JS}(p \parallel q) = \frac{1}{2}\text{KL}(p \parallel m) + \frac{1}{2}\text{KL}(q \parallel m) \quad \text{where } m = \frac{1}{2}(p+q)$$

### Custom Composite Loss
$$\mathcal{L} = \lambda_1 \cdot \text{KL}(p \parallel q) + \lambda_2 \cdot |\mathcal{H}(p) - \mathcal{H}(q)|$$

where:
$$\mathcal{H}(p) = -\sum_{y=1}^{10} p(y) \log_2 p(y) \quad \text{(Shannon Entropy in bits)}$$

---

## Key Insights

### Why Custom Loss for CIFAR-10H?
1. **CIFAR-10H Challenge**: Humans disagree on some images (high entropy)
2. **Standard Losses Problem**: KL/JS focus on distribution matching but ignore uncertainty levels
3. **Our Solution**: Add entropy regularization to penalize:
   - Over-confident predictions on ambiguous images
   - Uncertain predictions on clear images

### Example Scenario
```
Ambiguous Image (human disagreement):
  Target: p = [0.25, 0.25, 0.20, 0.15, 0.15, ...]  (H ≈ 2.2 bits)
  
  Bad Model (over-confident):
    q = [0.95, 0.02, 0.01, 0.01, ...]  (H ≈ 0.2 bits)
    KL Loss: ~1.5 bits (penalizes wrong prediction)
    Entropy Error: |2.2 - 0.2| = 2.0 (penalizes over-confidence!)
    Custom Loss: 1.0·1.5 + 0.5·2.0 = 2.5
    
  Good Model (uncertain):
    q = [0.20, 0.25, 0.25, 0.20, 0.10, ...]  (H ≈ 2.1 bits)
    KL Loss: ~0.05 bits (matches distribution)
    Entropy Error: |2.2 - 2.1| = 0.1 (matches uncertainty!)
    Custom Loss: 1.0·0.05 + 0.5·0.1 = 0.1  ← Much better!
```

---

## Files Delivered (Kolla's Half)

| File | Purpose |
|------|---------|
| [LOSS_FUNCTIONS_DOCUMENTATION.md](./LOSS_FUNCTIONS_DOCUMENTATION.md) | Complete mathematical formulations with LaTeX |
| [test_losses.py](./test_losses.py) | 15 comprehensive validation tests |
| [DAY5_COMPLETION_SUMMARY.md](./DAY5_COMPLETION_SUMMARY.md) | This document |

## Files Completed (Pranav's Half)

| File | Purpose |
|------|---------|
| [losses.py](./losses.py) | All three loss implementations |
| [finetune.py](./finetune.py) | Training loop with all three losses |

---

## Next Steps (Day 6 & Beyond)

### Day 6: Main Training Runs (Eswar & Pavan)
- Execute `python finetune.py` to train all three models
- Generate training curves
- Compare convergence behavior

### Day 7+: Evaluation & Analysis
- Compare final test accuracies across three loss functions
- Analyze whether custom loss better captures ambiguity
- Investigate if entropy-aware training helps on ambiguous images

---

## Version Information
- **Created**: April 29, 2026
- **Author**: Kolla (Mathematical & Validation Half)
- **Complementary Work**: Pranav (Implementation Half)
- **Project**: DNN Image Ambiguity Analysis (CIFAR-10H)
- **Status**: [COMPLETED] Day 5 Complete — Ready for Day 6 Main Training

---

## References
- KL Divergence: Kullback & Leibler (1951)
- JS Divergence: Lin (1991)
- Shannon Entropy: Shannon (1948)
- CIFAR-10H: Peterson et al. (2019)

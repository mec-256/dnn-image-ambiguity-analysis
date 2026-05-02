"""
losses.py — Loss Function Implementations for CIFAR-10H Soft Label Prediction
==============================================================================
Author: Pavan
Purpose: Define KL divergence, Jensen-Shannon divergence, and custom composite loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss: KL(pred||target) = Σ_y q(y) * log(q(y) / p(y))
    where q=pred, p=target
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
    
    def forward(self, pred_logits, target_dist):
        """
        Args:
            pred_logits: Raw logits from model (batch_size, 10)
            target_dist: True soft label distribution (batch_size, 10)
        Returns:
            KL divergence loss
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # Normalize target distribution
        target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute KL(pred || target) = Σ pred * log(pred/target)
        # More numerically stable than using log_softmax directly
        kl_loss = torch.sum(
            pred_probs * (torch.log(pred_probs + 1e-8) - torch.log(target_dist + 1e-8)),
            dim=1
        )
        
        return kl_loss.mean()

class JensenShannonDivergenceLoss(nn.Module):
    """
    Jensen-Shannon Divergence Loss: Symmetric and more stable than KL divergence
    Formula: JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5(p+q)
    """
    def __init__(self):
        super(JensenShannonDivergenceLoss, self).__init__()
    
    def forward(self, pred_logits, target_dist):
        """
        Args:
            pred_logits: Raw logits from model (batch_size, 10)
            target_dist: True soft label distribution (batch_size, 10)
        Returns:
            Jensen-Shannon divergence loss
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # Normalize target distribution
        target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean distribution: m = 0.5(p + q)
        mean_dist = 0.5 * (pred_probs + target_dist)
        
        # Compute KL(p||m) and KL(q||m)
        kl_p_m = torch.sum(target_dist * (torch.log(target_dist + 1e-8) - torch.log(mean_dist + 1e-8)), dim=1)
        kl_q_m = torch.sum(pred_probs * (torch.log(pred_probs + 1e-8) - torch.log(mean_dist + 1e-8)), dim=1)
        
        # JS = 0.5 * KL(p||m) + 0.5 * KL(q||m)
        js_loss = 0.5 * kl_p_m + 0.5 * kl_q_m
        
        return js_loss.mean()


class CustomCompositeEntropy(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.5):
        super(CustomCompositeEntropy, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def _entropy(self, probs):
        return -torch.sum(probs * torch.log2(probs + 1e-8), dim=1)
    
    def forward(self, pred_logits, target_dist):
        # Convert to probabilities
        pred_probs = F.softmax(pred_logits, dim=1)
        target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
        
        # KL term - use CORRECT approach
        kl_loss = torch.sum(
            pred_probs * (torch.log(pred_probs + 1e-8) - torch.log(target_dist + 1e-8)),
            dim=1
        ).mean()
        
        # Entropy regularization term
        true_entropy = self._entropy(target_dist)
        pred_entropy = self._entropy(pred_probs)
        entropy_error = torch.abs(true_entropy - pred_entropy).mean()
        
        # Combined loss
        total_loss = self.lambda1 * kl_loss + self.lambda2 * entropy_error
        
        return total_loss
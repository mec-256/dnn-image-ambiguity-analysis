"""
robustness.py — Day 9: Image Ambiguity Robustness Check
========================================================
Applies increasing levels of Gaussian Blur to the test set 
and evaluates how the models' cosine similarity degrades.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

from model import CIFAR10H_ResNet
from dataset import CIFAR10HDataset
from metrics import distribution_matching_metrics

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

MODELS = {
    'KL Divergence': 'best_model_kl.pth',
    'Jensen-Shannon': 'best_model_jsd.pth',
    'Custom (KL+Entropy)': 'best_model_custom.pth'
}

# Blur intensities (0.0 is baseline unblurred)
BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

def load_test_data():
    print("[Data] Downloading CIFAR-10 TEST split from HuggingFace ...")
    ds_test = load_dataset("cifar10", split="test")
    test_images = np.array([np.array(item["img"]) for item in tqdm(ds_test, desc="  images", unit="img")])
    soft_labels = np.load('./cifar10h-probs.npy')
    
    # Return ONLY the test split (8000:10000) used in our evaluation
    return test_images[8000:10000], soft_labels[8000:10000]

def evaluate_blur(model, images, soft_labels, sigma):
    # Base transforms
    transform_list = [transforms.ToTensor()]
    
    # Add blur if sigma > 0
    if sigma > 0:
        # kernel_size must be odd. 5x5 is standard for 32x32 images.
        transform_list.append(transforms.GaussianBlur(kernel_size=5, sigma=sigma))
        
    transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    tf = transforms.Compose(transform_list)
    
    loader = DataLoader(CIFAR10HDataset(images, soft_labels, tf), batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = F.softmax(model(imgs), dim=1).cpu().numpy()
            all_preds.append(probs)
            all_trues.append(labels.numpy())
            
    pred = np.concatenate(all_preds)
    true = np.concatenate(all_trues)
    
    metrics = distribution_matching_metrics(true, pred, return_all=False)
    return metrics['cosine_similarity']

def main():
    print("🚀 Starting Day 9: Robustness Checks")
    images, soft_labels = load_test_data()
    
    results = {name: [] for name in MODELS.keys()}
    
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"⚠️ Skipping {name} - {path} not found.")
            continue
            
        print(f"\n🧠 Evaluating: {name}")
        model = CIFAR10H_ResNet().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        
        for sigma in BLUR_SIGMAS:
            score = evaluate_blur(model, images, soft_labels, sigma)
            results[name].append(score)
            print(f"  Blur Sigma {sigma:.1f} -> Cosine Similarity: {score:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(9, 6))
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    markers = ['o', 's', '^']
    
    for (name, scores), color, marker in zip(results.items(), colors, markers):
        if scores:
            plt.plot(BLUR_SIGMAS, scores, label=name, color=color, marker=marker, linewidth=2, markersize=8)

    plt.title('Model Robustness: Degradation of Soft-Label Matching under Blur', fontsize=14, fontweight='bold')
    plt.xlabel('Gaussian Blur Intensity (Sigma)', fontsize=12)
    plt.ylabel('Cosine Similarity (Higher is Better)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/robustness_summary.png', dpi=300, bbox_inches='tight')
    print("\n✅ Done! Chart saved to plots/robustness_summary.png")

if __name__ == '__main__':
    main()
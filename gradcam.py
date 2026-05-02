"""
gradcam.py — Day 10: Generate Grad-CAM heatmaps for clear vs ambiguous CIFAR-10H images.

This script loads a trained CIFAR10H_ResNet checkpoint and visualizes
model attention on sample images selected from the least ambiguous
(clear) and most ambiguous image sets.

Outputs:
    plots/gradcam_clear.png
    plots/gradcam_ambiguous.png
    plots/gradcam_summary.png
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import CIFAR10H_ResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

MODEL_CHECKPOINTS = {
    'custom': 'best_model_custom.pth',
    'kl': 'best_model_kl.pth',
    'jsd': 'best_model_jsd.pth'
}


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def compute_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-9
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()


def load_cifar10_test() -> np.ndarray:
    dataset = CIFAR10(root='data', train=False, download=True)
    images = np.stack([np.array(img) for img, _ in dataset], axis=0)
    return images


def load_cifar10h_soft_labels() -> np.ndarray:
    candidates = ['cifar10h-probs.npy', 'cifar10h-probs.npy.2']
    for filename in candidates:
        if os.path.exists(filename):
            print(f"[Data] Loaded soft labels from {filename}")
            return np.load(filename).astype(np.float32)

    print('[Warning] cifar10h-probs.npy not found locally. Downloading from GitHub...')
    url = 'https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy'
    try:
        import urllib.request
        urllib.request.urlretrieve(url, 'cifar10h-probs.npy')
        print('[Data] Downloaded cifar10h-probs.npy successfully.')
        return np.load('cifar10h-probs.npy').astype(np.float32)
    except Exception as e:
        raise FileNotFoundError(
            'Missing CIFAR-10H soft labels and failed to download them.\n'
            'Please add cifar10h-probs.npy or cifar10h-probs.npy.2 to the repository root.\n'
            f'Original error: {e}'
        )


def select_examples(soft_labels: np.ndarray, num_clear: int, num_ambiguous: int):
    entropy = -np.sum(soft_labels * np.log2(np.clip(soft_labels, 1e-9, 1.0)), axis=1)
    sorted_indices = np.argsort(entropy)
    clear_indices = sorted_indices[:num_clear]
    ambiguous_indices = sorted_indices[::-1][:num_ambiguous]
    return clear_indices, ambiguous_indices, entropy


def build_preprocess():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])


def tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def make_heatmap(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap('inferno')
    colored = cmap(mask)[:, :, :3]
    colored = np.uint8(colored * 255)
    overlay = np.clip(0.5 * colored + 0.5 * image, 0, 255).astype(np.uint8)
    return colored, overlay


def plot_samples(samples, title, save_path):
    count = len(samples)
    fig, axes = plt.subplots(count, 3, figsize=(10, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples):
        original, mask, overlay, label, pred_class, entropy_value = sample
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f'Original\nTrue Entropy: {entropy_value:.3f}')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask, cmap='inferno')
        axes[row, 1].set_title(f'Grad-CAM Mask\nPredicted: {pred_class}')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title('Overlay')
        axes[row, 2].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def resolve_checkpoint(checkpoint_key: str):
    # Accept either a known key or a direct checkpoint path.
    if checkpoint_key in MODEL_CHECKPOINTS:
        candidate = MODEL_CHECKPOINTS[checkpoint_key]
        if os.path.exists(candidate):
            return candidate

        fallback = 'pretrained_backbone.pth'
        if os.path.exists(fallback):
            print(f"[Warning] '{candidate}' not found. Falling back to '{fallback}'.")
            return fallback

        print(f"[Warning] '{candidate}' not found and no pretrained backbone available.")
        print("         Grad-CAM will use a randomly initialized model instead.")
        return None

    if os.path.exists(checkpoint_key):
        return checkpoint_key

    print(f"[Warning] Provided checkpoint path not found: {checkpoint_key}")
    print("         Grad-CAM will use a randomly initialized model instead.")
    return None


def build_model(checkpoint_path: str = None) -> torch.nn.Module:
    model = CIFAR10H_ResNet().to(DEVICE)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("[Info] No checkpoint loaded; using random initialization for Grad-CAM.")
    model.eval()
    return model


def run_gradcam(checkpoint_key: str, num_clear: int, num_ambiguous: int):
    model_path = resolve_checkpoint(checkpoint_key)
    model = build_model(model_path)
    test_images = load_cifar10_test()
    soft_labels = load_cifar10h_soft_labels()[8000:10000]
    test_images = test_images[8000:10000]

    clear_idx, ambiguous_idx, entropy_values = select_examples(soft_labels, num_clear, num_ambiguous)
    preprocess = build_preprocess()
    gradcam = GradCAM(model, target_layer=model.backbone.layer4)

    clear_samples = []
    ambiguous_samples = []

    for idx in clear_idx:
        pil_image = test_images[idx]
        original = pil_image.astype(np.uint8)
        input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
        cam_mask = gradcam.compute_cam(input_tensor)
        heatmap, overlay = make_heatmap(original, cam_mask)
        logits = model(input_tensor)
        pred_class = int(torch.argmax(logits, dim=1).item())
        clear_samples.append((original, cam_mask, overlay, soft_labels[idx], pred_class, float(entropy_values[idx])))

    for idx in ambiguous_idx:
        pil_image = test_images[idx]
        original = pil_image.astype(np.uint8)
        input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
        cam_mask = gradcam.compute_cam(input_tensor)
        heatmap, overlay = make_heatmap(original, cam_mask)
        logits = model(input_tensor)
        pred_class = int(torch.argmax(logits, dim=1).item())
        ambiguous_samples.append((original, cam_mask, overlay, soft_labels[idx], pred_class, float(entropy_values[idx])))

    plot_samples(clear_samples, 'Grad-CAM: Clear CIFAR-10H Images', 'plots/gradcam_clear.png')
    plot_samples(ambiguous_samples, 'Grad-CAM: Ambiguous CIFAR-10H Images', 'plots/gradcam_ambiguous.png')

    # Combined summary plot
    summary_samples = clear_samples + ambiguous_samples
    plot_samples(summary_samples, 'Grad-CAM: Clear vs Ambiguous Images', 'plots/gradcam_summary.png')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmaps for clear vs ambiguous images.')
    parser.add_argument('--model', default='custom', help='Checkpoint key or path: custom, kl, jsd, or filepath')
    parser.add_argument('--clear', type=int, default=4, help='Number of clear images to visualize')
    parser.add_argument('--ambiguous', type=int, default=4, help='Number of ambiguous images to visualize')
    return parser.parse_args()


def main():
    args = parse_args()
    run_gradcam(args.model, args.clear, args.ambiguous)


if __name__ == '__main__':
    main()

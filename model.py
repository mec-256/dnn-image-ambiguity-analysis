import torch
import torch.nn as nn
import torchvision.models as models

class CIFAR10H_ResNet(nn.Module):
    def __init__(self):
        super(CIFAR10H_ResNet, self).__init__()
        
        # 1. Load the standard un-pretrained ResNet-18 backbone
        self.backbone = models.resnet18(weights=None)
        
        # 2. SURGERY: Replace the aggressive 7x7 convolution
        # A 3x3 kernel with stride 1 prevents the 32x32 image from shrinking too fast
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # 3. SURGERY: Remove the MaxPool layer entirely
        self.backbone.maxpool = nn.Identity()
        
        # 4. SURGERY: Adjust the Prediction Head
        # CIFAR-10 has exactly 10 classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.backbone(x)

# Quick local test block
if __name__ == "__main__":
    model = CIFAR10H_ResNet()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Architecture built successfully! Test output shape: {output.shape}")
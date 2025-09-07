# losses/perceptual.py
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGPerceptual(nn.Module):
    def __init__(self, layer='features_16', weight=1.0):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # Freeze params
        for p in vgg.parameters(): p.requires_grad = False
        self.slice = nn.Sequential(*list(vgg.children())[:17])  # up to relu4_1 (features_16)
        self.slice.eval()
        self.weight = weight

    def forward(self, pred, target):
        # pred/target in [0,1]
        def norm(x):
            # VGG expects ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
            return (x - mean) / std
        fp = self.slice(norm(pred))
        ft = self.slice(norm(target))
        return self.weight * torch.nn.functional.l1_loss(fp, ft)
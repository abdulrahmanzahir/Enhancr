import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, in_ch=3, feat=64):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, feat, 9, padding=4), nn.ReLU(True))
        self.body = nn.Sequential(nn.Conv2d(feat, 32, 5, padding=2), nn.ReLU(True))
        self.tail = nn.Conv2d(32, in_ch, 5, padding=2)
    def forward(self, x):
        return self.tail(self.body(self.head(x)))

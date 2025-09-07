# models/edsr_lite.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return x + res

class EDSRLite(nn.Module):
    def __init__(self, in_ch=3, n_feats=64, n_resblocks=12, res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(in_ch, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)],
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.tail = nn.Conv2d(n_feats, in_ch, 3, padding=1)
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        return self.tail(x)
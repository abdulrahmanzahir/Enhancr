# models/discriminator.py
import torch.nn as nn

# Lightweight PatchGAN-style discriminator
# Input: Bx3xHxW  -> Output: Bx1xhxw (patch map)
def conv_block(ic, oc, k=3, s=1, p=1, use_norm=True):
    layers = [nn.Conv2d(ic, oc, k, s, p)]
    if use_norm:
        layers.append(nn.InstanceNorm2d(oc, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        nf = 64
        self.net = nn.Sequential(
            # no norm on first block (common in PatchGAN)
            nn.Conv2d(in_ch, nf, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),

            conv_block(nf,    nf,    k=4, s=2, p=1),   # 1/2
            conv_block(nf,    nf*2,  k=4, s=2, p=1),   # 1/4
            conv_block(nf*2,  nf*4,  k=4, s=2, p=1),   # 1/8
            conv_block(nf*4,  nf*4,  k=3, s=1, p=1),   # keep spatial
            nn.Conv2d(nf*4, 1, 3, 1, 1)               # logits (no sigmoid)
        )

    def forward(self, x):
        return self.net(x)
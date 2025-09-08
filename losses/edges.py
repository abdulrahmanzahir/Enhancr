# losses/edges.py
import torch, torch.nn as nn
import torch.nn.functional as F

class SobelEdgeLoss(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edges(self, x):
        # x: BxCxHxW in [0,1]
        gray = 0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
        return mag

    def forward(self, pred, target):
        pe, te = self._edges(pred), self._edges(target)
        return self.weight * F.l1_loss(pe, te)
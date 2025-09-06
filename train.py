import os, csv, random, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from models.srcnn import SRCNN

class PairDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = list(csv.reader(open(csv_path)))[1:]
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        _, lr_p, hr_p = self.rows[idx]
        lr = np.array(Image.open(lr_p).convert('RGB'))
        hr = np.array(Image.open(hr_p).convert('RGB'))
        H, W, _ = hr.shape; s = 96
        if H>=s and W>=s:
            y = random.randint(0, H-s); x = random.randint(0, W-s)
            lr = lr[y:y+s, x:x+s]; hr = hr[y:y+s, x:x+s]
        lr = torch.from_numpy(lr).permute(2,0,1).float()/255.
        hr = torch.from_numpy(hr).permute(2,0,1).float()/255.
        return lr, hr

def train(csv_path='data/pairs/pairs.csv', epochs=5, bs=16, lr=2e-4, ckpt='weights/srcnn.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SRCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.L1Loss()
    dl = DataLoader(PairDataset(csv_path), batch_size=bs, shuffle=True, num_workers=0)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    for ep in range(1, epochs+1):
        net.train(); total=0
        for lri, hri in dl:
            lri, hri = lri.to(device), hri.to(device)
            pred = net(lri)
            loss = crit(pred, hri)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*lri.size(0)
        print(f'Epoch {ep}: L1={total/len(dl.dataset):.4f}')
        torch.save({'epoch':ep,'state_dict':net.state_dict()}, ckpt)

if __name__ == '__main__':
    train()

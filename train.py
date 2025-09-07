# train.py
import os, csv, random, math, time
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
from models.srcnn import SRCNN

# --------------------
# Dataset
# --------------------
class PairDataset(Dataset):
    def __init__(self, csv_path):
        rows = list(csv.reader(open(csv_path, newline='', encoding='utf-8')))
        self.rows = rows[1:]  # skip header
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        # tolerate extra CSV columns (meta_json, etc.)
        _, lr_p, hr_p, *_ = self.rows[idx]
        lr = np.array(Image.open(lr_p).convert('RGB'))
        hr = np.array(Image.open(hr_p).convert('RGB'))
        H, W, _ = hr.shape
        s = 96
        if H >= s and W >= s:
            y = np.random.randint(0, max(1, H - s + 1))
            x = np.random.randint(0, max(1, W - s + 1))
            lr = lr[y:y+s, x:x+s]
            hr = hr[y:y+s, x:x+s]
        lr = torch.from_numpy(lr).permute(2,0,1).float()/255.
        hr = torch.from_numpy(hr).permute(2,0,1).float()/255.
        return lr, hr

# --------------------
# Metrics
# --------------------
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0: return 99.0
    return 10 * math.log10(1.0 / mse)

# --------------------
# Train / Val loop
# --------------------
def train(
    csv_path='data/pairs/pairs.csv',
    epochs=20,
    batch_size=8,
    lr=2e-4,
    ckpt_best='weights/srcnn_best.pth',
    ckpt_last='weights/srcnn_last.pth',
    num_workers=0,
    early_stop_patience=5
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('weights', exist_ok=True)

    # dataset & split
    full = PairDataset(csv_path)
    n = len(full)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = max(1, int(0.1 * n))  # 10% val
    val_idx = idxs[:split]
    train_idx = idxs[split:]
    dtrain = Subset(full, train_idx)
    dval = Subset(full, val_idx)

    dl_train = DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(dval,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    net = SRCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.L1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device=='cuda'))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    best_psnr = -1.0
    epochs_no_improve = 0

    for ep in range(1, epochs+1):
        # -------- TRAIN --------
        net.train()
        t0 = time.time()
        total_loss = 0.0
        for lri, hri in dl_train:
            lri, hri = lri.to(device, non_blocking=True), hri.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                pred = net(lri)
                loss = crit(pred, hri)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * lri.size(0)

        train_loss = total_loss / len(dtrain)

        # -------- VALIDATE --------
        net.eval()
        total_val_loss = 0.0
        total_psnr = 0.0
        with torch.no_grad():
            for lri, hri in dl_val:
                lri, hri = lri.to(device, non_blocking=True), hri.to(device, non_blocking=True)
                pred = net(lri).clamp(0,1)
                total_val_loss += crit(pred, hri).item() * lri.size(0)
                total_psnr += psnr(pred, hri) * lri.size(0)

        val_loss = total_val_loss / len(dval)
        val_psnr = total_psnr / len(dval)

        # scheduler on val_loss
        sched.step(val_loss)

        # logging
        dt = time.time() - t0
        curr_lr = opt.param_groups[0]['lr']
        print(f"Epoch {ep:02d}/{epochs} | "
              f"train_L1={train_loss:.4f} | val_L1={val_loss:.4f} | val_PSNR={val_psnr:.2f} dB | "
              f"lr={curr_lr:.1e} | {dt:.1f}s")

        # save last
        torch.save({'epoch':ep,'state_dict':net.state_dict()}, ckpt_last)

        # save best (by PSNR)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_no_improve = 0
            torch.save({'epoch':ep,'state_dict':net.state_dict()}, ckpt_best)
            print(f"  ↳ New best PSNR: {best_psnr:.2f} dB (saved {ckpt_best})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stop: no PSNR improvement for {early_stop_patience} epoch(s).")
                break

if __name__ == '__main__':
    # seed for reproducibility of the split
    random.seed(1337)
    torch.manual_seed(1337)
    train()
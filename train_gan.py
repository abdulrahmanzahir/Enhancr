# train_gan.py
import os, csv, random, math, time
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==== [NEW — active generator/discriminator + losses] ====
from models.edsr_lite import EDSRLite
from models.discriminator import Discriminator
from losses.perceptual import VGGPerceptual
from losses.edges import SobelEdgeLoss

# ==== [PAST — EDSR non-GAN training kept in train.py] ====
# (We keep it as-is; this file only adds adversarial training.)

# ==== [VERY PAST — SRCNN baseline kept as comments in train.py] ====
# from models.srcnn import SRCNN


# --------------------
# Dataset (same as train.py)
# --------------------
class PairDataset(Dataset):
    def __init__(self, csv_path):
        rows = list(csv.reader(open(csv_path, newline='', encoding='utf-8')))
        self.rows = rows[1:]
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
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
# GAN Train
# --------------------
def train_gan(
    csv_path='data/pairs/pairs.csv',
    epochs=50,
    batch_size=8,
    lr_G=2e-4,
    lr_D=1e-4,
    ckpt_best='weights/edsr_gan_best.pth',
    ckpt_last='weights/edsr_gan_last.pth',
    num_workers=0,
    early_stop_patience=6,
    # loss weights
    perc_weight=0.12,
    edge_weight=0.07,
    adv_weight=2e-3,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('weights', exist_ok=True)

    # data
    full = PairDataset(csv_path)
    n = len(full)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = max(1, int(0.1 * n))
    val_idx = idxs[:split]
    train_idx = idxs[split:]
    dtrain = Subset(full, train_idx)
    dval = Subset(full, val_idx)

    dl_train = DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(dval,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # models
    netG = EDSRLite(n_feats=96, n_resblocks=16, res_scale=0.1).to(device)
    netD = Discriminator().to(device)

    # opt
    optG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.9, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.9, 0.999))

    # losses
    crit_l1   = nn.L1Loss()
    crit_perc = VGGPerceptual(weight=perc_weight).to(device)
    crit_edge = SobelEdgeLoss(weight=edge_weight).to(device)
    crit_adv  = nn.BCEWithLogitsLoss()

    scalerG = torch.amp.GradScaler('cuda', enabled=(device=='cuda'))
    scalerD = torch.amp.GradScaler('cuda', enabled=(device=='cuda'))

    best_psnr = -1.0
    epochs_no_improve = 0

    for ep in range(1, epochs+1):
        netG.train(); netD.train()
        t0 = time.time()
        g_loss_running = 0.0
        d_loss_running = 0.0

        with tqdm(dl_train, desc=f"Epoch {ep:02d}/{epochs}", ncols=100, dynamic_ncols=True, leave=False) as pbar:
            for lri, hri in pbar:
                lri, hri = lri.to(device, non_blocking=True), hri.to(device, non_blocking=True)

                # --------------------
                # 1) Update D
                # --------------------
                optD.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device=='cuda')):
                    fake = netG(lri).detach()
                    logits_real = netD(hri)
                    logits_fake = netD(fake)
                    # targets are patch maps of ones/zeros
                    ones  = torch.ones_like(logits_real)
                    zeros = torch.zeros_like(logits_fake)
                    d_loss = crit_adv(logits_real, ones) + crit_adv(logits_fake, zeros)
                scalerD.scale(d_loss).backward()
                scalerD.step(optD)
                scalerD.update()

                # --------------------
                # 2) Update G
                # --------------------
                optG.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device=='cuda')):
                    pred = netG(lri)
                    logits_fake_for_G = netD(pred)

                    # reconstruction + perceptual + edge
                    loss_l1   = crit_l1(pred, hri)
                    loss_perc = crit_perc(pred.clamp(0,1), hri)
                    loss_edge = crit_edge(pred.clamp(0,1), hri)
                    # adversarial (want D to say "real")
                    g_adv = crit_adv(logits_fake_for_G, torch.ones_like(logits_fake_for_G))

                    g_loss = loss_l1 + loss_perc + loss_edge + adv_weight * g_adv

                scalerG.scale(g_loss).backward()
                torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                scalerG.step(optG)
                scalerG.update()

                g_loss_running += g_loss.item() * lri.size(0)
                d_loss_running += d_loss.item() * lri.size(0)
                pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

        train_g = g_loss_running / len(dtrain)
        train_d = d_loss_running / len(dtrain)

        # -------- VALIDATE (no GAN terms) --------
        netG.eval()
        total_val_loss = 0.0
        total_psnr_val = 0.0
        with torch.no_grad():
            for lri, hri in dl_val:
                lri, hri = lri.to(device, non_blocking=True), hri.to(device, non_blocking=True)
                pred = netG(lri).clamp(0,1)
                v_l1   = crit_l1(pred, hri).item()
                v_perc = crit_perc(pred, hri).item()
                v_edge = crit_edge(pred, hri).item()
                total_val_loss += (v_l1 + v_perc + v_edge) * lri.size(0)
                total_psnr_val += psnr(pred, hri) * lri.size(0)

        val_loss = total_val_loss / len(dval)
        val_psnr = total_psnr_val / len(dval)

        # Log
        dt = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs} | G={train_g:.4f} | D={train_d:.4f} | "
              f"val_loss={val_loss:.4f} | val_PSNR={val_psnr:.2f} dB | {dt:.1f}s")

        # Save latest G
        torch.save({'epoch': ep, 'state_dict': netG.state_dict()}, ckpt_last)

        # Save best G by PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_no_improve = 0
            torch.save({'epoch': ep, 'state_dict': netG.state_dict()}, ckpt_best)
            print(f"  ↳ New best PSNR: {best_psnr:.2f} dB (saved {ckpt_best})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stop: no PSNR improvement for {early_stop_patience} epoch(s).")
                break


if __name__ == '__main__':
    random.seed(1337); torch.manual_seed(1337)
    train_gan()
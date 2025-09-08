import torch
from PIL import Image
import numpy as np

# ==== [NEW — active model: EDSR-lite generator, GAN-trained weights] ====
from models.edsr_lite import EDSRLite

# ==== [VERY PAST — SRCNN baseline (kept for reference)] ====
# from models.srcnn import SRCNN

# ==== [NEW — optional helpers for sharper inference] ====
import cv2


def tta_predict(net, x):  # x: 1x3xHxW in [0,1]
    outs = []
    for k in range(4):  # 0,90,180,270
        xr = torch.rot90(x, k, [2,3])
        for flip in [False, True]:
            xrf = torch.flip(xr, [3]) if flip else xr
            y = net(xrf).clamp(0,1)
            y = torch.flip(y, [3]) if flip else y
            y = torch.rot90(y, -k, [2,3])
            outs.append(y)
    return torch.mean(torch.stack(outs, dim=0), dim=0)


def unsharp_mask(img_bgr_uint8, amount=0.6, radius=1.2, threshold=0):
    blur = cv2.GaussianBlur(img_bgr_uint8, (0,0), radius)
    sharp = cv2.addWeighted(img_bgr_uint8, 1+amount, blur, -amount, 0)
    if threshold > 0:
        mask = (cv2.absdiff(img_bgr_uint8, blur) < threshold)
        sharp[mask] = img_bgr_uint8[mask]
    return sharp


@torch.no_grad()
def enhance_image(
    input_path,
    output_path,
    ckpt='weights/edsr_gan_last.pth',   # ==== [NEW — default to GAN weights] ====

    # ==== [PAST — pre-GAN EDSR weights] ====
    # ckpt='weights/edsr_lite_best.pth',

    # ==== [VERY PAST — SRCNN baseline weights] ====
    # ckpt='weights/srcnn_best.pth',

    USE_TTA=False,       # toggle test-time augmentation
    USE_UNSHARP=False    # toggle sharpening
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==== [NEW — EDSR-lite generator active; weights come from GAN training] ====
    net = EDSRLite(n_feats=96, n_resblocks=16, res_scale=0.1).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'])
    net.eval()

    # ==== [VERY PAST — SRCNN baseline] ====
    # net = SRCNN().to(device)
    # net.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'])
    # net.eval()

    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    ten = torch.from_numpy(arr).permute(2,0,1).float().unsqueeze(0) / 255.0
    ten = ten.to(device)

    # ==== [NEW — forward pass with TTA toggle] ====
    if USE_TTA:
        pred = tta_predict(net, ten)
    else:
        pred = net(ten).clamp(0,1)

    # ==== [PAST — plain forward pass always] ====
    # pred = net(ten).clamp(0,1)

    out = pred.clamp(0,1).cpu().squeeze(0)
    rgb = (out.permute(1,2,0).numpy() * 255).astype('uint8')

    # ==== [NEW — optional sharpening toggle] ====
    if USE_UNSHARP:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = unsharp_mask(bgr, amount=0.6, radius=1.2, threshold=0)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    Image.fromarray(rgb).save(output_path, format='PNG')


if __name__ == '__main__':
    # You can toggle TTA / unsharp here
    enhance_image('sample_input.jpg', 'enhanced.png', USE_TTA=False, USE_UNSHARP=False)
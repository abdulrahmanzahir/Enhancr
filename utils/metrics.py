import torch, math
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0: return 99.0
    return 10 * math.log10(1.0 / mse)

# utils/degrade.py
import cv2, math, random, numpy as np

# -----------------------------
# Helpers
# -----------------------------
def _rand(a, b):
    return random.uniform(a, b)

def _choice(seq):
    return random.choice(seq)

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

def _to_float(img_bgr):
    return img_bgr.astype(np.float32) / 255.0

def _to_uint8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)

# -----------------------------
# Blur kernels
# -----------------------------
def gaussian_kernel(ksize, sigma_x, sigma_y=None, theta=0.0):
    """Anisotropic Gaussian via covariance matrix rotation."""
    if sigma_y is None:
        sigma_y = sigma_x
    half = ksize // 2
    xs = np.arange(-half, half + 1)
    ys = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(xs, ys)
    # rotate coords by theta
    c, s = math.cos(theta), math.sin(theta)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    ker = np.exp(-0.5 * ((xr ** 2) / (sigma_x ** 2) + (yr ** 2) / (sigma_y ** 2)))
    ker /= np.sum(ker) + 1e-12
    return ker.astype(np.float32)

def motion_kernel(ksize, angle_deg, intensity=1.0):
    """Simple linear motion blur kernel."""
    k = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    angle = math.radians(angle_deg)
    dx = math.cos(angle); dy = math.sin(angle)
    # draw a line through the center
    for t in np.linspace(-center, center, num=ksize):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < ksize and 0 <= y < ksize:
            k[y, x] = 1.0
    k = cv2.GaussianBlur(k, (0, 0), sigmaX=ksize * 0.05 + 1.0)
    k /= (k.sum() + 1e-12)
    k *= intensity
    k /= (k.sum() + 1e-12)
    return k

def apply_kernel(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# -----------------------------
# Resize / downscale
# -----------------------------
_INTERS = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def random_downscale(img, min_scale=1.5, max_scale=4.0):
    """Downscale by a random factor [min_scale, max_scale], then return LR and scale factor."""
    h, w = img.shape[:2]
    scale = _rand(min_scale, max_scale)
    nh, nw = max(8, int(round(h / scale))), max(8, int(round(w / scale)))
    inter = _choice(_INTERS)
    lr = cv2.resize(img, (nw, nh), interpolation=inter)
    return lr, scale, inter

def upscale_to(img, target_hw, inter=None):
    h, w = target_hw
    if inter is None: inter = _choice(_INTERS)
    return cv2.resize(img, (w, h), interpolation=inter)

# -----------------------------
# Noise / compression / color
# -----------------------------
def add_gaussian_noise(img, sigma=0.02):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return _clip01(img + noise)

def add_poisson_noise(img, strength=30.0):
    # scale to counts, add Poisson, rescale
    img_clip = _clip01(img)
    vals = strength
    noisy = np.random.poisson(img_clip * vals) / float(vals)
    return _clip01(noisy.astype(np.float32))

def jpeg_compress_decode(bgr_uint8, quality):
    enc = cv2.imencode('.jpg', bgr_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

def gamma_adjust(img, gamma):
    img = _clip01(img)
    return np.power(img, gamma)

def contrast_brightness(img, alpha, beta):
    # alpha: contrast (0.8–1.2), beta: brightness (-0.05–0.05)
    out = img * alpha + beta
    return _clip01(out)

def chroma_subsample_ycbcr(bgr_uint8):
    """Simulate 4:2:0 subsampling artifacts."""
    ycrcb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    # downsample chroma
    Cr_ds = cv2.resize(Cr, (Cr.shape[1]//2, Cr.shape[0]//2), interpolation=cv2.INTER_AREA)
    Cb_ds = cv2.resize(Cb, (Cb.shape[1]//2, Cb.shape[0]//2), interpolation=cv2.INTER_AREA)
    # upsample back
    Cr_up = cv2.resize(Cr_ds, (Cr.shape[1], Cr.shape[0]), interpolation=_choice(_INTERS))
    Cb_up = cv2.resize(Cb_ds, (Cb.shape[1], Cb.shape[0]), interpolation=_choice(_INTERS))
    ycrcb = cv2.merge([Y, Cr_up, Cb_up])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# -----------------------------
# Main pipeline
# -----------------------------
def random_degrade(hr_bgr, min_scale=1.5, max_scale=4.0):
    """
    Complex random pipeline:
      1) 0–2 blurs (Gaussian &/or anisotropic &/or motion)
      2) random downscale (random interpolation)
      3) 0–2 noises (Gaussian/Poisson) in random order
      4) optional chroma subsampling simulation
      5) optional JPEG compression (quality 30–95)
      6) optional gamma/contrast/brightness jitter
      7) upscale back to HR size (random interpolation)
    Returns (lr_up_bgr_uint8, hr_bgr_uint8, params_dict)
    """
    H, W = hr_bgr.shape[:2]
    params = {}

    # work in float [0,1] until JPEG/compression steps that need uint8
    x = hr_bgr.copy()

    # (1) BLUR SEQUENCE
    num_blurs = _choice([0, 1, 2])
    params['num_blurs'] = num_blurs
    for i in range(num_blurs):
        blur_type = _choice(['iso_gauss', 'aniso_gauss', 'motion'])
        if blur_type == 'iso_gauss':
            k = _choice([3,5,7])
            sigma = _rand(0.5, 2.0)
            ker = gaussian_kernel(k, sigma_x=sigma)
            x = apply_kernel(x, ker)
            params[f'blur{i}'] = {'type':'iso_gauss','k':k,'sigma':sigma}
        elif blur_type == 'aniso_gauss':
            k = _choice([5,7,9])
            sx = _rand(0.6, 2.5)
            sy = _rand(0.6, 2.5)
            theta = _rand(0, math.pi)
            ker = gaussian_kernel(k, sigma_x=sx, sigma_y=sy, theta=theta)
            x = apply_kernel(x, ker)
            params[f'blur{i}'] = {'type':'aniso_gauss','k':k,'sx':sx,'sy':sy,'theta':theta}
        else:  # motion
            k = _choice([7,9,11,13])
            ang = _rand(0, 180)
            ker = motion_kernel(k, ang, intensity=1.0)
            x = apply_kernel(x, ker)
            params[f'blur{i}'] = {'type':'motion','k':k,'angle':ang}

    # (2) DOWNSCALE
    lr_small, scale, inter_down = random_downscale(x, min_scale=min_scale, max_scale=max_scale)
    params['downscale'] = {'scale': scale, 'inter': int(inter_down)}

    # switch to float for noise/jitter
    lr = _to_float(lr_small)

    # (3) NOISES (0–2)
    num_noises = _choice([0, 1, 2])
    params['num_noises'] = num_noises
    for i in range(num_noises):
        ntype = _choice(['gauss', 'poisson'])
        if ntype == 'gauss':
            sigma = _rand(0.004, 0.02)  # ~[1/255 .. 5/255]
            lr = add_gaussian_noise(lr, sigma=sigma)
            params[f'noise{i}'] = {'type':'gauss','sigma':sigma}
        else:
            strength = _rand(10, 60)
            lr = add_poisson_noise(lr, strength=strength)
            params[f'noise{i}'] = {'type':'poisson','strength':strength}

    # (4) CHROMA SUBSAMPLING (optional ~50%)
    if random.random() < 0.5:
        lr_uint8 = _to_uint8(lr)
        lr_uint8 = chroma_subsample_ycbcr(lr_uint8)
        lr = _to_float(lr_uint8)
        params['chroma_420'] = True
    else:
        params['chroma_420'] = False

    # (5) JPEG (optional ~70%)
    if random.random() < 0.7:
        q = int(round(_rand(30, 95)))
        lr_uint8 = _to_uint8(lr)
        lr_uint8 = jpeg_compress_decode(lr_uint8, q)
        lr = _to_float(lr_uint8)
        params['jpeg_q'] = q
    else:
        params['jpeg_q'] = None

    # (6) TONE JITTER (optional)
    if random.random() < 0.7:
        if random.random() < 0.5:
            gamma = _rand(0.8, 1.2)
            lr = gamma_adjust(lr, gamma)
            params['gamma'] = gamma
        if random.random() < 0.5:
            alpha = _rand(0.9, 1.1)
            beta = _rand(-0.05, 0.05)
            lr = contrast_brightness(lr, alpha, beta)
            params['contrast_alpha'] = alpha
            params['brightness_beta'] = beta

    # (7) UPSCALE BACK TO HR SIZE
    lr_up = upscale_to(_to_uint8(lr), (H, W), inter=None)

    return lr_up, hr_bgr, params
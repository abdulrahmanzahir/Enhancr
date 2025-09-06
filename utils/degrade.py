import random, cv2, numpy as np

def random_degrade(hr_bgr, scale=2):
    h, w = hr_bgr.shape[:2]
    k = random.choice([0,1,3,5])
    if k>0: hr_bgr = cv2.GaussianBlur(hr_bgr, (k,k), 0)
    nh, nw = h//scale, w//scale
    interp = random.choice([cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR])
    lr = cv2.resize(hr_bgr, (nw, nh), interpolation=interp)
    if random.random() < 0.7:
        noise = np.random.normal(0, random.choice([2,4,6]), lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        q = random.randint(30, 90)
        _, buf = cv2.imencode('.jpg', lr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        lr = cv2.imdecode(buf, 1)
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    return lr_up, hr_bgr

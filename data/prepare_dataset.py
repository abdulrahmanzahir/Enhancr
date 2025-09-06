import os, glob, csv, cv2
from utils.degrade import random_degrade

SRC = 'data/raw_hr'
OUT = 'data/pairs'
os.makedirs(f'{OUT}/lr', exist_ok=True)
os.makedirs(f'{OUT}/hr', exist_ok=True)

rows = [['id','lr_path','hr_path']]
paths = sorted(glob.glob(f'{SRC}/*'))
for i, path in enumerate(paths):
    img = cv2.imread(path)
    if img is None: continue
    lr, hr = random_degrade(img, scale=2)
    lr_p = f'{OUT}/lr/{i:06d}.png'
    hr_p = f'{OUT}/hr/{i:06d}.png'
    cv2.imwrite(lr_p, lr)
    cv2.imwrite(hr_p, hr)
    rows.append([i, lr_p, hr_p])

with open(f'{OUT}/pairs.csv','w', newline='') as f:
    csv.writer(f).writerows(rows)

print('Prepared', len(rows)-1, 'pairs from', len(paths), 'HR images')

# data/prepare_dataset.py
import os, glob, csv, cv2, json, time, argparse, sys
from utils.degrade import random_degrade

# Ensure imports work when run as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXT

def prepare(src='data/raw_hr', out='data/pairs', variants=3, limit=None, min_scale=1.5, max_scale=4.0, progress_every=10):
    os.makedirs(f'{out}/lr', exist_ok=True)
    os.makedirs(f'{out}/hr', exist_ok=True)
    os.makedirs('data/manifests', exist_ok=True)

    # collect only supported images
    all_paths = [p for p in sorted(glob.glob(os.path.join(src, '*'))) if is_image(p)]
    if limit is not None:
        all_paths = all_paths[:limit]

    rows = [['id','lr_path','hr_path','meta_json']]
    pair_id = 0
    n = len(all_paths)
    t0 = time.time()

    if n == 0:
        print("No images found in", src)
        return

    print(f"Found {n} HR images. Creating {variants} variant(s) per image...")
    for idx, path in enumerate(all_paths, 1):
        hr = cv2.imread(path)
        if hr is None:
            print('Skip unreadable:', path)
            continue

        # Save HR once for the first pair of this source (use a unique id for each pair)
        # We'll write hr multiple times (one per pair) to keep lr/hr indices aligned.
        for k in range(variants):
            lr, hr_out, meta = random_degrade(hr, min_scale=min_scale, max_scale=max_scale)
            lr_p = f'{out}/lr/{pair_id:06d}.png'
            hr_p = f'{out}/hr/{pair_id:06d}.png'
            cv2.imwrite(hr_p, hr_out)
            cv2.imwrite(lr_p, lr)
            rows.append([pair_id, lr_p, hr_p, json.dumps(meta)])
            pair_id += 1

        if idx % progress_every == 0:
            elapsed = time.time() - t0
            print(f"[{idx}/{n}] pairs so far: {pair_id} | elapsed {elapsed:.1f}s", flush=True)

    with open(f'{out}/pairs.csv','w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)

    with open('data/manifests/pairs_summary.txt','w', encoding='utf-8') as f:
        f.write(f"HR source images: {n}\n")
        f.write(f"Total pairs: {pair_id}\n")
        f.write(f"Variants per HR: {variants}\n")

    print(f"Done. Total pairs: {pair_id}. Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/raw_hr')
    ap.add_argument('--out', default='data/pairs')
    ap.add_argument('--variants', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None, help='limit HR images for a quick run')
    ap.add_argument('--min-scale', type=float, default=1.5)
    ap.add_argument('--max-scale', type=float, default=4.0)
    ap.add_argument('--progress-every', type=int, default=10)
    args = ap.parse_args()
    prepare(args.src, args.out, args.variants, args.limit, args.min_scale, args.max_scale, args.progress_every)
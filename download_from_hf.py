"""
Download NWPU-RESISC45 from HuggingFace (jonathan-roberts1/NWPU-RESISC45)
and organize into ImageFolder layout.

Output: data/NWPU-RESISC45/{class_name}/{class_name}_{idx:03d}.jpg
        45 classes, 700 images each = 31,500 total (~550 MB)

Usage:
    python download_from_hf.py            # download + save
    python download_from_hf.py --verify   # check existing dataset
"""

import argparse
import time
import sys
from pathlib import Path

TARGET         = Path('data/NWPU-RESISC45')
N_CLASSES      = 45
IMGS_PER_CLASS = 700

# HF uses spaces; our folder names use underscores
HF_TO_FOLDER = {
    'airplane': 'airplane',
    'airport': 'airport',
    'baseball diamond': 'baseball_diamond',
    'basketball court': 'basketball_court',
    'beach': 'beach',
    'bridge': 'bridge',
    'chaparral': 'chaparral',
    'church': 'church',
    'circular farmland': 'circular_farmland',
    'cloud': 'cloud',
    'commercial area': 'commercial_area',
    'dense residential': 'dense_residential',
    'desert': 'desert',
    'forest': 'forest',
    'freeway': 'freeway',
    'golf course': 'golf_course',
    'ground track field': 'ground_track_field',
    'harbor': 'harbor',
    'industrial area': 'industrial_area',
    'intersection': 'intersection',
    'island': 'island',
    'lake': 'lake',
    'meadow': 'meadow',
    'medium residential': 'medium_residential',
    'mobile home park': 'mobile_home_park',
    'mountain': 'mountain',
    'overpass': 'overpass',
    'palace': 'palace',
    'parking lot': 'parking_lot',
    'railway': 'railway',
    'railway station': 'railway_station',
    'rectangular farmland': 'rectangular_farmland',
    'river': 'river',
    'roundabout': 'roundabout',
    'runway': 'runway',
    'sea ice': 'sea_ice',
    'ship': 'ship',
    'snowberg': 'snowberg',
    'sparse residential': 'sparse_residential',
    'stadium': 'stadium',
    'storage tank': 'storage_tank',
    'tennis court': 'tennis_court',
    'terrace': 'terrace',
    'thermal power station': 'thermal_power_station',
    'wetland': 'wetland',
}


def verify(target: Path) -> bool:
    if not target.exists():
        print(f'[FAIL] {target} does not exist.')
        return False
    class_dirs = [d for d in target.iterdir() if d.is_dir()]
    n_imgs = sum(
        len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png')))
        for d in class_dirs
    )
    print(f'Classes: {len(class_dirs)}/{N_CLASSES}, Images: {n_imgs}/{N_CLASSES*IMGS_PER_CLASS}')
    ok = len(class_dirs) >= N_CLASSES and n_imgs >= N_CLASSES * 50  # at least 50/class
    print('[OK] Dataset complete.' if ok else '[WARN] Dataset incomplete.')
    return ok


def download(target: Path):
    try:
        from datasets import load_dataset
    except ImportError:
        print('[FAIL] pip install datasets')
        return False

    print('Downloading from HuggingFace: jonathan-roberts1/NWPU-RESISC45')
    print('~550 MB, 31,500 images. Please wait ...\n')
    t0 = time.time()

    try:
        # Full download (not streaming) — downloads parquet shards once then serves locally
        print('Loading dataset (this caches to ~/.cache/huggingface/)...')
        ds = load_dataset('jonathan-roberts1/NWPU-RESISC45')
        # Combine all splits into one
        from datasets import concatenate_datasets
        parts = list(ds.values())
        full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        print(f'Total samples: {len(full)}')
    except Exception as e:
        print(f'[FAIL] {e}')
        print('\nTry setting HF_TOKEN for faster downloads:')
        print('  set HF_TOKEN=your_token_here')
        return False

    # Get class names from dataset features
    class_names = full.features['label'].names   # list of 45 names (with spaces)
    print(f'Classes: {class_names[:5]}... ({len(class_names)} total)\n')

    target.mkdir(parents=True, exist_ok=True)
    counters  = {}
    skipped   = 0
    n_total   = len(full)

    for i, sample in enumerate(full):
        img      = sample['image']
        lbl_idx  = sample['label']
        hf_name  = class_names[lbl_idx]
        folder   = HF_TO_FOLDER.get(hf_name, hf_name.replace(' ', '_'))

        cls_dir  = target / folder
        cls_dir.mkdir(exist_ok=True)

        n = counters.get(folder, 0) + 1
        out_path = cls_dir / f'{folder}_{n:03d}.jpg'

        if not out_path.exists():
            img.convert('RGB').save(str(out_path), 'JPEG', quality=95)
        else:
            skipped += 1

        counters[folder] = n

        if (i + 1) % 1000 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n_total - i - 1) / rate
            pct       = 100.0 * (i + 1) / n_total
            print(f'  {i+1:5d}/{n_total}  {pct:.1f}%  '
                  f'{rate:.0f} img/s  ~{remaining/60:.1f} min left')
            sys.stdout.flush()

    elapsed  = time.time() - t0
    saved    = sum(counters.values()) - skipped
    print(f'\nDone in {elapsed/60:.1f} min: {saved} saved, {skipped} already existed')
    return verify(target)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    p.add_argument('--target', default=str(TARGET))
    return p.parse_args()


def main():
    args   = parse_args()
    target = Path(args.target)

    if args.verify:
        verify(target)
        return

    if verify(target):
        print(f'\nDataset already complete. Run training:  python run_all.py')
        return

    ok = download(target)
    if ok:
        print('\nDataset ready! Next steps:')
        print('  python run_all.py')
    else:
        print('\n[FAIL] Download failed.')
        print('Manual option: download from https://huggingface.co/datasets/jonathan-roberts1/NWPU-RESISC45')
        print('               and extract to data/NWPU-RESISC45/')


if __name__ == '__main__':
    main()

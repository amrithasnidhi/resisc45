"""
Download NWPU-RESISC45 dataset.

Tries (in order):
  1. torchgeo automatic download
  2. Kaggle API (if kaggle.json configured)

Then symlinks / copies into data/NWPU-RESISC45/ for training scripts.

Usage:
    python download_dataset.py
    python download_dataset.py --verify   # just check existing dataset
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

TARGET = Path('data/NWPU-RESISC45')
N_CLASSES = 45


def check_complete(path: Path) -> bool:
    if not path.exists():
        return False
    classes = [d for d in path.iterdir() if d.is_dir()]
    if len(classes) < N_CLASSES:
        return False
    # Sample check: at least one class has images
    sample = list(classes[0].glob('*.jpg')) + list(classes[0].glob('*.png'))
    return len(sample) > 0


def try_torchgeo(target: Path) -> bool:
    """Try torchgeo NWPURESISC45 download."""
    print("[1] Trying torchgeo download ...")
    try:
        from torchgeo.datasets import NWPURESISC45
        dl_dir = target.parent
        dl_dir.mkdir(parents=True, exist_ok=True)
        # torchgeo downloads to root/NWPURESISC45/
        dataset = NWPURESISC45(root=str(dl_dir), download=True, split='train')
        print(f"    torchgeo downloaded dataset.")

        # torchgeo may use a different folder name — find it
        possible = list(dl_dir.glob('NWPU*')) + list(dl_dir.glob('nwpu*')) + list(dl_dir.glob('resisc*'))
        for p in possible:
            if p.is_dir() and p != target:
                print(f"    Renaming {p.name} -> {target.name}")
                if target.exists():
                    shutil.rmtree(target)
                p.rename(target)
                break

        return check_complete(target)
    except Exception as e:
        print(f"    torchgeo failed: {e}")
        return False


def try_kaggle(target: Path) -> bool:
    """Try Kaggle API download."""
    print("[2] Trying Kaggle download ...")
    try:
        import kaggle
    except ImportError:
        print("    kaggle not installed. Run: pip install kaggle")
        return False

    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print(f"    Kaggle credentials not found at {kaggle_json}")
        print("    Get your API key from https://www.kaggle.com/settings")
        print("    and place it at ~/.kaggle/kaggle.json")
        return False

    try:
        import subprocess
        dl_dir = target.parent
        dl_dir.mkdir(parents=True, exist_ok=True)
        # Common Kaggle dataset name for NWPU-RESISC45
        cmd = [
            sys.executable, '-m', 'kaggle', 'datasets', 'download',
            '-d', 'happysky16/nwpu-resisc45',
            '-p', str(dl_dir), '--unzip'
        ]
        print(f"    Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"    Kaggle error: {result.stderr}")
            return False

        # Find downloaded folder and rename to target
        for p in dl_dir.iterdir():
            if p.is_dir() and p != target and ('resisc' in p.name.lower() or 'nwpu' in p.name.lower()):
                print(f"    Moving {p} -> {target}")
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(p), str(target))
                break

        return check_complete(target)
    except Exception as e:
        print(f"    Kaggle download failed: {e}")
        return False


def print_manual_instructions():
    print("""
==============================================================
MANUAL DOWNLOAD REQUIRED
==============================================================

NWPU-RESISC45 is a research dataset that requires manual download.

Option A - Official OneDrive link (request from authors):
  Contact: Jun Wei Han  <junweihan@nwpu.edu.cn>
  Or check: https://github.com/0xAbdulKhalid/NWPU-RESISC45

Option B - Kaggle (if available):
  1. Install kaggle: pip install kaggle
  2. Set up ~/.kaggle/kaggle.json
  3. Run: python download_dataset.py

Option C - Direct mirror (if you have access):
  Download and extract so the structure is:
    data/NWPU-RESISC45/
      airplane/
        airplane_001.jpg
        airplane_002.jpg
        ...
      airport/
        airport_001.jpg
        ...
      ... (45 class folders, 700 images each = 31,500 total)

After placing the data, run:
    python download_dataset.py --verify

==============================================================
""")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true', help='Verify existing dataset only')
    p.add_argument('--target', default=str(TARGET), help='Target dataset directory')
    return p.parse_args()


def main():
    args = parse_args()
    target = Path(args.target)

    print(f"\nNWPU-RESISC45 Dataset Setup")
    print(f"Target: {target.absolute()}\n")

    if args.verify:
        if check_complete(target):
            classes = [d for d in target.iterdir() if d.is_dir()]
            imgs = sum(1 for d in classes for _ in d.glob('*.jpg'))
            imgs += sum(1 for d in classes for _ in d.glob('*.png'))
            print(f"[OK] Dataset complete: {len(classes)} classes, ~{imgs} images")
        else:
            print(f"[FAIL] Dataset incomplete or missing at {target}")
        return

    if check_complete(target):
        classes = [d for d in target.iterdir() if d.is_dir()]
        print(f"[OK] Dataset already at {target} ({len(classes)} classes). Nothing to do.")
        return

    # Try automatic downloads
    success = try_torchgeo(target) or try_kaggle(target)

    if success:
        classes = [d for d in target.iterdir() if d.is_dir()]
        imgs = sum(1 for d in classes for _ in d.glob('*.jpg'))
        print(f"\n[OK] Dataset ready: {len(classes)} classes, {imgs} images")
        print(f"     Location: {target.absolute()}")
        print("\nYou can now run training:")
        print("  python run_all.py")
    else:
        print_manual_instructions()


if __name__ == '__main__':
    main()

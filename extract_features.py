"""
Pre-extract and cache frozen Branch-A backbone features for fusion models.

This is critical for CPU training speed on i5-1335U:
  - Without caching: MobileViT-S/DenseNet-121 forward pass every batch (~5-10 min/epoch)
  - With caching: run backbone ONCE → save to disk → load tensors each epoch (~seconds)

Run this ONCE before train_fusion.py:
    python extract_features.py --experiment E11   # caches MobileViT-S features
    python extract_features.py --experiment E12   # same as E11 (shared backbone)
    python extract_features.py --experiment E13   # caches DenseNet-121 features
    python extract_features.py --experiment E14   # same as E13

Or cache all at once:
    python extract_features.py --all
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from shared_config import (
    DEVICE, DATA_DIR, SEED, CACHE_DIR, BACKBONE_SIZE,
    NUM_WORKERS, PIN_MEMORY,
)
from dataset import get_datasets, get_transform
from torch.utils.data import Subset
from torchvision import datasets
import torch.nn.functional as F
import timm


BACKBONE_MAP = {
    'E11': 'mobilevit_s',
    'E12': 'mobilevit_s',
    'E13': 'densenet121',
    'E14': 'densenet121',
}

FEAT_DIM_MAP = {
    'mobilevit_s': 640,
    'densenet121':  1024,
}


def _resize_collate(batch):
    """Custom collate that resizes images to BACKBONE_SIZE."""
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    if images.shape[-1] != BACKBONE_SIZE:
        images = F.interpolate(images, size=(BACKBONE_SIZE, BACKBONE_SIZE),
                               mode='bilinear', align_corners=False)
    return images, labels


def extract_and_save(experiment: str, data_dir=DATA_DIR, seed=SEED):
    backbone_name = BACKBONE_MAP[experiment]
    cache_key     = backbone_name   # E11 and E12 share the same cache

    print(f"\n[{experiment}] Backbone: {backbone_name}")
    print(f"  Cache dir: {CACHE_DIR}")

    # Check if already cached
    needed = all(
        (CACHE_DIR / f"{cache_key}_{split}.pt").exists()
        for split in ['train', 'val', 'test']
    )
    if needed:
        print(f"  ✓ Already cached – skipping.")
        return

    # Load datasets (eval transform only – no augmentation for caching)
    ds_eval = datasets.ImageFolder(data_dir,
                                   transform=get_transform('val', BACKBONE_SIZE))
    n = len(ds_eval)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val

    gen  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    splits = {
        'train': Subset(ds_eval, perm[:n_train]),
        'val':   Subset(ds_eval, perm[n_train : n_train + n_val]),
        'test':  Subset(ds_eval, perm[n_train + n_val :]),
    }

    # Load backbone (frozen)
    print(f"  Loading pretrained {backbone_name} …")
    backbone = timm.create_model(backbone_name, pretrained=True,
                                 num_classes=0, global_pool='avg')
    backbone.eval()
    backbone.to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad = False

    # Extract features split by split
    for split_name, subset in splits.items():
        cache_path = CACHE_DIR / f"{cache_key}_{split_name}.pt"
        if cache_path.exists():
            print(f"  ✓ {split_name}: already exists")
            continue

        loader = DataLoader(subset, batch_size=32, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        feats_list  = []
        labels_list = []
        t0 = time.time()
        print(f"  Extracting {split_name} ({len(subset)} samples) …", end='', flush=True)

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                # Already resized to BACKBONE_SIZE by transform
                feats = backbone(images).cpu()
                feats_list.append(feats)
                labels_list.append(labels)

        feats_tensor  = torch.cat(feats_list,  dim=0)   # (N, D)
        labels_tensor = torch.cat(labels_list, dim=0)   # (N,)

        torch.save({'feats': feats_tensor, 'labels': labels_tensor}, cache_path)
        elapsed = time.time() - t0
        print(f" done in {elapsed:.1f}s  → {cache_path.name}")

    print(f"  [OK] {backbone_name} features cached.\n")


def load_cached_feats(backbone_name: str, split: str, device=DEVICE):
    """Load previously cached features and labels."""
    path = CACHE_DIR / f"{backbone_name}_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run:  python extract_features.py --experiment <E11|E12|E13|E14>"
        )
    data = torch.load(path, map_location=device)
    return data['feats'], data['labels']


def parse_args():
    p = argparse.ArgumentParser(description='Cache frozen backbone features')
    p.add_argument('--experiment', choices=['E11', 'E12', 'E13', 'E14'],
                   help='Which experiment backbone to cache')
    p.add_argument('--all', action='store_true',
                   help='Cache all experiment backbones')
    p.add_argument('--data-dir', default=str(DATA_DIR))
    p.add_argument('--seed',     type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.all and args.experiment is None:
        print("Specify --experiment <E11|E12|E13|E14> or --all")
        return

    targets = ['E11', 'E12', 'E13', 'E14'] if args.all else [args.experiment]
    done    = set()
    for exp in targets:
        bname = BACKBONE_MAP[exp]
        if bname in done:
            print(f"[{exp}] Shares cache with earlier extraction → skip")
            continue
        extract_and_save(exp, args.data_dir, args.seed)
        done.add(bname)

    print("Feature extraction complete.")


if __name__ == '__main__':
    main()

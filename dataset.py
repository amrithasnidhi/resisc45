"""NWPU-RESISC45 dataset loading with reproducible train/val/test split."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
from shared_config import (
    DATA_DIR, IMG_SIZE, TRAIN_RATIO, VAL_RATIO, SEED,
    NUM_WORKERS, PIN_MEMORY, QUANTUM_BATCH_SIZE,
)


def get_transform(split: str, img_size: int = IMG_SIZE):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_datasets(data_dir=DATA_DIR, img_size=IMG_SIZE, seed=SEED):
    """Return (train_set, val_set, test_set) as Subset objects."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir}'.\n"
            "Download NWPU-RESISC45 and place it so that:\n"
            "  data/NWPU-RESISC45/airplane/*.jpg\n"
            "  data/NWPU-RESISC45/airport/*.jpg  ... etc."
        )

    # Build separate ImageFolder objects so each split can have its own transform
    ds_train = datasets.ImageFolder(data_dir, transform=get_transform('train', img_size))
    ds_eval  = datasets.ImageFolder(data_dir, transform=get_transform('val',   img_size))

    n       = len(ds_train)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    # Reproducible permutation
    gen  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val :]

    return (
        Subset(ds_train, train_idx),
        Subset(ds_eval,  val_idx),
        Subset(ds_eval,  test_idx),
    )


def get_loaders(
    data_dir=DATA_DIR,
    img_size=IMG_SIZE,
    batch_size=QUANTUM_BATCH_SIZE,
    seed=SEED,
):
    """Return (train_loader, val_loader, test_loader)."""
    train_set, val_set, test_set = get_datasets(data_dir, img_size, seed)

    common = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  **common)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, **common)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader


def get_class_names(data_dir=DATA_DIR):
    """Return sorted class names from the dataset folder."""
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())

"""
Train hybrid quantum-classical fusion models (E11–E14) on RESISC-45.

3-stage training:
  Stage 1 (5  epochs) : Only fusion head (Branch A + B frozen)
  Stage 2 (25 epochs) : Unfreeze Branch B + quantum layers (early stop)
  Stage 3 (5  epochs) : Very low LR polish of all unfrozen params

Usage (matches guide interface):
    python train_fusion.py --experiment E11 --dataset resisc45 --epochs 35
    python train_fusion.py --experiment E12 --fusion cross_attention
    python train_fusion.py --experiment E13 --num_heads 4
    python train_fusion.py --experiment E14

CPU-optimised defaults for HP Pavilion Plus (i5-1335U, 16 GB RAM, Intel Iris Xe).
Branch-A features are loaded from cache (run extract_features.py first).
"""

import argparse
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from shared_config import (
    DEVICE, DATA_DIR, IMG_SIZE, FUSION_BATCH_SIZE, SEED,
    FUSION_STAGE1_EPOCHS, FUSION_STAGE2_EPOCHS, FUSION_STAGE3_EPOCHS,
    EARLY_STOP_PATIENCE,
    LR_FUSION_S1, LR_FUSION_S2, LR_FUSION_S3, N_QUBITS, CACHE_DIR,
)
from dataset import get_loaders, get_class_names
from models import FUSION_MODELS
from extract_features import load_cached_feats, BACKBONE_MAP
from utils import (
    train_epoch, eval_epoch,
    save_checkpoint, load_checkpoint,
    CSVLogger, EarlyStopping,
    save_confusion_matrix, save_classification_report,
    save_model_report, compute_metrics,
    count_parameters, get_lr,
)


# Maps experiment -> which branch-A backbone cache key
BRANCH_A_CACHE = {
    'E11': 'mobilevit_s',
    'E12': 'mobilevit_s',
    'E13': 'densenet121',
    'E14': 'densenet121',
}

ARCH_NOTES = {
    'E11': ('MobileViT-S (frozen, 4.94M) -> Branch A feats\n'
            'ResNet-18 + 4Q PQC (StronglyEntangling, 2L) -> Branch B\n'
            'Fusion: Concatenation (640+4=644) -> Linear(644,128) -> Linear(128,45)'),
    'E12': ('MobileViT-S (frozen, 4.94M) -> Branch A feats\n'
            'ResNet-18 + 4Q PQC -> Branch B\n'
            'Fusion: Cross-Attention (Q=quantum, K=V=CNN, dim=128) + LayerNorm -> Linear(128,45)'),
    'E13': ('DenseNet-121 (frozen, 6.96M) -> Branch A feats\n'
            'MobileViT-S + 4Q PQC -> Branch B\n'
            'Fusion: Multi-Head Cross-Attention (4 heads, dim=256) + LayerNorm -> Linear(256,45)'),
    'E14': ('DenseNet-121 (frozen, 6.96M) -> Branch A feats\n'
            'MobileViT-S + 4Q PQC -> Branch B\n'
            'Fusion: Concatenation (1024+4=1028) -> Linear(1028,256) -> Linear(256,45)'),
}

MODEL_NAME_MAP = {
    'E11': 'E11_MobileViT_ResNet_Concat',
    'E12': 'E12_MobileViT_ResNet_CrossAttn',
    'E13': 'E13_DenseNet_MobileViT_CrossAttn',
    'E14': 'E14_DenseNet_MobileViT_Concat',
}


def parse_args():
    p = argparse.ArgumentParser(description='Train fusion model on RESISC-45')
    p.add_argument('--experiment', required=True, choices=['E11', 'E12', 'E13', 'E14'])
    p.add_argument('--dataset',     default='resisc45')
    p.add_argument('--data-dir',    default=str(DATA_DIR))
    p.add_argument('--epochs',      type=int, default=None,
                   help='Total epochs (overrides stage1+2+3 sum)')
    p.add_argument('--stage1-epochs', type=int, default=FUSION_STAGE1_EPOCHS)
    p.add_argument('--stage2-epochs', type=int, default=FUSION_STAGE2_EPOCHS)
    p.add_argument('--stage3-epochs', type=int, default=FUSION_STAGE3_EPOCHS)
    p.add_argument('--batch-size',  type=int,   default=FUSION_BATCH_SIZE)
    p.add_argument('--qubits',      type=int,   default=N_QUBITS)
    p.add_argument('--num_heads',   type=int,   default=4,
                   help='MHA heads for E13 (default 4)')
    p.add_argument('--seed',        type=int,   default=SEED)
    p.add_argument('--no-cache',    action='store_true',
                   help='Disable Branch-A feature caching (slower but no disk needed)')
    # Ignored args kept for guide-compatible CLI
    p.add_argument('--cnn_branch',       default=None)
    p.add_argument('--quantum_branch',   default=None)
    p.add_argument('--fusion',           default=None)
    p.add_argument('--dataset_name',     default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_stage(model, stage: int, experiment: str):
    """Freeze/unfreeze params according to training stage."""
    # Branch A is ALWAYS frozen
    if hasattr(model, 'branch_a'):
        for p in model.branch_a.parameters():
            p.requires_grad = False

    if stage == 1:
        # Freeze Branch B; only fusion head / classifier trains
        if hasattr(model, 'branch_b'):
            for p in model.branch_b.parameters():
                p.requires_grad = False

    elif stage in (2, 3):
        # Unfreeze Branch B (ResNet-18 / MobileViT-S) + quantum layers
        if hasattr(model, 'branch_b'):
            for p in model.branch_b.parameters():
                p.requires_grad = True


def make_optimizer(model, lr: float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=lr)


def _load_cache_split(experiment, split, use_cache, device=DEVICE):
    """Return cached Branch-A features tensor or None."""
    if not use_cache:
        return None
    bname = BRANCH_A_CACHE[experiment]
    try:
        feats, _ = load_cached_feats(bname, split, device)
        return feats
    except FileNotFoundError as e:
        print(f"\n[WARN] {e}")
        print("  Falling back to live inference (slower).\n")
        return None


def run_stage(model, stage, train_loader, val_loader,
              n_epochs, lr, criterion, logger, dataset,
              early_patience, global_epoch_offset,
              cache_train, cache_val, model_name, device=DEVICE):
    """Run one training stage. Returns (best_val_acc, epochs_run, final_epoch)."""

    set_stage(model, stage, '')
    optimizer = make_optimizer(model, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )
    early_stop  = EarlyStopping(patience=early_patience)
    best_val    = 0.0
    epoch_count = 0

    print(f"\n--- Stage {stage} | lr={lr:.0e} | epochs<={n_epochs} ---")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {trainable:,}")

    for ep in range(1, n_epochs + 1):
        ep_start = time.time()
        global_ep = global_epoch_offset + ep

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            cached_feats=cache_train,
        )
        vl_loss, vl_acc, _, _ = eval_epoch(
            model, val_loader, criterion, device,
            cached_feats=cache_val,
        )

        scheduler.step(vl_acc)
        elapsed = time.time() - ep_start
        lr_now  = get_lr(optimizer)

        logger.log(global_ep, f'stage{stage}', tr_loss, tr_acc,
                   vl_loss, vl_acc, lr_now, elapsed)

        print(f"  ep {global_ep:3d} [{stage}] "
              f"tr={tr_acc:.1f}% vl={vl_acc:.1f}% "
              f"lr={lr_now:.1e} {elapsed:.0f}s")

        if vl_acc > best_val:
            best_val = vl_acc
            save_checkpoint(model, optimizer, global_ep, vl_acc,
                            model_name, dataset)

        epoch_count += 1
        if stage == 2 and early_stop.step(vl_acc):
            print(f"  Early stop at stage-2 epoch {ep}.")
            break

    return best_val, epoch_count, global_epoch_offset + epoch_count


def main():
    args    = parse_args()
    exp     = args.experiment
    dataset = args.dataset.upper()
    set_seed(args.seed)

    model_name = MODEL_NAME_MAP[exp]
    use_cache  = not args.no_cache

    # Stage epochs
    if args.epochs is not None:
        # Divide total epochs proportionally: 1/7 : 5/7 : 1/7
        total = args.epochs
        s1 = max(1, total // 7)
        s3 = max(1, total // 7)
        s2 = total - s1 - s3
    else:
        s1, s2, s3 = args.stage1_epochs, args.stage2_epochs, args.stage3_epochs

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp}  ({model_name})")
    print(f"  Dataset    : {dataset}")
    print(f"  Stages     : {s1} + {s2} + {s3} = {s1+s2+s3} epochs")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Device     : {DEVICE}")
    print(f"  Cache B-A  : {use_cache}")
    print(f"{'='*60}\n")

    # -- Data ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_loaders(
        data_dir   = args.data_dir,
        img_size   = IMG_SIZE,
        batch_size = args.batch_size,
        seed       = args.seed,
    )
    class_names = get_class_names(args.data_dir)
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # -- Branch-A feature cache -------------------------------------------------
    cache_train = _load_cache_split(exp, 'train', use_cache)
    cache_val   = _load_cache_split(exp, 'val',   use_cache)
    cache_test  = _load_cache_split(exp, 'test',  use_cache)

    if cache_train is not None:
        print(f"Branch-A cache loaded: train={cache_train.shape}, "
              f"val={cache_val.shape}, test={cache_test.shape}")

    # -- Model -----------------------------------------------------------------
    ModelClass = FUSION_MODELS[exp]
    if exp == 'E13':
        model = ModelClass(n_qubits=args.qubits, n_heads=args.num_heads)
    else:
        model = ModelClass(n_qubits=args.qubits)
    model = model.to(DEVICE)

    total_params = count_parameters(model)
    print(f"Total parameters (trainable at some stage): {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    logger    = CSVLogger(model_name, dataset)
    t0        = time.time()

    # -- Stage 1 ---------------------------------------------------------------
    best_s1, n_s1, offset = run_stage(
        model, 1, train_loader, val_loader,
        s1, LR_FUSION_S1, criterion, logger, dataset,
        early_patience=s1,   # No early stop in stage 1
        global_epoch_offset=0,
        cache_train=cache_train, cache_val=cache_val,
        model_name=model_name,
    )

    # -- Stage 2 ---------------------------------------------------------------
    best_s2, n_s2, offset = run_stage(
        model, 2, train_loader, val_loader,
        s2, LR_FUSION_S2, criterion, logger, dataset,
        early_patience=EARLY_STOP_PATIENCE,
        global_epoch_offset=offset,
        cache_train=cache_train, cache_val=cache_val,
        model_name=model_name,
    )

    # -- Stage 3 ---------------------------------------------------------------
    best_s3, n_s3, _ = run_stage(
        model, 3, train_loader, val_loader,
        s3, LR_FUSION_S3, criterion, logger, dataset,
        early_patience=s3,
        global_epoch_offset=offset,
        cache_train=cache_train, cache_val=cache_val,
        model_name=model_name,
    )

    logger.close()
    total_time = time.time() - t0
    print(f"\nAll stages done in {total_time/3600:.2f} h")

    # -- Test evaluation -------------------------------------------------------
    print("\nLoading best checkpoint for test evaluation …")
    ckpt_path = Path('results/checkpoints') / f"{model_name}_{dataset}_best.pth"
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, DEVICE)

    _, test_acc, preds, labels = eval_epoch(
        model, test_loader, criterion, DEVICE, cached_feats=cache_test
    )
    metrics = compute_metrics(preds, labels)

    print(f"\nTest Accuracy : {metrics['accuracy']:.2f}%")
    print(f"Precision     : {metrics['precision']:.2f}%")
    print(f"Recall        : {metrics['recall']:.2f}%")
    print(f"F1-Score      : {metrics['f1']:.2f}%")
    print(f"Cohen Kappa   : {metrics['kappa']:.4f}")

    # -- Save outputs ----------------------------------------------------------
    save_classification_report(preds, labels, class_names, model_name, dataset)
    save_confusion_matrix(preds, labels, class_names, model_name, dataset)
    save_model_report(
        model_name   = model_name,
        dataset      = dataset,
        metrics      = metrics,
        arch_notes   = ARCH_NOTES.get(exp, ''),
        train_time_s = total_time,
        hyperparams  = {
            'stage1_epochs': n_s1,
            'stage2_epochs': n_s2,
            'stage3_epochs': n_s3,
            'batch_size':    args.batch_size,
            'lr_stage1':     LR_FUSION_S1,
            'lr_stage2':     LR_FUSION_S2,
            'lr_stage3':     LR_FUSION_S3,
            'qubits':        args.qubits,
            'cache_branch_a': use_cache,
        },
        n_params = total_params,
    )

    print(f"\n{'='*60}")
    print(f"  All outputs saved for {model_name}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

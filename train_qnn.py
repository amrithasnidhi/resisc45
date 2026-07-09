"""
Train pure quantum models (Q1, Q2, Q3) on RESISC-45.

Usage:
    python train_qnn.py --model Q1_QNN4EO   --dataset resisc45 --epochs 20 --qubits 4
    python train_qnn.py --model Q2_AngleSEL  --dataset resisc45 --epochs 20
    python train_qnn.py --model Q3_DataReupload --dataset resisc45 --epochs 20

CPU-optimised defaults for HP Pavilion Plus (i5-1335U, 16 GB RAM, Intel Iris Xe):
  - batch_size : 8
  - num_workers: 0
  - pin_memory : False
  - device     : cpu
"""

import argparse
import sys
import time
import random
import ctypes
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# Prevent Windows from sleeping during training
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
except Exception:
    pass

from shared_config import (
    DEVICE, DATA_DIR, IMG_SIZE, QUANTUM_BATCH_SIZE,
    QUANTUM_EPOCHS, SEED, LR_QUANTUM, N_QUBITS,
)
from dataset import get_loaders, get_class_names
from models import QUANTUM_MODELS
from utils import (
    train_epoch, eval_epoch,
    save_checkpoint, load_checkpoint,
    CSVLogger, EarlyStopping,
    save_confusion_matrix, save_classification_report,
    save_model_report, compute_metrics,
    count_parameters, get_lr,
)


def parse_args():
    p = argparse.ArgumentParser(description='Train quantum model on RESISC-45')
    p.add_argument('--model',      required=True,
                   choices=list(QUANTUM_MODELS.keys()),
                   help='Model name: Q1_QNN4EO | Q2_AngleSEL | Q3_DataReupload')
    p.add_argument('--dataset',    default='resisc45')
    p.add_argument('--data-dir',   default=str(DATA_DIR))
    p.add_argument('--epochs',     type=int, default=QUANTUM_EPOCHS)
    p.add_argument('--batch-size', type=int, default=QUANTUM_BATCH_SIZE)
    p.add_argument('--lr',         type=float, default=LR_QUANTUM)
    p.add_argument('--qubits',     type=int,   default=N_QUBITS)
    p.add_argument('--q-layers',   type=int,   default=2,
                   help='Quantum circuit depth (default: 2, Q3 uses 3 internally)')
    p.add_argument('--seed',       type=int,   default=SEED)
    p.add_argument('--resume',     default=None,
                   help='Path to checkpoint to resume from')
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args     = parse_args()
    dataset  = args.dataset.upper()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {dataset}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  LR      : {args.lr}")
    print(f"  Device  : {DEVICE}")
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

    # -- Model -----------------------------------------------------------------
    ModelClass = QUANTUM_MODELS[args.model]
    if args.model == 'Q3_DataReupload':
        model = ModelClass(n_qubits=args.qubits, n_layers=3)
    else:
        model = ModelClass(n_qubits=args.qubits, n_layers=args.q_layers)
    model = model.to(DEVICE)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}\n")

    # -- Optimizer & loss ------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # -- Resume ----------------------------------------------------------------
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(model, args.resume, DEVICE)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    # -- Training --------------------------------------------------------------
    logger       = CSVLogger(args.model, dataset)
    early_stop   = EarlyStopping(patience=10)   # More patience for quantum models
    t0           = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        ep_start = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step(vl_acc)
        elapsed = time.time() - ep_start
        lr_now  = get_lr(optimizer)

        logger.log(epoch, 'train', tr_loss, tr_acc, vl_loss, vl_acc, lr_now, elapsed)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.2f}% | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.2f}% | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt = save_checkpoint(model, optimizer, epoch, vl_acc,
                                   args.model, dataset)
            print(f"  [OK] Saved best checkpoint  (val_acc={vl_acc:.2f}%)")

        if early_stop.step(vl_acc):
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    logger.close()
    total_time = time.time() - t0
    print(f"\nTraining done in {total_time/3600:.2f} h")

    # -- Evaluation on test set ------------------------------------------------
    print("\nLoading best checkpoint for test evaluation …")
    ckpt_path = Path('results/checkpoints') / f"{args.model}_{dataset}_best.pth"
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, DEVICE)

    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion, DEVICE)
    metrics = compute_metrics(preds, labels)

    print(f"\nTest Accuracy : {metrics['accuracy']:.2f}%")
    print(f"Precision     : {metrics['precision']:.2f}%")
    print(f"Recall        : {metrics['recall']:.2f}%")
    print(f"F1-Score      : {metrics['f1']:.2f}%")
    print(f"Cohen Kappa   : {metrics['kappa']:.4f}")

    # -- Save outputs ----------------------------------------------------------
    save_classification_report(preds, labels, class_names, args.model, dataset)
    save_confusion_matrix(preds, labels, class_names, args.model, dataset)

    arch_map = {
        'Q1_QNN4EO':       'LeNet-style CNN -> 4 angles -> AngleEmbedding + StronglyEntanglingLayers (2L) -> Linear(4,45)',
        'Q2_AngleSEL':     'Quadrant CNN (x4, shared) -> per-quadrant PQC -> Cross-Attention fusion -> Linear(16,45)',
        'Q3_DataReupload': 'CNN encoder -> 4 angles -> Data-Reuploading circuit (3L, ring-CNOT, RY/RZ) -> Linear(4,45)',
    }
    save_model_report(
        model_name  = args.model,
        dataset     = dataset,
        metrics     = metrics,
        arch_notes  = arch_map.get(args.model, ''),
        train_time_s= total_time,
        hyperparams = {
            'epochs':      args.epochs,
            'batch_size':  args.batch_size,
            'lr':          args.lr,
            'qubits':      args.qubits,
            'optimizer':   'Adam',
            'scheduler':   'ReduceLROnPlateau(factor=0.5, patience=5)',
        },
        n_params    = n_params,
    )

    print(f"\n{'='*60}")
    print(f"  All outputs saved for {args.model}")
    print(f"  results/checkpoints/{args.model}_{dataset}_best.pth")
    print(f"  results/metrics/{args.model}_{dataset}_training_log.csv")
    print(f"  results/metrics/{args.model}_{dataset}_classification_report.txt")
    print(f"  results/confusion_matrices/{args.model}_{dataset}_cm.png")
    print(f"  results/model_reports/{args.model}_{dataset}_report.md")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    try:
        main()
    finally:
        # Re-enable sleep after training completes or errors
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        except Exception:
            pass

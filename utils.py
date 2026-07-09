"""Shared training utilities: loop, metrics, checkpointing, outputs."""

import csv
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    precision_score, recall_score, f1_score,
)

from shared_config import DEVICE, CHECKPOINT_DIR, METRICS_DIR, CM_DIR, REPORT_DIR


# -- Training / evaluation loops -----------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device=DEVICE,
                cached_feats=None):
    """Run one training epoch. Returns (avg_loss, accuracy %)."""
    model.train()
    total_loss = correct = total = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if cached_feats is not None:
            # Fusion models: pass cached Branch-A features
            batch_idx = slice(i * loader.batch_size,
                              min((i + 1) * loader.batch_size, len(cached_feats)))
            ca = cached_feats[batch_idx].to(device)
            outputs = model(images, cached_a=ca)
        else:
            outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device=DEVICE, cached_feats=None):
    """Run validation/test epoch. Returns (avg_loss, accuracy %, all_preds, all_labels)."""
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        if cached_feats is not None:
            batch_idx = slice(i * loader.batch_size,
                              min((i + 1) * loader.batch_size, len(cached_feats)))
            ca = cached_feats[batch_idx].to(device)
            outputs = model(images, cached_a=ca)
        else:
            outputs = model(images)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        100.0 * correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


# -- Checkpointing -------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, val_acc, model_name, dataset='RESISC45'):
    path = CHECKPOINT_DIR / f"{model_name}_{dataset}_best.pth"
    torch.save({
        'epoch':     epoch,
        'val_acc':   val_acc,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }, path)
    return path


def load_checkpoint(model, path, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    return ckpt.get('epoch', 0), ckpt.get('val_acc', 0.0)


# -- CSV logger ----------------------------------------------------------------

class CSVLogger:
    def __init__(self, model_name: str, dataset: str = 'RESISC45'):
        self.path = METRICS_DIR / f"{model_name}_{dataset}_training_log.csv"
        self._file   = open(self.path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ['epoch', 'stage', 'train_loss', 'train_acc',
             'val_loss', 'val_acc', 'lr', 'elapsed_s']
        )

    def log(self, epoch, stage, train_loss, train_acc, val_loss, val_acc,
            lr, elapsed_s):
        self._writer.writerow([
            epoch, stage,
            f"{train_loss:.4f}", f"{train_acc:.2f}",
            f"{val_loss:.4f}",   f"{val_acc:.2f}",
            f"{lr:.2e}", f"{elapsed_s:.1f}",
        ])
        self._file.flush()

    def close(self):
        self._file.close()


# -- Early stopping ------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_val   = -1.0
        self.counter    = 0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_val + self.min_delta:
            self.best_val = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# -- Output generation ---------------------------------------------------------

def save_confusion_matrix(preds, labels, class_names, model_name,
                          dataset='RESISC45'):
    cm   = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(22, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(f'{model_name} – Confusion Matrix ({dataset})', fontsize=14)
    plt.tight_layout()
    out_path = CM_DIR / f"{model_name}_{dataset}_cm.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_classification_report(preds, labels, class_names, model_name,
                                dataset='RESISC45'):
    report = classification_report(labels, preds, target_names=class_names,
                                   digits=4)
    out_path = METRICS_DIR / f"{model_name}_{dataset}_classification_report.txt"
    out_path.write_text(report, encoding='utf-8')
    return out_path


def compute_metrics(preds, labels):
    """Return dict of aggregate metrics."""
    return {
        'accuracy':  100.0 * (preds == labels).mean(),
        'precision': 100.0 * precision_score(labels, preds, average='macro', zero_division=0),
        'recall':    100.0 * recall_score(labels, preds, average='macro', zero_division=0),
        'f1':        100.0 * f1_score(labels, preds, average='macro', zero_division=0),
        'kappa':     cohen_kappa_score(labels, preds),
    }


def save_model_report(model_name: str, dataset: str, metrics: dict,
                      arch_notes: str, train_time_s: float,
                      hyperparams: dict, n_params: int):
    ts      = datetime.now().strftime('%Y-%m-%d %H:%M')
    content = f"""# Model Report: {model_name}

**Dataset**: {dataset}
**Generated**: {ts}
**Device**: CPU (Intel i5-1335U, 16 GB RAM, Intel Iris Xe)

---

## Architecture

{arch_notes}

**Parameters**: {n_params:,}

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
"""
    for k, v in hyperparams.items():
        content += f"| {k} | {v} |\n"

    content += f"""
---

## Results ({dataset} Test Set)

| Metric | Value |
|--------|-------|
| Accuracy  | {metrics['accuracy']:.2f}% |
| Precision | {metrics['precision']:.2f}% |
| Recall    | {metrics['recall']:.2f}% |
| F1-Score  | {metrics['f1']:.2f}% |
| Cohen κ   | {metrics['kappa']:.4f} |

**Training time**: {train_time_s/3600:.2f} h ({train_time_s:.0f} s)

---

## Output Files

- Checkpoint : `results/checkpoints/{model_name}_{dataset}_best.pth`
- Training log: `results/metrics/{model_name}_{dataset}_training_log.csv`
- Classification report: `results/metrics/{model_name}_{dataset}_classification_report.txt`
- Confusion matrix: `results/confusion_matrices/{model_name}_{dataset}_cm.png`
"""
    out_path = REPORT_DIR / f"{model_name}_{dataset}_report.md"
    out_path.write_text(content, encoding='utf-8')
    return out_path


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]['lr']

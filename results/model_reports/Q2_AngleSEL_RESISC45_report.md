# Model Report: Q2_AngleSEL

**Dataset**: RESISC45
**Generated**: 2026-07-10 19:04
**Device**: CPU (Intel i5-1335U, 16 GB RAM, Intel Iris Xe)

---

## Architecture

Quadrant CNN (x4, shared) -> per-quadrant PQC -> Cross-Attention fusion -> Linear(16,45)

**Parameters**: 39,881

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| epochs | 20 |
| batch_size | 8 |
| lr | 0.001 |
| qubits | 4 |
| optimizer | Adam |
| scheduler | ReduceLROnPlateau(factor=0.5, patience=5) |

---

## Results (RESISC45 Test Set)

| Metric | Value |
|--------|-------|
| Accuracy  | 40.30% |
| Precision | 43.20% |
| Recall    | 39.90% |
| F1-Score  | 37.76% |
| Cohen κ   | 0.3893 |

**Training time**: 11.53 h (41520 s)

---

## Output Files

- Checkpoint : `results/checkpoints/Q2_AngleSEL_RESISC45_best.pth`
- Training log: `results/metrics/Q2_AngleSEL_RESISC45_training_log.csv`
- Classification report: `results/metrics/Q2_AngleSEL_RESISC45_classification_report.txt`
- Confusion matrix: `results/confusion_matrices/Q2_AngleSEL_RESISC45_cm.png`

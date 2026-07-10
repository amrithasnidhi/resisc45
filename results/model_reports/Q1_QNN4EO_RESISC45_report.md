# Model Report: Q1_QNN4EO

**Dataset**: RESISC45
**Generated**: 2026-07-10 01:53
**Device**: CPU (Intel i5-1335U, 16 GB RAM, Intel Iris Xe)

---

## Architecture

LeNet-style CNN -> 4 angles -> AngleEmbedding + StronglyEntanglingLayers (2L) -> Linear(4,45)

**Parameters**: 1,277

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
| Accuracy  | 18.16% |
| Precision | 13.43% |
| Recall    | 18.15% |
| F1-Score  | 13.36% |
| Cohen κ   | 0.1631 |

**Training time**: 14.66 h (52776 s)

---

## Output Files

- Checkpoint : `results/checkpoints/Q1_QNN4EO_RESISC45_best.pth`
- Training log: `results/metrics/Q1_QNN4EO_RESISC45_training_log.csv`
- Classification report: `results/metrics/Q1_QNN4EO_RESISC45_classification_report.txt`
- Confusion matrix: `results/confusion_matrices/Q1_QNN4EO_RESISC45_cm.png`

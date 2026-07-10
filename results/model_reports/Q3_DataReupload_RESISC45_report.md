# Model Report: Q3_DataReupload

**Dataset**: RESISC45
**Generated**: 2026-07-10 20:46
**Device**: CPU (Intel i5-1335U, 16 GB RAM, Intel Iris Xe)

---

## Architecture

CNN encoder -> 4 angles -> Data-Reuploading circuit (3L, ring-CNOT, RY/RZ) -> Linear(4,45)

**Parameters**: 4,109

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
| Accuracy  | 22.39% |
| Precision | 19.28% |
| Recall    | 22.37% |
| F1-Score  | 17.36% |
| Cohen κ   | 0.2063 |

**Training time**: 1.63 h (5882 s)

---

## Output Files

- Checkpoint : `results/checkpoints/Q3_DataReupload_RESISC45_best.pth`
- Training log: `results/metrics/Q3_DataReupload_RESISC45_training_log.csv`
- Classification report: `results/metrics/Q3_DataReupload_RESISC45_classification_report.txt`
- Confusion matrix: `results/confusion_matrices/Q3_DataReupload_RESISC45_cm.png`

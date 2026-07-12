# Model Report: E11_MobileViT_ResNet_Concat

**Dataset**: RESISC45
**Generated**: 2026-07-12 09:33
**Device**: CPU (Intel i5-1335U, 16 GB RAM, Intel Iris Xe)

---

## Architecture

MobileViT-S (frozen, 4.94M) -> Branch A feats
ResNet-18 + 4Q PQC (StronglyEntangling, 2L) -> Branch B
Fusion: Concatenation (640+4=644) -> Linear(644,128) -> Linear(128,45)

**Parameters**: 11,266,953

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| stage1_epochs | 5 |
| stage2_epochs | 25 |
| stage3_epochs | 5 |
| batch_size | 16 |
| lr_stage1 | 0.001 |
| lr_stage2 | 0.0001 |
| lr_stage3 | 1e-05 |
| qubits | 4 |
| cache_branch_a | True |

---

## Results (RESISC45 Test Set)

| Metric | Value |
|--------|-------|
| Accuracy  | 33.21% |
| Precision | 32.87% |
| Recall    | 32.80% |
| F1-Score  | 31.38% |
| Cohen κ   | 0.3168 |

**Training time**: 21.76 h (78325 s)

---

## Output Files

- Checkpoint : `results/checkpoints/E11_MobileViT_ResNet_Concat_RESISC45_best.pth`
- Training log: `results/metrics/E11_MobileViT_ResNet_Concat_RESISC45_training_log.csv`
- Classification report: `results/metrics/E11_MobileViT_ResNet_Concat_RESISC45_classification_report.txt`
- Confusion matrix: `results/confusion_matrices/E11_MobileViT_ResNet_Concat_RESISC45_cm.png`

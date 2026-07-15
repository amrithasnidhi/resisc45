# Model Report: E12_MobileViT_ResNet_CrossAttn

**Dataset**: RESISC45
**Generated**: 2026-07-15 20:59
**Device**: CPU (Intel i5-1335U, 16 GB RAM, Intel Iris Xe)

---

## Architecture

MobileViT-S (frozen, 4.94M) -> Branch A feats
ResNet-18 + 4Q PQC -> Branch B
Fusion: Cross-Attention (Q=quantum, K=V=CNN, dim=128) + LayerNorm -> Linear(128,45)

**Parameters**: 11,349,385

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
| Accuracy  | 51.81% |
| Precision | 51.09% |
| Recall    | 51.52% |
| F1-Score  | 51.02% |
| Cohen κ   | 0.5071 |

**Training time**: 71.90 h (258824 s)

---

## Output Files

- Checkpoint : `results/checkpoints/E12_MobileViT_ResNet_CrossAttn_RESISC45_best.pth`
- Training log: `results/metrics/E12_MobileViT_ResNet_CrossAttn_RESISC45_training_log.csv`
- Classification report: `results/metrics/E12_MobileViT_ResNet_CrossAttn_RESISC45_classification_report.txt`
- Confusion matrix: `results/confusion_matrices/E12_MobileViT_ResNet_CrossAttn_RESISC45_cm.png`

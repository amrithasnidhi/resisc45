# 🚀 Person 2 - RESISC-45 Training Guide

**Mission**: Train Quantum Neural Networks and Hybrid Quantum-Classical Fusion models on RESISC-45 dataset (45 classes)

**Expected Outputs** (same as existing models):
1. ✅ Model checkpoint (`.pth`)
2. ✅ Training log (`.csv`)
3. ✅ Classification report (`.txt`)
4. ✅ Confusion matrix (`.png`)
5. ✅ Model report (`.md`)

---

## 📋 Models to Train (7 models total)

### **Pure Quantum Models (3 models)**
1. **Q1_QNN4EO** - ESA baseline quantum model
2. **Q2_AngleSEL** - Quadrant PQC with cross-attention
3. **Q3_DataReupload** - Data reuploading quantum circuit

### **Hybrid Quantum-Classical Fusion (4 models)**
4. **E11_MobileViT_ResNet_Concat** - MobileViT-S + (ResNet-18 + 4Q PQC) - Concatenation
5. **E12_MobileViT_ResNet_CrossAttn** - MobileViT-S + (ResNet-18 + 4Q PQC) - Cross-Attention
6. **E13_DenseNet_MobileViT_CrossAttn** - DenseNet-121 + (MobileViT-S + 4Q PQC) - Cross-Attention
7. **E14_DenseNet_MobileViT_Concat** - DenseNet-121 + (MobileViT-S + 4Q PQC) - Concatenation

---

## 📊 Expected Deliverables

After training all models, Person 2 will provide two comprehensive metrics tables:

### **Table 1: Pure Quantum Neural Networks (3 rows)**

```
| Model | Classical Encoder | Quantum Ansatz | Dataset | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Kappa | Params (M) | GFLOPs | Qubits | Inference (ms) |
|-------|-------------------|----------------|---------|--------------|---------------|-----------|-------------|--------|-----------|--------|--------|----------------|
| Q1_QNN4EO | LeNet-style CNN (Conv2d ×3 → Linear(8,4)) | AngleEmbedding(Y) + StronglyEntanglingLayers (2 layers) | RESISC-45 | ... | ... | ... | ... | ... | ... | ... | 4 | ... |
| Q2_AngleSEL | Shared per-quadrant CNN (Conv2d ×2, BN, MaxPool, Linear(32,4)) | AngleEmbedding(Y) per quadrant + StronglyEntanglingLayers (2 layers, shared) | RESISC-45 | ... | ... | ... | ... | ... | ... | ... | 4 | ... |
| Q3_DataReupload | CNN (Conv2d ×3 + BN + AdaptiveAvgPool → Linear(16,4)) | Data Reuploading: AngleEmbed(Y) + ring CNOT + RY/RZ (3 layers) | RESISC-45 | ... | ... | ... | ... | ... | ... | ... | 4 | ... |
```

### **Table 2: Hybrid Quantum-Classical Fusion Models (4 rows)**

```
| Model | CNN Branch A | Quantum Branch B | Fusion Type | Qubits | Dataset | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Kappa | Params (M) | GFLOPs | Inference (ms) |
|-------|--------------|------------------|-------------|--------|---------|--------------|---------------|-----------|-------------|--------|-----------|--------|----------------|
| E11_MobileViT_ResNet_Concat | MobileViT-S | ResNet-18 + 4Q PQC | Concatenation (parallel concat, mean-pool → MLP) | 4 | RESISC-45 | ... | ... | ... | ... | ... | ... | ... | ... |
| E12_MobileViT_ResNet_CrossAttn | MobileViT-S | ResNet-18 + 4Q PQC | Cross-Attention (Q=quantum, K=V=CNN, residual + LN) | 4 | RESISC-45 | ... | ... | ... | ... | ... | ... | ... | ... |
| E13_DenseNet_MobileViT_CrossAttn | DenseNet-121 | MobileViT-S + 4Q PQC | Cross-Attention (Q=quantum, K=V=CNN, MHA 4 heads) | 4 | RESISC-45 | ... | ... | ... | ... | ... | ... | ... | ... |
| E14_DenseNet_MobileViT_Concat | DenseNet-121 | MobileViT-S + 4Q PQC | Concatenation (mean-pool + concat → MLP) | 4 | RESISC-45 | ... | ... | ... | ... | ... | ... | ... | ... |
```

### **Plus All Output Files:**
For each of the 7 models, the following files must be generated:
1. ✅ **Checkpoint**: `results/checkpoints/{model}_RESISC45_best.pth`
2. ✅ **Training Log**: `results/metrics/{model}_RESISC45_training_log.csv`
3. ✅ **Classification Report**: `results/metrics/{model}_RESISC45_classification_report.txt`
4. ✅ **Confusion Matrix**: `results/confusion_matrices/{model}_RESISC45_cm.png`
5. ✅ **Model Report**: `results/model_reports/{model}_RESISC45_report.md`

**Total Files: 7 models × 5 outputs = 35 files**

---

## 🎯 Model Architecture Details

### **Q1: QNN4EO (ESA Baseline)**

**Architecture:**
```
Input (B, 3, 64, 64)
    ↓
LeNet-style CNN Encoder:
  - Conv2d(3, 4, kernel=3) + ReLU
  - Conv2d(4, 8, kernel=3) + ReLU
  - Conv2d(8, 8, kernel=3) + ReLU
  - Flatten → (B, 8*58*58)
    ↓
Linear(8, 4) → (B, 4) angles
    ↓
Quantum Circuit (4 qubits):
  - AngleEmbedding (RY rotations)
  - StronglyEntanglingLayers (2 layers)
  - Measure: PauliZ → (B, 4) expectations
    ↓
Linear(4, 45) → (B, 45) logits
```

**Key Parameters:**
- Qubits: 4
- Quantum Layers: 2
- Encoder: LeNet-style (lightweight)
- Total Params: ~0.008M

---

### **Q2: AngleSEL (Quadrant PQC + Cross-Attention)**

**Architecture:**
```
Input (B, 3, 64, 64)
    ↓
Split into 4 quadrants (32×32 each)
    ↓
Shared CNN Encoder per quadrant:
  - Conv2d(3, 16, kernel=3, padding=1) + BN + ReLU
  - Conv2d(16, 32, kernel=3, padding=1) + BN + ReLU
  - MaxPool2d(2, 2)
  - Flatten + Linear(32*16*16, 4) → 4 angles per quadrant
    ↓
4 Quantum Circuits (shared weights):
  - AngleEmbedding (RY)
  - StronglyEntanglingLayers (2 layers)
  - Measure: PauliZ → 4 expectations per quadrant
    ↓
Cross-Attention Fusion:
  - Q: quantum features (4×4)
  - K, V: CNN features (4×32)
  - Output: (B, 16)
    ↓
Linear(16, 45) → (B, 45) logits
```

**Key Parameters:**
- Qubits: 4 (per quadrant, shared circuit)
- Quantum Layers: 2
- Fusion: Cross-Attention
- Total Params: ~0.019M

---

### **Q3: DataReupload (Data Reuploading Circuit)**

**Architecture:**
```
Input (B, 3, 64, 64)
    ↓
CNN Encoder:
  - Conv2d(3, 8, kernel=3) + BN + ReLU
  - Conv2d(8, 16, kernel=3) + BN + ReLU
  - Conv2d(16, 16, kernel=3) + BN + ReLU
  - AdaptiveAvgPool2d(1) → (B, 16)
    ↓
Linear(16, 4) → (B, 4) angles
    ↓
Quantum Circuit (4 qubits, 3 layers):
  Layer 1:
    - AngleEmbedding(angles)
    - Ring CNOT (0→1→2→3→0)
    - RY(θ₁), RZ(φ₁) trainable
  Layer 2:
    - Re-upload angles
    - Ring CNOT
    - RY(θ₂), RZ(φ₂) trainable
  Layer 3:
    - Re-upload angles
    - Ring CNOT
    - RY(θ₃), RZ(φ₃) trainable
    ↓
Measure: PauliZ → (B, 4) expectations
    ↓
Linear(4, 45) → (B, 45) logits
```

**Key Parameters:**
- Qubits: 4
- Quantum Layers: 3 (data reuploading)
- Total Params: ~0.029M

---

### **E11: MobileViT-S + (ResNet-18 + 4Q PQC) - Concatenation**

**Architecture:**
```
Input (B, 3, 64, 64)
    ↓
Branch A: MobileViT-S (pretrained, frozen)
  → Global features: (B, 640)
    ↓
Branch B: ResNet-18 + 4Q PQC
  - ResNet-18 backbone → (B, 512)
  - Linear(512, 4) + Tanh → angles
  - Quantum Circuit: AngleEmbedding + StronglyEntanglingLayers (2 layers)
  → Quantum features: (B, 4) expectations
    ↓
Fusion: Concatenation
  - Mean pool: (B, 640) → (B, 640)
  - Quantum: (B, 4) → (B, 4)
  - Concat: (B, 644)
    ↓
MLP Head:
  - Linear(644, 128) + ReLU
  - Dropout(0.3)
  - Linear(128, 45) → (B, 45) logits
```

**Key Parameters:**
- Branch A: MobileViT-S (4.94M, frozen)
- Branch B: ResNet-18 + PQC (11.18M)
- Fusion: Concatenation
- Total Params: ~16.12M

---

### **E12: MobileViT-S + (ResNet-18 + 4Q PQC) - Cross-Attention**

**Architecture:**
```
Input (B, 3, 64, 64)
    ↓
Branch A: MobileViT-S → (B, 640)
Branch B: ResNet-18 + 4Q PQC → (B, 4)
    ↓
Cross-Attention Fusion:
  - Q: Quantum features (B, 4) → Linear(4, 128) → (B, 128)
  - K, V: CNN features (B, 640) → Linear(640, 128) → (B, 128)
  - Attention: Softmax(QK^T/√d) V
  - Residual connection: Q + Attention
  - LayerNorm → (B, 128)
    ↓
Linear(128, 45) → (B, 45) logits
```

**Key Parameters:**
- Fusion: Cross-Attention (Q=quantum, K=V=CNN)
- Attention dim: 128
- Total Params: ~16.19M

---

### **E13: DenseNet-121 + (MobileViT-S + 4Q PQC) - Cross-Attention**

**Architecture:**
```
Input (B, 3, 64, 64)
    ↓
Branch A: DenseNet-121 (pretrained, frozen) → (B, 1024)
Branch B: MobileViT-S + 4Q PQC
  - MobileViT-S → (B, 640)
  - Linear(640, 4) + Tanh → angles
  - Quantum Circuit → (B, 4) expectations
    ↓
Cross-Attention Fusion (Multi-Head, 4 heads):
  - Q: Quantum (B, 4) → Linear(4, 256) → (B, 256)
  - K, V: DenseNet (B, 1024) → Linear(1024, 256) → (B, 256)
  - Multi-Head Attention (4 heads, dim=64 each)
  - Residual + LayerNorm → (B, 256)
    ↓
Linear(256, 45) → (B, 45) logits
```

**Key Parameters:**
- Branch A: DenseNet-121 (6.96M, frozen)
- Branch B: MobileViT-S + PQC (4.94M + quantum)
- Fusion: Multi-Head Cross-Attention (4 heads)
- Total Params: ~12.02M

---

### **E14: DenseNet-121 + (MobileViT-S + 4Q PQC) - Concatenation**

**Architecture:**
```
Input (B, 3, 64, 64)
    ↓
Branch A: DenseNet-121 → (B, 1024)
Branch B: MobileViT-S + 4Q PQC → (B, 4)
    ↓
Fusion: Concatenation
  - Mean pool DenseNet: (B, 1024) → (B, 1024)
  - Quantum: (B, 4) → (B, 4)
  - Concat: (B, 1028)
    ↓
MLP Head:
  - Linear(1028, 256) + ReLU
  - Dropout(0.3)
  - Linear(256, 45) → (B, 45) logits
```

**Key Parameters:**
- Fusion: Simple concatenation + MLP
- Total Params: ~11.97M

---

## 🛠️ Setup Instructions

### **Step 1: Verify Dataset**

```bash
# Check if RESISC-45 dataset exists
python -c "from pathlib import Path; print('✅ Found' if Path('data/NWPU-RESISC45').exists() else '❌ Missing')"
```

### **Step 2: Install Dependencies**

```bash
pip install torch torchvision pennylane pennylane-lightning matplotlib seaborn scikit-learn timm
```

**Required:**
- Python: 3.8+
- PyTorch: 1.13+
- PennyLane: 0.30+
- timm: 0.9.0+

---

## 🎮 Training Instructions

### **Pure Quantum Models**

Use the existing quantum training pipeline from the repo:

```bash
# Q1: QNN4EO
python train_qnn.py --model Q1_QNN4EO --dataset resisc45 --epochs 30 --qubits 4

# Q2: AngleSEL
python train_qnn.py --model Q2_AngleSEL --dataset resisc45 --epochs 30 --qubits 4

# Q3: DataReupload
python train_qnn.py --model Q3_DataReupload --dataset resisc45 --epochs 30 --qubits 4
```

### **Hybrid Fusion Models**

Use the CSGA training pipeline (adapt for your fusion types):

```bash
# E11: MobileViT + ResNet + Concat
python train_fusion.py \
  --experiment E11 \
  --cnn_branch mobilevit_s \
  --quantum_branch resnet18_pqc \
  --fusion concatenation \
  --dataset resisc45 \
  --epochs 65

# E12: MobileViT + ResNet + Cross-Attention
python train_fusion.py \
  --experiment E12 \
  --cnn_branch mobilevit_s \
  --quantum_branch resnet18_pqc \
  --fusion cross_attention \
  --dataset resisc45 \
  --epochs 65

# E13: DenseNet + MobileViT + Cross-Attention (MHA)
python train_fusion.py \
  --experiment E13 \
  --cnn_branch densenet121 \
  --quantum_branch mobilevit_pqc \
  --fusion cross_attention_mha \
  --num_heads 4 \
  --dataset resisc45 \
  --epochs 65

# E14: DenseNet + MobileViT + Concat
python train_fusion.py \
  --experiment E14 \
  --cnn_branch densenet121 \
  --quantum_branch mobilevit_pqc \
  --fusion concatenation \
  --dataset resisc45 \
  --epochs 65
```

---

## ⚡ Training Tips

### **1. Training Time Estimates**
- **Q1-Q3 (Pure Quantum)**: 8-12 hours each (quantum circuits are slow)
- **E11-E14 (Hybrid Fusion)**: 10-15 hours each (3-stage training)

### **2. GPU Memory**
- Use batch_size=16 for quantum models (larger batches cause OOM)
- Hybrid models can use batch_size=32

### **3. Checkpointing**
All models use 3-stage training:
- Stage 1: Head-only (5 epochs)
- Stage 2: Joint fine-tune (up to 50 epochs, early stop patience=10)
- Stage 3: Final fine-tune (5-10 epochs)

Best checkpoint saved automatically.

### **4. Expected Performance (RESISC-45)**
- **Q1-Q3**: 30-50% (pure quantum struggles with 45 classes)
- **E11-E14**: 88-92% (fusion helps significantly)

---

## 📈 Progress Tracking

| Model | Type | Status | Accuracy | Training Time | Notes |
|-------|------|--------|----------|---------------|-------|
| Q1_QNN4EO | Quantum | ⬜ TODO | -- | -- | Pure quantum baseline |
| Q2_AngleSEL | Quantum | ⬜ TODO | -- | -- | Quadrant approach |
| Q3_DataReupload | Quantum | ⬜ TODO | -- | -- | Data reuploading |
| E11_MobileViT_ResNet_Concat | Fusion | ⬜ TODO | -- | -- | Concatenation fusion |
| E12_MobileViT_ResNet_CrossAttn | Fusion | ⬜ TODO | -- | -- | Cross-attention fusion |
| E13_DenseNet_MobileViT_CrossAttn | Fusion | ⬜ TODO | -- | -- | MHA 4 heads |
| E14_DenseNet_MobileViT_Concat | Fusion | ⬜ TODO | -- | -- | Simple concat |

---

## 📊 Output Verification

After each model trains, verify all 5 outputs exist:

```bash
# Example: Check Q1 outputs
ls -lh results/checkpoints/Q1_QNN4EO_RESISC45_best.pth
ls -lh results/metrics/Q1_QNN4EO_RESISC45_training_log.csv
ls -lh results/metrics/Q1_QNN4EO_RESISC45_classification_report.txt
ls -lh results/confusion_matrices/Q1_QNN4EO_RESISC45_cm.png
ls -lh results/model_reports/Q1_QNN4EO_RESISC45_report.md
```

---

## 🚨 Common Issues

### **Issue 1: Quantum Circuit Too Slow**
**Solution**: Use `lightning.qubit` device (fastest), reduce batch size to 8-16

### **Issue 2: Fusion Dimension Mismatch**
**Solution**: Check Linear projection dimensions in fusion layer match branch outputs

### **Issue 3: OOM with Hybrid Models**
**Solution**:
- Freeze backbones (already frozen in architecture)
- Reduce batch size: `--batch-size 16`
- Use gradient checkpointing

### **Issue 4: Poor Quantum Performance**
**Expected**: Pure quantum models (Q1-Q3) will have lower accuracy (30-50%) on RESISC-45 due to:
- 45 classes (hard for 4-qubit circuits)
- Domain shift from EuroSAT
- Limited quantum expressivity

Hybrid models (E11-E14) should perform much better (88-92%).

---

## 📦 Final Deliverables

When all models are trained, you should have:

**7 models × 5 outputs = 35 files**

```
results/
├── checkpoints/          (7 .pth files)
├── metrics/              (14 files: 7 logs + 7 reports)
├── confusion_matrices/   (7 .png files)
└── model_reports/        (7 .md files)
```

**Plus 2 metrics tables** (as shown above)

---

## 🎯 Success Criteria

✅ All 7 models trained (3 quantum + 4 fusion)
✅ All 35 output files generated
✅ Quantum models: 30-50% accuracy (expected for pure quantum)
✅ Fusion models: 88-92% accuracy
✅ Training logs show convergence
✅ Confusion matrices saved
✅ Complete metrics tables provided

---

## 💬 Key Differences from Person 1

**Person 1**: Classical CNNs (simple architectures)
**Person 2**: Quantum NNs + Hybrid Fusion (complex architectures)

**Challenges**:
- Quantum circuits are 10-20× slower than classical CNNs
- 3-stage training required for fusion models
- Lower accuracy expected for pure quantum models
- More GPU memory needed for hybrid models

**Advantages**:
- Novel architectures (quantum + fusion)
- Better interpretability (quantum circuits)
- Potential for quantum advantage in future

---

## 📚 Reference Files

Check existing models for examples:
- `results/model_reports/Q4_ResNetPQC_report.md` - Quantum baseline example
- `results/model_reports/E9_QuantumFiLM_report.md` - Fusion model example
- `shared_config.py` - Configuration and utilities
- `train.py` - Main training pipeline

---

**Good luck, Person 2! 🚀**

**You're training the most advanced models in this project!**

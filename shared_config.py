# Configuration for RESISC-45 training
# Device: HP Pavilion Plus 14-ew0xxx | i5-1335U | 16GB RAM | Intel Iris Xe (integrated, no CUDA)

import torch
from pathlib import Path

# ── Device ──────────────────────────────────────────────────────────────────
# Intel Iris Xe is integrated GPU sharing system RAM — no CUDA support.
# All training runs on CPU.
DEVICE = torch.device('cpu')

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path('data/NWPU-RESISC45')
RESULTS_DIR     = Path('results')
CHECKPOINT_DIR  = RESULTS_DIR / 'checkpoints'
METRICS_DIR     = RESULTS_DIR / 'metrics'
CM_DIR          = RESULTS_DIR / 'confusion_matrices'
REPORT_DIR      = RESULTS_DIR / 'model_reports'
CACHE_DIR       = RESULTS_DIR / 'feature_cache'

for _d in [CHECKPOINT_DIR, METRICS_DIR, CM_DIR, REPORT_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
NUM_CLASSES = 45
IMG_SIZE    = 64        # All models accept 64×64 input
BACKBONE_SIZE = 224     # Pretrained backbones resized internally to this

RESISC45_CLASSES = [
    'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
    'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
    'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
    'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
    'intersection', 'island', 'lake', 'meadow', 'medium_residential',
    'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot',
    'railway', 'railway_station', 'rectangular_farmland', 'river',
    'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg',
    'sparse_residential', 'stadium', 'storage_tank', 'tennis_court',
    'terrace', 'thermal_power_station', 'wetland',
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

# ── DataLoader ───────────────────────────────────────────────────────────────
# num_workers=0 avoids Windows multiprocessing spawn issues on CPU
NUM_WORKERS = 0
PIN_MEMORY  = False     # Only meaningful with CUDA

# Batch sizes adapted for CPU RAM (16 GB total, shared with OS)
QUANTUM_BATCH_SIZE = 8   # Small: quantum circuits serialize per batch
FUSION_BATCH_SIZE  = 16  # Moderate: backbone features cached to avoid OOM

# ── Training epochs (adapted from guide for CPU speed) ───────────────────────
# Guide default: Quantum=30, Fusion=65 (GPU-assumed)
# CPU default: Quantum=20, Fusion=35 (5+25+5 stages)
QUANTUM_EPOCHS        = 20
FUSION_STAGE1_EPOCHS  = 5   # Head-only warm-up
FUSION_STAGE2_EPOCHS  = 25  # Joint fine-tune (early stop patience=8)
FUSION_STAGE3_EPOCHS  = 5   # Low-LR final polish
EARLY_STOP_PATIENCE   = 8

# ── Learning rates ────────────────────────────────────────────────────────────
LR_QUANTUM    = 1e-3
LR_FUSION_S1  = 1e-3    # Stage 1: head only
LR_FUSION_S2  = 1e-4    # Stage 2: joint fine-tune
LR_FUSION_S3  = 1e-5    # Stage 3: final polish

# ── Quantum ───────────────────────────────────────────────────────────────────
N_QUBITS = 4

# Pick fastest available PennyLane backend for CPU
try:
    import pennylane as qml
    _test_dev = qml.device('lightning.qubit', wires=1)
    QUANTUM_BACKEND = 'lightning.qubit'
    DIFF_METHOD     = 'adjoint'          # Fastest for lightning.qubit
except Exception:
    QUANTUM_BACKEND = 'default.qubit'
    DIFF_METHOD     = 'best'             # Falls back to parameter-shift

"""
Quick sanity-check script — run this FIRST before any training.

Checks:
  1. Python + PyTorch version
  2. Device (confirms CPU, no CUDA for Intel Iris Xe)
  3. PennyLane + lightning.qubit availability
  4. timm availability and key pretrained model names
  5. Dataset existence and class count
  6. Output directory structure
  7. A tiny forward pass through each model (random data, no training)

Usage:
    python verify_setup.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def check(label, fn):
    try:
        result = fn()
        print(f"  ✓  {label}: {result}")
        return True
    except Exception as e:
        print(f"  ✗  {label}: {e}")
        return False


print("\n" + "=" * 60)
print("  RESISC-45 Setup Verification")
print("=" * 60)

# 1. Python
print("\n[1] Python & PyTorch")
check("Python", lambda: sys.version.split()[0])

import_ok = True
try:
    import torch
    check("PyTorch",    lambda: torch.__version__)
    check("CUDA avail", lambda: str(torch.cuda.is_available()) + "  (expected False on Intel Iris Xe)")
    check("CPU threads", lambda: str(torch.get_num_threads()))
except ImportError as e:
    print(f"  ✗  PyTorch not found: {e}")
    import_ok = False

# 2. PennyLane
print("\n[2] PennyLane")
try:
    import pennylane as qml
    check("PennyLane version", lambda: qml.__version__)

    try:
        dev = qml.device('lightning.qubit', wires=4)
        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def test_circuit(x):
            qml.AngleEmbedding(x, wires=range(4))
            qml.StronglyEntanglingLayers(
                qml.numpy.random.random(qml.StronglyEntanglingLayers.shape(2, 4)),
                wires=range(4)
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        out = test_circuit(qml.numpy.array([0.1, 0.2, 0.3, 0.4]))
        print(f"  ✓  lightning.qubit (adjoint): circuit output shape = {len(out)}")
    except Exception as e:
        print(f"  ⚠  lightning.qubit unavailable ({e})")
        print(f"       Install: pip install pennylane-lightning")
        print(f"       Falling back to default.qubit (slower)")

except ImportError:
    print("  ✗  PennyLane not found. Install: pip install pennylane pennylane-lightning")

# 3. timm
print("\n[3] timm (pretrained backbones)")
try:
    import timm
    check("timm version", lambda: timm.__version__)

    for mname in ['mobilevit_s', 'densenet121', 'resnet18']:
        ok = mname in timm.list_models(pretrained=True)
        print(f"  {'✓' if ok else '✗'}  {mname} pretrained: {ok}")

except ImportError:
    print("  ✗  timm not found. Install: pip install timm")

# 4. Other deps
print("\n[4] Other dependencies")
for pkg in ['sklearn', 'matplotlib', 'seaborn', 'numpy']:
    try:
        mod = __import__(pkg)
        print(f"  ✓  {pkg}: {mod.__version__}")
    except Exception:
        print(f"  ✗  {pkg}: not found — pip install scikit-learn matplotlib seaborn numpy")

# 5. Dataset
print("\n[5] Dataset")
from shared_config import DATA_DIR, RESISC45_CLASSES
data_path = Path(DATA_DIR)
if data_path.exists():
    classes = sorted(p.name for p in data_path.iterdir() if p.is_dir())
    n_imgs  = sum(1 for p in data_path.rglob('*')
                  if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    print(f"  ✓  Dataset found: {data_path}")
    print(f"     Classes: {len(classes)} (expected 45)")
    print(f"     Images : {n_imgs} (expected ~31500)")
    if len(classes) != 45:
        print(f"  ⚠  Expected 45 classes, found {len(classes)}")
        missing = set(RESISC45_CLASSES) - set(classes)
        if missing:
            print(f"     Missing: {missing}")
else:
    print(f"  ✗  Dataset NOT found at: {data_path}")
    print(f"     Download NWPU-RESISC45 and unzip to:  {data_path}/")
    print(f"     Expected structure: {data_path}/airplane/*.jpg  etc.")

# 6. Output dirs
print("\n[6] Output directories")
from shared_config import CHECKPOINT_DIR, METRICS_DIR, CM_DIR, REPORT_DIR, CACHE_DIR
for d in [CHECKPOINT_DIR, METRICS_DIR, CM_DIR, REPORT_DIR, CACHE_DIR]:
    status = '✓' if d.exists() else '✗'
    print(f"  {status}  {d}")

# 7. Quick model forward pass (random data, no training)
print("\n[7] Model smoke test (random batch, no training)")
if import_ok and 'torch' in dir():
    try:
        from shared_config import DEVICE
        x = torch.randn(2, 3, 64, 64)   # tiny batch of 2

        from models.quantum_models import Q1_QNN4EO, Q2_AngleSEL, Q3_DataReupload
        for ModelCls, name in [
            (Q1_QNN4EO,       'Q1_QNN4EO'),
            (Q2_AngleSEL,     'Q2_AngleSEL'),
            (Q3_DataReupload, 'Q3_DataReupload'),
        ]:
            t = time.time()
            m   = ModelCls().to(DEVICE)
            out = m(x)
            elapsed = time.time() - t
            print(f"  ✓  {name}: output={tuple(out.shape)}  ({elapsed:.2f}s)")
            del m

    except Exception as e:
        print(f"  ✗  Quantum model smoke test failed: {e}")

    try:
        # Only test fusion model IF timm is available
        import timm
        from models.fusion_models import E14_DenseNet_MobileViT_Concat
        print("\n  Testing E14 (smallest fusion model) …")
        t = time.time()
        m   = E14_DenseNet_MobileViT_Concat().to(DEVICE)
        out = m(x)
        elapsed = time.time() - t
        print(f"  ✓  E14_DenseNet_MobileViT_Concat: output={tuple(out.shape)}  ({elapsed:.2f}s)")
        del m
    except Exception as e:
        print(f"  ⚠  Fusion model smoke test skipped/failed: {e}")

print("\n" + "=" * 60)
print("  Verification complete.  Fix any ✗ issues before training.")
print("=" * 60 + "\n")

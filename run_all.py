"""
Run all 7 models sequentially.

Order:
  1. Extract Branch-A features (once, ~10-30 min)
  2. Q1_QNN4EO      (20 epochs, ~1-4 h on CPU)
  3. Q2_AngleSEL    (20 epochs, ~2-6 h on CPU)
  4. Q3_DataReupload (20 epochs, ~1-4 h on CPU)
  5. E11            (35 epochs, ~1-2 h with cache)
  6. E12            (35 epochs, ~1-2 h with cache)
  7. E13            (35 epochs, ~2-4 h with cache)
  8. E14            (35 epochs, ~2-4 h with cache)

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-quantum     # Only fusion models (assumes Q done)
    python run_all.py --skip-fusion      # Only quantum models
    python run_all.py --quantum-epochs 10 --fusion-epochs 20  # Faster run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  STARTING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status  = "[OK] SUCCESS" if result.returncode == 0 else "[FAIL] FAILED"
    print(f"\n{status}: {label}  ({elapsed/3600:.2f} h)\n")
    return result.returncode == 0


def parse_args():
    p = argparse.ArgumentParser(description='Run all RESISC-45 training')
    p.add_argument('--skip-quantum',    action='store_true')
    p.add_argument('--skip-fusion',     action='store_true')
    p.add_argument('--quantum-epochs',  type=int, default=20)
    p.add_argument('--fusion-epochs',   type=int, default=35)
    p.add_argument('--batch-size',      type=int, default=None)
    p.add_argument('--data-dir',        default='data/NWPU-RESISC45')
    p.add_argument('--skip-extract',    action='store_true',
                   help='Skip feature extraction (if cache already exists)')
    return p.parse_args()


def main():
    args     = parse_args()
    py       = sys.executable
    results  = {}
    t_total  = time.time()

    # -- Feature extraction for fusion models ----------------------------------
    if not args.skip_fusion and not args.skip_extract:
        ok = run_cmd(
            [py, 'extract_features.py', '--all', '--data-dir', args.data_dir],
            'Feature extraction (Branch-A backbones)',
        )
        results['extract'] = ok

    # -- Quantum models --------------------------------------------------------
    if not args.skip_quantum:
        for model_name in ['Q1_QNN4EO', 'Q2_AngleSEL', 'Q3_DataReupload']:
            cmd = [
                py, 'train_qnn.py',
                '--model',   model_name,
                '--dataset', 'resisc45',
                '--epochs',  str(args.quantum_epochs),
                '--data-dir', args.data_dir,
            ]
            if args.batch_size:
                cmd += ['--batch-size', str(args.batch_size)]
            results[model_name] = run_cmd(cmd, model_name)

    # -- Fusion models ---------------------------------------------------------
    if not args.skip_fusion:
        for exp in ['E11', 'E12', 'E13', 'E14']:
            cmd = [
                py, 'train_fusion.py',
                '--experiment', exp,
                '--dataset',    'resisc45',
                '--epochs',     str(args.fusion_epochs),
                '--data-dir',   args.data_dir,
            ]
            if args.batch_size:
                cmd += ['--batch-size', str(args.batch_size)]
            results[exp] = run_cmd(cmd, exp)

    # -- Summary ---------------------------------------------------------------
    total_h = (time.time() - t_total) / 3600
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY  (total: {total_h:.2f} h)")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {'[OK]' if ok else '[FAIL]'}  {name}")
    print()

    # Check output files
    print("Output file check:")
    all_names = []
    if not args.skip_quantum:
        all_names += ['Q1_QNN4EO', 'Q2_AngleSEL', 'Q3_DataReupload']
    if not args.skip_fusion:
        all_names += [
            'E11_MobileViT_ResNet_Concat',
            'E12_MobileViT_ResNet_CrossAttn',
            'E13_DenseNet_MobileViT_CrossAttn',
            'E14_DenseNet_MobileViT_Concat',
        ]

    ds = 'RESISC45'
    total_files = expected_files = 0
    for mname in all_names:
        files = [
            Path(f'results/checkpoints/{mname}_{ds}_best.pth'),
            Path(f'results/metrics/{mname}_{ds}_training_log.csv'),
            Path(f'results/metrics/{mname}_{ds}_classification_report.txt'),
            Path(f'results/confusion_matrices/{mname}_{ds}_cm.png'),
            Path(f'results/model_reports/{mname}_{ds}_report.md'),
        ]
        for f in files:
            expected_files += 1
            if f.exists():
                total_files += 1
                print(f"  [OK]  {f}")
            else:
                print(f"  [FAIL]  {f}  (MISSING)")

    print(f"\n  Files: {total_files}/{expected_files}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

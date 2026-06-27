"""
UUS Data Preparation (v2)
=========================
Searches /kaggle/input/ for MNIST, Fashion-MNIST, and CIFAR-10 data
regardless of the exact dataset slug or filename.

Usage:
  Step 1: Run the discovery cell FIRST to see what you have:
      !ls -R /kaggle/input/ | head -80

  Step 2: Then run this script:
      !python prepare_data.py
"""

import os, shutil, glob
import numpy as np
from pathlib import Path

OUT = Path("/kaggle/working/uus-data")
if not Path("/kaggle").exists():
    OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

INPUT = Path("/kaggle/input")


def find_file(patterns):
    """Return the first existing path that matches any glob pattern."""
    for pat in patterns:
        hits = sorted(glob.glob(str(INPUT / pat)))
        if hits:
            return Path(hits[0])
    return None


# ============================================================
# 1. MNIST
# ============================================================
def prepare_mnist():
    src = find_file([
        "mnist-in-csv/mnist_train.csv",
        "mnist-in-csv/train.csv",
        "mnist*/mnist_train.csv",
        "mnist*/train.csv",
        "*mnist*/*train*.csv",
        "digit-recognizer/train.csv",
    ])
    if src:
        dst = OUT / "mnist_train.csv"
        shutil.copy(src, dst)
        print(f"[MNIST] OK  {src}  ->  {dst}")
    else:
        print("[MNIST] NOT FOUND.")
        print("   Fix: In your notebook, click  + Add data  and search for:")
        print('         "mnist-in-csv"  by oddrationale')
        print("   URL: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")


# ============================================================
# 2. Fashion-MNIST
# ============================================================
def prepare_fashion_mnist():
    src = find_file([
        "fashionmnist/fashion-mnist_train.csv",
        "fashionmnist/train.csv",
        "fashion-mnist/fashion-mnist_train.csv",
        "fashion*mnist*/*train*.csv",
        "*fashion*/*train*.csv",
    ])
    if src:
        dst = OUT / "fashion_mnist_train.csv"
        shutil.copy(src, dst)
        print(f"[Fashion-MNIST] OK  {src}  ->  {dst}")
    else:
        print("[Fashion-MNIST] NOT FOUND.")
        print("   Fix: In your notebook, click  + Add data  and search for:")
        print('         "fashionmnist"  by zalando-research')
        print("   URL: https://www.kaggle.com/datasets/zalando-research/fashionmnist")


# ============================================================
# 3. CIFAR-10
# ============================================================
def prepare_cifar10():
    # A) Check for an already-projected NPZ
    npz = find_file(["*cifar*features*.npz", "*cifar*/*.npz"])
    if npz:
        dst = OUT / "cifar10_features.npz"
        shutil.copy(npz, dst)
        print(f"[CIFAR-10] OK (npz)  {npz}  ->  {dst}")
        return

    # B) Look for a CSV with pixel columns
    csv_src = find_file([
        "cifar10-python-in-csv/train.csv",
        "cifar10-python-in-csv/cifar10_train.csv",
        "cifar-10-training-set-csv/*.csv",
        "*cifar*/*train*.csv",
        "*cifar*/*.csv",
    ])
    if csv_src:
        import pandas as pd
        print(f"[CIFAR-10] Found CSV: {csv_src}")
        print("           Projecting to 512-d (may take ~30s)...")
        df = pd.read_csv(csv_src)
        if "label" in df.columns:
            y = df["label"].values.astype(int)
            X = df.drop(columns=["label"]).values.astype(np.float32)
        else:
            y = df.iloc[:, 0].values.astype(int)
            X = df.iloc[:, 1:].values.astype(np.float32)

        if X.max() > 1.0:
            X = X / 255.0

        rng = np.random.RandomState(0)
        proj = rng.randn(X.shape[1], 512).astype(np.float32) / np.sqrt(512)
        X_proj = X @ proj

        dst = OUT / "cifar10_features.npz"
        np.savez_compressed(dst, X=X_proj, y=y)
        print(f"           Saved {dst}  shape={X_proj.shape}")
        return

    # C) Look for pickle / batch files (original CIFAR-10 Python format)
    batch = find_file([
        "*cifar*/*data_batch*",
        "*cifar*/*batch*",
        "*cifar*/cifar-10-batches-py/data_batch_1",
    ])
    if batch:
        import pickle
        print(f"[CIFAR-10] Found pickle batches near: {batch}")
        batch_dir = batch.parent
        X_list, y_list = [], []
        for i in range(1, 6):
            bp = batch_dir / f"data_batch_{i}"
            if not bp.exists():
                bp = batch_dir / "cifar-10-batches-py" / f"data_batch_{i}"
            if bp.exists():
                with open(bp, "rb") as f:
                    d = pickle.load(f, encoding="bytes")
                X_list.append(d[b"data"].astype(np.float32) / 255.0)
                y_list.append(np.array(d[b"labels"], dtype=int))
        if X_list:
            X = np.concatenate(X_list)
            y = np.concatenate(y_list)
            rng = np.random.RandomState(0)
            proj = rng.randn(X.shape[1], 512).astype(np.float32) / np.sqrt(512)
            X_proj = X @ proj
            dst = OUT / "cifar10_features.npz"
            np.savez_compressed(dst, X=X_proj, y=y)
            print(f"           Saved {dst}  shape={X_proj.shape}")
            return

    print("[CIFAR-10] NOT FOUND.")
    print("   Fix: In your notebook, click  + Add data  and search for:")
    print('         "CIFAR-10 Python in CSV"  by fedesoriano')
    print("   URL: https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv")


# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("UUS Data Preparation")
    print("=" * 55)

    if INPUT.exists():
        mounted = sorted([d.name for d in INPUT.iterdir() if d.is_dir()])
        if mounted:
            print(f"\nMounted datasets in /kaggle/input/:")
            for m in mounted:
                print(f"  - {m}/")
        else:
            print("\n  (no datasets mounted yet)")
    print()

    prepare_mnist()
    prepare_fashion_mnist()
    prepare_cifar10()

    print(f"\n{'=' * 55}")
    print(f"Output directory: {OUT}/")
    ready = list(OUT.iterdir())
    if ready:
        for f in sorted(ready):
            mb = f.stat().st_size / 1e6
            print(f"  {f.name}  ({mb:.1f} MB)")
        if len(ready) == 3:
            print("\nAll 3 datasets ready. Run:")
            print("  !python UUS_Experiments.py")
        else:
            print(f"\n{len(ready)}/3 found. Add the missing dataset(s) above, then re-run.")
    else:
        print("  (empty, no datasets found)")
    print("=" * 55)

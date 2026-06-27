"""
Unlearning Under Siege (UUS): Experiments
=========================================
Paper 3 of the Trustworthy Machine Unlearning Trilogy.

Investigates how Newton-step certified unlearning degrades adversarial robustness.
All experiments are CPU-only (Kaggle free tier compatible).

Experiments:
  1. Robustness Erosion Curves (random vs adversarial vs improving deletions)
  2. Margin Analysis (empirical vs theoretical bound)
  3. Noise-Robustness Tradeoff (certified removal noise vs adversarial accuracy)
  4. Targeted Attack After Unlearning (attack success rate before/after)

Outputs:
  - JSON results for each experiment
  - CSV summary tables
  - PDF figures (fig1_erosion_curves.pdf, fig2_margin_analysis.pdf,
    fig3_noise_robustness.pdf, fig4_targeted_attack.pdf)

Usage:
  python UUS_Experiments.py
"""

import numpy as np
import json
import csv
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# DATA DISCOVERY
# Auto-find CSV/NPZ files anywhere under /kaggle/input/ or ./data/
# No need for prepare_data.py; this script finds everything itself.
# -------------------------------------------------------------------
import glob as _glob

def _find_file(*patterns):
    """Search /kaggle/input/ recursively for the first file matching any pattern."""
    search_roots = ["/kaggle/input", "/kaggle/working", "data"]
    for root in search_roots:
        for pat in patterns:
            hits = sorted(_glob.glob(f"{root}/**/{pat}", recursive=True))
            if hits:
                return hits[0]
    return None

LAMBDA = 0.001         # L2 regularization (lower = larger per-deletion shifts)
N_SAMPLES = 2000       # samples per class (total 2*N_SAMPLES per dataset)
DELETION_STEPS = [5, 10, 25, 50, 100, 150, 200]
PGD_STEPS = 40
PGD_STEP_SIZE = 0.005
NOISE_EPSILONS = [0.5, 1.0, 2.0, 5.0]
N_RANDOM_TRIALS = 5    # repeated random trials for error bars

# PGD epsilon per dataset (L_inf)
PGD_EPS = {
    "MNIST": 0.1,
    "FashionMNIST": 0.1,
    "CIFAR10_features": 0.03,
}


# ============================================================
# Dataset Loading
# ============================================================

def _load_csv_dataset(filepath, label_col, pos_label, neg_label, n_per_class=N_SAMPLES):
    """
    Generic loader for CSV datasets (MNIST, Fashion-MNIST format).
    Auto-detects label column by name or position.
    Returns X in [0,1], y in {-1, +1}.
    """
    import pandas as pd
    print(f"  Loading {filepath}...")
    df = pd.read_csv(filepath)

    # Auto-detect label column
    label_candidates = ["label", "Label", "class", "Class", "target", "Target"]
    label_name = None
    for lc in label_candidates:
        if lc in df.columns:
            label_name = lc
            break

    if label_name:
        labels = df[label_name].values.astype(int)
        pixels = df.drop(columns=[label_name]).values.astype(np.float64)
    else:
        labels = df.iloc[:, label_col].values.astype(int)
        pixels = df.iloc[:, 1:].values.astype(np.float64) if label_col == 0 else df.drop(columns=[df.columns[label_col]]).values.astype(np.float64)

    mask = (labels == pos_label) | (labels == neg_label)
    X, raw_y = pixels[mask], labels[mask]
    y = np.where(raw_y == pos_label, 1, -1)

    # Normalize to [0,1] if values are in 0-255 range
    if X.max() > 1.0:
        X = X / 255.0

    # Subsample to n_per_class per class
    idx_pos = np.where(y == 1)[0][:n_per_class]
    idx_neg = np.where(y == -1)[0][:n_per_class]
    idx = np.concatenate([idx_pos, idx_neg])
    np.random.shuffle(idx)
    print(f"    Selected {len(idx)} samples (d={X.shape[1]})")
    return X[idx], y[idx]


def load_mnist_binary(digit_a=3, digit_b=8, n_per_class=N_SAMPLES):
    """
    Find and load MNIST binary (digit_a vs digit_b).
    Searches for any CSV with 'mnist' in the name (not 'fashion').
    """
    path = _find_file("mnist_train.csv", "mnist_test.csv")
    if path is None:
        # Broader search: any csv with 'mnist' but not 'fashion'
        for root in ["/kaggle/input", "/kaggle/working", "data"]:
            hits = sorted(_glob.glob(f"{root}/**/*.csv", recursive=True))
            for h in hits:
                h_lower = h.lower()
                if "mnist" in h_lower and "fashion" not in h_lower:
                    path = h
                    break
            if path:
                break
    if path is None:
        raise FileNotFoundError(
            "MNIST CSV not found anywhere under /kaggle/input/.\n"
            "Please add a dataset containing mnist_train.csv."
        )
    return _load_csv_dataset(path, label_col=0, pos_label=digit_a, neg_label=digit_b, n_per_class=n_per_class)


def load_fashion_mnist_binary(class_a=0, class_b=1, n_per_class=N_SAMPLES):
    """
    Find and load Fashion-MNIST binary.
    Searches for any CSV with 'fashion' in the name.
    """
    path = _find_file("fashion-mnist_train.csv", "fashion_mnist_train.csv")
    if path is None:
        for root in ["/kaggle/input", "/kaggle/working", "data"]:
            hits = sorted(_glob.glob(f"{root}/**/*.csv", recursive=True))
            for h in hits:
                if "fashion" in h.lower():
                    path = h
                    break
            if path:
                break
    if path is None:
        raise FileNotFoundError(
            "Fashion-MNIST CSV not found anywhere under /kaggle/input/.\n"
            "Please add a dataset containing fashion-mnist_train.csv."
        )
    return _load_csv_dataset(path, label_col=0, pos_label=class_a, neg_label=class_b, n_per_class=n_per_class)


def load_cifar10_features(class_a=0, class_b=1, n_per_class=N_SAMPLES):
    """
    Find and load CIFAR-10 features.
    Searches for NPZ first (pre-projected), then CSV (raw pixels, will project).
    """
    # Try NPZ
    npz_path = _find_file("cifar10_features.npz", "cifar*features*.npz")
    if npz_path:
        print(f"  Loading pre-extracted CIFAR-10 features from {npz_path}...")
        data = np.load(npz_path)
        X_all, y_all = data["X"].astype(np.float64), data["y"].astype(int)
    else:
        # Try CSV: any csv with 'cifar' in the name
        csv_path = _find_file("cifar10_train.csv", "train.csv")
        if csv_path is None:
            for root in ["/kaggle/input", "/kaggle/working", "data"]:
                hits = sorted(_glob.glob(f"{root}/**/*.csv", recursive=True))
                for h in hits:
                    if "cifar" in h.lower():
                        csv_path = h
                        break
                if csv_path:
                    break
        if csv_path is None:
            raise FileNotFoundError(
                "CIFAR-10 data not found anywhere under /kaggle/input/.\n"
                "Please add a dataset containing CIFAR-10 CSV or NPZ."
            )
        print(f"  Loading raw CIFAR-10 from {csv_path}, projecting to 512-d...")
        import pandas as pd
        df = pd.read_csv(csv_path)

        # Auto-detect label column
        label_candidates = ["label", "Label", "class", "Class", "target"]
        label_name = None
        for lc in label_candidates:
            if lc in df.columns:
                label_name = lc
                break
        if label_name:
            y_all = df[label_name].values.astype(int)
            X_raw = df.drop(columns=[label_name]).values.astype(np.float64)
        else:
            y_all = df.iloc[:, 0].values.astype(int)
            X_raw = df.iloc[:, 1:].values.astype(np.float64)

        if X_raw.max() > 1.0:
            X_raw = X_raw / 255.0

        # Random projection to 512 dimensions
        rng = np.random.RandomState(0)
        proj = rng.randn(X_raw.shape[1], 512) / np.sqrt(512)
        X_all = X_raw @ proj
        print(f"    Projected {X_raw.shape[1]}-d -> 512-d")

    mask = (y_all == class_a) | (y_all == class_b)
    X, raw_y = X_all[mask], y_all[mask]
    y = np.where(raw_y == class_a, 1, -1)

    idx_pos = np.where(y == 1)[0][:n_per_class]
    idx_neg = np.where(y == -1)[0][:n_per_class]
    idx = np.concatenate([idx_pos, idx_neg])
    np.random.shuffle(idx)
    print(f"    Selected {len(idx)} samples (d={X.shape[1]})")
    return X[idx], y[idx]


DATASET_LOADERS = {
    "MNIST": load_mnist_binary,
    "FashionMNIST": load_fashion_mnist_binary,
    "CIFAR10_features": load_cifar10_features,
}


# ============================================================
# Core Functions
# ============================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(X, y, lam=LAMBDA):
    """Train L2-regularized logistic regression, return weight vector."""
    n, d = X.shape
    clf = LogisticRegression(
        C=1.0 / (n * lam),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-8,
    )
    clf.fit(X, y)
    w = clf.coef_.flatten()
    return w


def compute_hessian(X, y, w, lam=LAMBDA):
    """Compute Hessian of regularized logistic loss."""
    n, d = X.shape
    z = y * (X @ w)
    s = sigmoid(z)
    diag = s * (1 - s)  # second derivative of logistic loss
    H = (X.T * diag) @ X / n + lam * np.eye(d)
    return H


def compute_gradient(X, y, w):
    """Compute per-sample gradients of logistic loss."""
    n, d = X.shape
    z = y * (X @ w)
    s = sigmoid(z)
    # grad_i = -(1 - s_i) * y_i * x_i
    coeff = -(1 - s) * y
    grads = X * coeff[:, np.newaxis]  # (n, d)
    return grads


def newton_step_delete(w, X, y, j, H_inv, lam=LAMBDA):
    """Compute Newton-step parameter update for deleting point j."""
    n = X.shape[0]
    grad_j = compute_gradient(X[j:j+1], y[j:j+1], w).flatten()
    delta_j = (1.0 / n) * H_inv @ grad_j
    return delta_j


def geometric_margin(w, X, y):
    """Compute geometric margin: y * w^T x / ||w|| for each point."""
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return np.zeros(len(y))
    return (y * (X @ w)) / w_norm


def min_margin(w, X, y):
    """Minimum margin over the test set."""
    margins = geometric_margin(w, X, y)
    return np.min(margins)


def compute_margin_gradient(w, x, y_val):
    """
    Gradient of margin m(x; w) = y * w^T x / ||w|| with respect to w.
    Returns d-dimensional vector.
    """
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return np.zeros_like(w)
    wtx = w @ x
    grad = y_val / w_norm * (x - (wtx / w_norm**2) * w)
    return grad


def robustness_influence(w, X_train, y_train, X_test, y_test, H_inv, lam=LAMBDA, top_k_test=50):
    """
    Compute SIGNED robustness influence for each training point.

    Positive IF_rob(z_j) means deleting z_j INCREASES the min margin (good).
    Negative IF_rob(z_j) means deleting z_j DECREASES the min margin (bad).

    The adversary should pick the most NEGATIVE values (margin-reducing).
    The defender should pick the most POSITIVE values (margin-improving).

    We focus on the most vulnerable test points (smallest margin).
    For each training point, we compute the margin change at the single
    most vulnerable test point (the one with the minimum margin).
    """
    n_train = X_train.shape[0]
    margins = geometric_margin(w, X_test, y_test)

    # Focus on the most vulnerable test points (smallest margin, including negative)
    vulnerable_idx = np.argsort(margins)[:top_k_test]
    X_vuln = X_test[vulnerable_idx]
    y_vuln = y_test[vulnerable_idx]

    # Precompute margin gradients for vulnerable points
    margin_grads = np.array([
        compute_margin_gradient(w, X_vuln[i], y_vuln[i])
        for i in range(len(vulnerable_idx))
    ])  # (top_k_test, d)

    # Compute signed influence for each training point
    # grad_margin^T @ delta_j > 0 means deletion increases margin (good)
    # grad_margin^T @ delta_j < 0 means deletion decreases margin (bad)
    influences = np.zeros(n_train)
    for j in range(n_train):
        delta_j = newton_step_delete(w, X_train, y_train, j, H_inv, lam)
        # Signed inner product: how much does deleting j shift the margin?
        # Take the MINIMUM across vulnerable points (worst-case for defender)
        signed_changes = margin_grads @ delta_j  # (top_k_test,)
        influences[j] = np.min(signed_changes)   # most margin-reducing effect

    return influences


def pgd_attack(w, x, y_val, epsilon=0.1, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    """
    PGD attack for logistic regression.
    Find x_adv that minimizes y * w^T x_adv subject to ||x_adv - x||_inf <= epsilon.
    """
    x_adv = x.copy()
    for _ in range(steps):
        logit = w @ x_adv
        prob = sigmoid(y_val * logit)
        # Gradient of loss w.r.t. x_adv: -(1-prob) * y * w
        grad_x = -(1 - prob) * y_val * w
        # Gradient ascent on loss (to find adversarial example)
        x_adv = x_adv + step_size * np.sign(grad_x)
        # Project back to L_inf ball
        x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
        # Clip to valid range [0, 1] for image data
        x_adv = np.clip(x_adv, 0, 1)
    return x_adv


def adversarial_accuracy(w, X, y, epsilon=0.1, max_samples=500):
    """Compute adversarial accuracy under PGD attack. Uses fixed seed for reproducibility."""
    n = min(len(y), max_samples)
    rng_eval = np.random.RandomState(12345)  # fixed seed for consistent evaluation
    idx = rng_eval.choice(len(y), n, replace=False)
    correct = 0
    for i in idx:
        x_adv = pgd_attack(w, X[i], y[i], epsilon=epsilon)
        pred = np.sign(w @ x_adv)
        if pred == 0:
            pred = 1
        if pred == y[i]:
            correct += 1
    return correct / n


def clean_accuracy(w, X, y):
    """Clean (non-adversarial) accuracy."""
    preds = np.sign(X @ w)
    preds[preds == 0] = 1
    return np.mean(preds == y)


def certified_removal_noise_sigma(lam, n, B, eps_priv):
    """
    Noise standard deviation for certified removal (Guo et al. 2020).
    sigma = B / (n * lam * sqrt(eps_priv / 2))
    """
    return B / (n * lam * np.sqrt(eps_priv / 2))


# ============================================================
# Experiment 1: Robustness Erosion Curves
# ============================================================

def experiment1_erosion(X_train, y_train, X_test, y_test, dataset_name):
    """
    Delete points under three orderings:
    - adversarial: greedy, recompute influences every 25 deletions
    - random: average over N_RANDOM_TRIALS random orderings
    - improving: reverse of adversarial (most margin-increasing first)
    Measure adversarial accuracy after each batch.
    """
    print(f"\n  [Exp 1] Robustness Erosion Curves: {dataset_name}")
    eps = PGD_EPS[dataset_name]
    w_orig = train_logistic(X_train, y_train)
    H = compute_hessian(X_train, y_train, w_orig)
    H_inv = np.linalg.inv(H)

    # Baseline
    base_adv_acc = adversarial_accuracy(w_orig, X_test, y_test, epsilon=eps)
    base_clean_acc = clean_accuracy(w_orig, X_test, y_test)
    print(f"    Baseline: clean={base_clean_acc:.3f}, adv={base_adv_acc:.3f}")

    results = {"baseline_clean": base_clean_acc, "baseline_adv": base_adv_acc, "steps": []}

    # --- Compute initial influences ---
    print("    Computing robustness influences...")
    influences = robustness_influence(w_orig, X_train, y_train, X_test, y_test, H_inv)

    # Adversarial ordering: most negative (margin-reducing) first
    adv_order = np.argsort(influences)
    # Improving ordering: most positive (margin-increasing) first
    imp_order = np.argsort(-influences)

    max_k = max(DELETION_STEPS)

    for ordering_name in ["adversarial", "random", "improving"]:
        print(f"    Ordering: {ordering_name}")

        if ordering_name == "adversarial":
            order = adv_order
        elif ordering_name == "improving":
            order = imp_order
        else:
            order = None

        for k in DELETION_STEPS:
            if k > len(y_train) - 10:
                continue

            if ordering_name == "random":
                adv_accs = []
                clean_accs = []
                for trial in range(N_RANDOM_TRIALS):
                    rng = np.random.RandomState(SEED + trial)
                    rand_order = rng.permutation(len(y_train))[:k]
                    w_k = w_orig.copy()
                    for idx in rand_order:
                        delta = newton_step_delete(w_k, X_train, y_train, idx, H_inv)
                        w_k = w_k + delta
                    adv_accs.append(adversarial_accuracy(w_k, X_test, y_test, epsilon=eps))
                    clean_accs.append(clean_accuracy(w_k, X_test, y_test))
                adv_acc = np.mean(adv_accs)
                adv_std = np.std(adv_accs)
                clean_acc_val = np.mean(clean_accs)
            else:
                delete_idx = order[:k]
                w_k = w_orig.copy()
                for idx in delete_idx:
                    delta = newton_step_delete(w_k, X_train, y_train, idx, H_inv)
                    w_k = w_k + delta
                adv_acc = adversarial_accuracy(w_k, X_test, y_test, epsilon=eps)
                adv_std = 0.0
                clean_acc_val = clean_accuracy(w_k, X_test, y_test)

            results["steps"].append({
                "ordering": ordering_name,
                "k": k,
                "adv_accuracy": round(adv_acc, 4),
                "adv_std": round(adv_std, 4),
                "clean_accuracy": round(clean_acc_val, 4),
            })
            print(f"      k={k:3d}: adv_acc={adv_acc:.4f}, clean_acc={clean_acc_val:.4f}")

    return results


def plot_erosion_curves(all_results):
    """Plot Figure 1: Robustness erosion curves for all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {"adversarial": "#d62728", "random": "#2ca02c", "improving": "#1f77b4"}
    markers = {"adversarial": "v", "random": "o", "improving": "^"}

    for ax, (dname, res) in zip(axes, all_results.items()):
        for ordering in ["adversarial", "random", "improving"]:
            steps_data = [s for s in res["steps"] if s["ordering"] == ordering]
            ks = [0] + [s["k"] for s in steps_data]
            accs = [res["baseline_adv"]] + [s["adv_accuracy"] for s in steps_data]
            stds = [0] + [s["adv_std"] for s in steps_data]

            ax.plot(ks, accs, color=colors[ordering], marker=markers[ordering],
                    label=ordering.capitalize(), linewidth=2, markersize=6)
            if ordering == "random":
                accs_arr = np.array(accs)
                stds_arr = np.array(stds)
                ax.fill_between(ks, accs_arr - stds_arr, accs_arr + stds_arr,
                                color=colors[ordering], alpha=0.15)

        ax.set_xlabel("Number of Deletions (k)")
        ax.set_ylabel("Adversarial Accuracy")
        ax.set_title(dname)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_erosion_curves.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig1_erosion_curves.pdf")


# ============================================================
# Experiment 2: Margin Analysis
# ============================================================

def experiment2_margin(X_train, y_train, X_test, y_test, dataset_name):
    """
    Track minimum margin as adversarial deletions accumulate.
    Compare empirical trajectory with Theorem 1 bound.
    """
    print(f"\n  [Exp 2] Margin Analysis: {dataset_name}")
    w = train_logistic(X_train, y_train)
    H = compute_hessian(X_train, y_train, w)
    H_inv = np.linalg.inv(H)

    influences = robustness_influence(w, X_train, y_train, X_test, y_test, H_inv)
    adv_order = np.argsort(influences)  # most negative first (margin-reducing)

    n = X_train.shape[0]
    d = X_train.shape[1]
    w_norm = np.linalg.norm(w)

    # Gradient bound B: max ||grad_l||
    grads = compute_gradient(X_train, y_train, w)
    B = np.max(np.linalg.norm(grads, axis=1))

    # Theoretical per-deletion bound (Theorem 1)
    x_norms = np.linalg.norm(X_test, axis=1)
    wtx = np.abs(X_test @ w)
    max_grad_margin_norm = np.max(x_norms / w_norm + wtx / w_norm**2)
    per_deletion_bound = B / (LAMBDA * n) * max_grad_margin_norm

    # Empirical per-deletion bound: use the actual max |influence| observed
    empirical_per_deletion = np.max(np.abs(influences))

    base_min_margin = min_margin(w, X_test, y_test)

    results = {"base_min_margin": round(base_min_margin, 6), "steps": []}

    w_k = w.copy()
    fine_steps = sorted(set([1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]))
    fine_steps = [s for s in fine_steps if s <= max(DELETION_STEPS)]

    prev_k = 0
    for k in fine_steps:
        for idx in adv_order[prev_k:k]:
            delta = newton_step_delete(w_k, X_train, y_train, idx, H_inv)
            w_k = w_k + delta
        prev_k = k

        empirical_margin = min_margin(w_k, X_test, y_test)
        theoretical_lower = base_min_margin - k * per_deletion_bound
        empirical_bound = base_min_margin - k * empirical_per_deletion

        results["steps"].append({
            "k": k,
            "empirical_margin": round(empirical_margin, 6),
            "theoretical_lower": round(theoretical_lower, 6),
            "empirical_bound": round(empirical_bound, 6),
        })
        print(f"    k={k:3d}: margin={empirical_margin:.6f}, emp_bound={empirical_bound:.6f}, thm_bound={theoretical_lower:.6f}")

    results["per_deletion_bound"] = round(per_deletion_bound, 8)
    results["empirical_per_deletion"] = round(empirical_per_deletion, 8)
    return results


def plot_margin_analysis(all_results):
    """Plot Figure 2: Margin trajectory vs theoretical bounds (two panels per dataset)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, (dname, res) in enumerate(all_results.items()):
        ks = [0] + [s["k"] for s in res["steps"]]
        empirical = [res["base_min_margin"]] + [s["empirical_margin"] for s in res["steps"]]
        emp_bound = [res["base_min_margin"]] + [s["empirical_bound"] for s in res["steps"]]
        thm_bound = [res["base_min_margin"]] + [s["theoretical_lower"] for s in res["steps"]]

        # Top row: empirical margin + empirical bound (tight, readable)
        ax_top = axes[0, col]
        ax_top.plot(ks, empirical, "o-", color="#1f77b4", label="Empirical margin", linewidth=2, markersize=5)
        ax_top.plot(ks, emp_bound, "s--", color="#ff7f0e", label="Empirical influence bound", linewidth=2, markersize=5)
        ax_top.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax_top.set_ylabel("Minimum Margin")
        ax_top.set_title(dname)
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        # Bottom row: all three on same axes (shows looseness of Theorem 1)
        ax_bot = axes[1, col]
        ax_bot.plot(ks, empirical, "o-", color="#1f77b4", label="Empirical margin", linewidth=2, markersize=5)
        ax_bot.plot(ks, emp_bound, "s--", color="#ff7f0e", label="Empirical influence bound", linewidth=2, markersize=4)
        ax_bot.plot(ks, thm_bound, "v--", color="#d62728", label="Theorem 1 worst-case", linewidth=2, markersize=4)
        ax_bot.set_xlabel("Number of Adversarial Deletions (k)")
        ax_bot.set_ylabel("Minimum Margin")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_margin_analysis.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_margin_analysis.pdf")


# ============================================================
# Experiment 3: Noise-Robustness Tradeoff
# ============================================================

def experiment3_noise(X_train, y_train, X_test, y_test, dataset_name):
    """
    For different privacy epsilon values, add certified removal noise
    and measure adversarial + clean accuracy. Validates Theorem 3.
    """
    print(f"\n  [Exp 3] Noise-Robustness Tradeoff: {dataset_name}")
    pgd_eps = PGD_EPS[dataset_name]
    w = train_logistic(X_train, y_train)
    n, d = X_train.shape

    # Gradient bound
    grads = compute_gradient(X_train, y_train, w)
    B = np.max(np.linalg.norm(grads, axis=1))

    base_adv = adversarial_accuracy(w, X_test, y_test, epsilon=pgd_eps)
    base_clean = clean_accuracy(w, X_test, y_test)
    base_margin = min_margin(w, X_test, y_test)

    results = {"baseline_adv": base_adv, "baseline_clean": base_clean,
               "baseline_margin": round(base_margin, 6), "noise_results": []}

    for eps_priv in NOISE_EPSILONS:
        sigma = certified_removal_noise_sigma(LAMBDA, n, B, eps_priv)
        # Average over multiple noise draws
        adv_accs = []
        clean_accs = []
        margins = []
        for trial in range(N_RANDOM_TRIALS):
            rng = np.random.RandomState(SEED + trial + 1000)
            eta = rng.normal(0, sigma, size=d)
            w_noisy = w + eta
            adv_accs.append(adversarial_accuracy(w_noisy, X_test, y_test, epsilon=pgd_eps))
            clean_accs.append(clean_accuracy(w_noisy, X_test, y_test))
            margins.append(min_margin(w_noisy, X_test, y_test))

        # Theoretical margin reduction (Theorem 3)
        w_norm = np.linalg.norm(w)
        theoretical_ratio = w_norm / np.sqrt(w_norm**2 + sigma**2 * d)
        theoretical_margin = base_margin * theoretical_ratio

        result = {
            "eps_priv": eps_priv,
            "sigma": round(sigma, 6),
            "adv_accuracy": round(np.mean(adv_accs), 4),
            "adv_std": round(np.std(adv_accs), 4),
            "clean_accuracy": round(np.mean(clean_accs), 4),
            "clean_std": round(np.std(clean_accs), 4),
            "empirical_margin": round(np.mean(margins), 6),
            "theoretical_margin": round(theoretical_margin, 6),
        }
        results["noise_results"].append(result)
        print(f"    eps={eps_priv:.1f}: sigma={sigma:.4f}, adv={np.mean(adv_accs):.4f}, "
              f"clean={np.mean(clean_accs):.4f}, margin={np.mean(margins):.6f}")

    return results


def plot_noise_robustness(all_results):
    """Plot Figure 3: Noise-robustness tradeoff."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (dname, res) in zip(axes, all_results.items()):
        eps_vals = [r["eps_priv"] for r in res["noise_results"]]
        adv_accs = [r["adv_accuracy"] for r in res["noise_results"]]
        adv_stds = [r["adv_std"] for r in res["noise_results"]]
        clean_accs = [r["clean_accuracy"] for r in res["noise_results"]]

        ax.errorbar(eps_vals, adv_accs, yerr=adv_stds, fmt="o-", color="#d62728",
                    label="Adversarial Accuracy", linewidth=2, markersize=6, capsize=3)
        ax.plot(eps_vals, clean_accs, "s--", color="#1f77b4",
                label="Clean Accuracy", linewidth=2, markersize=6)

        # Add baseline
        ax.axhline(y=res["baseline_adv"], color="#d62728", linestyle=":", alpha=0.4, label="Baseline (adv)")
        ax.axhline(y=res["baseline_clean"], color="#1f77b4", linestyle=":", alpha=0.4, label="Baseline (clean)")

        ax.set_xlabel("Privacy Parameter (eps)")
        ax.set_ylabel("Accuracy")
        ax.set_title(dname)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_noise_robustness.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig3_noise_robustness.pdf")


# ============================================================
# Experiment 4: Targeted Attack After Unlearning
# ============================================================

def experiment4_targeted(X_train, y_train, X_test, y_test, dataset_name):
    """
    Identify near-boundary test points. Delete 50 training points adversarially
    (targeting margin reduction). Compare attack success rate before and after.
    """
    print(f"\n  [Exp 4] Targeted Attack After Unlearning: {dataset_name}")
    pgd_eps = PGD_EPS[dataset_name]
    k_delete = 100

    w = train_logistic(X_train, y_train)
    H = compute_hessian(X_train, y_train, w)
    H_inv = np.linalg.inv(H)

    # Identify near-boundary test points (smallest signed margin, i.e. most vulnerable)
    margins = geometric_margin(w, X_test, y_test)
    n_target = min(200, len(y_test) // 5)
    target_idx = np.argsort(margins)[:n_target]  # smallest margin (most vulnerable)
    X_target = X_test[target_idx]
    y_target = y_test[target_idx]

    # Attack success rate BEFORE unlearning
    success_before = 0
    for i in range(len(y_target)):
        x_adv = pgd_attack(w, X_target[i], y_target[i], epsilon=pgd_eps)
        pred = np.sign(w @ x_adv)
        if pred != y_target[i]:
            success_before += 1
    asr_before = success_before / len(y_target)

    # Adversarial deletions: most margin-reducing (most negative influence)
    influences = robustness_influence(w, X_train, y_train, X_test, y_test, H_inv)
    adv_order = np.argsort(influences)[:k_delete]  # most negative first

    w_unlearned = w.copy()
    for idx in adv_order:
        delta = newton_step_delete(w_unlearned, X_train, y_train, idx, H_inv)
        w_unlearned = w_unlearned + delta

    # Attack success rate AFTER unlearning
    success_after = 0
    for i in range(len(y_target)):
        x_adv = pgd_attack(w_unlearned, X_target[i], y_target[i], epsilon=pgd_eps)
        pred = np.sign(w_unlearned @ x_adv)
        if pred != y_target[i]:
            success_after += 1
    asr_after = success_after / len(y_target)

    # Also check: FGSM (one-step attack) for comparison
    fgsm_before = 0
    fgsm_after = 0
    for i in range(len(y_target)):
        # FGSM on original model
        grad_x = -(1 - sigmoid(y_target[i] * (w @ X_target[i]))) * y_target[i] * w
        x_fgsm = X_target[i] + pgd_eps * np.sign(grad_x)
        x_fgsm = np.clip(x_fgsm, 0, 1)
        if np.sign(w @ x_fgsm) != y_target[i]:
            fgsm_before += 1

        # FGSM on unlearned model
        grad_x = -(1 - sigmoid(y_target[i] * (w_unlearned @ X_target[i]))) * y_target[i] * w_unlearned
        x_fgsm = X_target[i] + pgd_eps * np.sign(grad_x)
        x_fgsm = np.clip(x_fgsm, 0, 1)
        if np.sign(w_unlearned @ x_fgsm) != y_target[i]:
            fgsm_after += 1

    fgsm_asr_before = fgsm_before / len(y_target)
    fgsm_asr_after = fgsm_after / len(y_target)

    # Margin statistics (signed, not absolute)
    margin_before = np.mean(geometric_margin(w, X_target, y_target))
    margin_after = np.mean(geometric_margin(w_unlearned, X_target, y_target))
    min_margin_before = np.min(geometric_margin(w, X_target, y_target))
    min_margin_after = np.min(geometric_margin(w_unlearned, X_target, y_target))

    results = {
        "k_deletions": k_delete,
        "n_target_points": len(y_target),
        "pgd_asr_before": round(asr_before, 4),
        "pgd_asr_after": round(asr_after, 4),
        "pgd_asr_increase": round(asr_after - asr_before, 4),
        "fgsm_asr_before": round(fgsm_asr_before, 4),
        "fgsm_asr_after": round(fgsm_asr_after, 4),
        "fgsm_asr_increase": round(fgsm_asr_after - fgsm_asr_before, 4),
        "mean_margin_before": round(margin_before, 6),
        "mean_margin_after": round(margin_after, 6),
        "min_margin_before": round(min_margin_before, 6),
        "min_margin_after": round(min_margin_after, 6),
    }

    print(f"    PGD ASR: {asr_before:.4f} -> {asr_after:.4f} (+{asr_after - asr_before:.4f})")
    print(f"    FGSM ASR: {fgsm_asr_before:.4f} -> {fgsm_asr_after:.4f} (+{fgsm_asr_after - fgsm_asr_before:.4f})")
    print(f"    Mean margin: {margin_before:.6f} -> {margin_after:.6f}")
    print(f"    Min margin:  {min_margin_before:.6f} -> {min_margin_after:.6f}")

    return results


def plot_targeted_attack(all_results):
    """Plot Figure 4: Targeted attack success rates before/after unlearning."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    datasets = list(all_results.keys())
    x = np.arange(len(datasets))
    width = 0.18

    pgd_before = [all_results[d]["pgd_asr_before"] for d in datasets]
    pgd_after = [all_results[d]["pgd_asr_after"] for d in datasets]
    fgsm_before = [all_results[d]["fgsm_asr_before"] for d in datasets]
    fgsm_after = [all_results[d]["fgsm_asr_after"] for d in datasets]

    ax.bar(x - 1.5*width, pgd_before, width, label="PGD Before", color="#1f77b4", alpha=0.7)
    ax.bar(x - 0.5*width, pgd_after, width, label="PGD After", color="#d62728", alpha=0.7)
    ax.bar(x + 0.5*width, fgsm_before, width, label="FGSM Before", color="#1f77b4", alpha=0.4, hatch="//")
    ax.bar(x + 1.5*width, fgsm_after, width, label="FGSM After", color="#d62728", alpha=0.4, hatch="//")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack Success Rate Before/After Adversarial Unlearning (k=100)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_targeted_attack.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig4_targeted_attack.pdf")


# ============================================================
# Summary Tables
# ============================================================

def save_summary_tables(exp1_all, exp2_all, exp3_all, exp4_all):
    """Save CSV summary tables."""
    # Table 1: Erosion summary (k=100)
    with open(OUTPUT_DIR / "table1_erosion_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Ordering", "k", "Adv Accuracy", "Clean Accuracy"])
        for dname, res in exp1_all.items():
            for step in res["steps"]:
                if step["k"] == 100:
                    writer.writerow([dname, step["ordering"], step["k"],
                                     step["adv_accuracy"], step["clean_accuracy"]])

    # Table 2: Noise tradeoff summary
    with open(OUTPUT_DIR / "table2_noise_tradeoff.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "eps_priv", "sigma", "Adv Accuracy", "Clean Accuracy",
                          "Empirical Margin", "Theoretical Margin"])
        for dname, res in exp3_all.items():
            for nr in res["noise_results"]:
                writer.writerow([dname, nr["eps_priv"], nr["sigma"],
                                 nr["adv_accuracy"], nr["clean_accuracy"],
                                 nr["empirical_margin"], nr["theoretical_margin"]])

    # Table 3: Targeted attack summary
    with open(OUTPUT_DIR / "table3_targeted_attack.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "PGD ASR Before", "PGD ASR After", "PGD Increase",
                          "FGSM ASR Before", "FGSM ASR After", "FGSM Increase",
                          "Min Margin Before", "Min Margin After"])
        for dname, res in exp4_all.items():
            writer.writerow([dname, res["pgd_asr_before"], res["pgd_asr_after"],
                             res["pgd_asr_increase"], res["fgsm_asr_before"],
                             res["fgsm_asr_after"], res["fgsm_asr_increase"],
                             res.get("min_margin_before", ""), res.get("min_margin_after", "")])

    print("\n  Saved summary CSV tables.")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Unlearning Under Siege (UUS) Experiments")
    print("=" * 60)

    # --- File discovery diagnostics ---
    print("\nSearching for datasets...")
    all_csvs = sorted(_glob.glob("/kaggle/input/**/*.csv", recursive=True))
    all_npzs = sorted(_glob.glob("/kaggle/input/**/*.npz", recursive=True))
    print(f"  Found {len(all_csvs)} CSV files, {len(all_npzs)} NPZ files under /kaggle/input/")
    for f in all_csvs[:15]:
        print(f"    {f}")
    if len(all_csvs) > 15:
        print(f"    ... and {len(all_csvs) - 15} more")
    for f in all_npzs:
        print(f"    {f}")
    print()

    exp1_all = {}
    exp2_all = {}
    exp3_all = {}
    exp4_all = {}

    for dname, loader in DATASET_LOADERS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dname}")
        print(f"{'='*60}")

        X, y = loader()
        print(f"  Loaded: n={len(y)}, d={X.shape[1]}, "
              f"pos={np.sum(y==1)}, neg={np.sum(y==-1)}")

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

        exp1_all[dname] = experiment1_erosion(X_train, y_train, X_test, y_test, dname)
        exp2_all[dname] = experiment2_margin(X_train, y_train, X_test, y_test, dname)
        exp3_all[dname] = experiment3_noise(X_train, y_train, X_test, y_test, dname)
        exp4_all[dname] = experiment4_targeted(X_train, y_train, X_test, y_test, dname)

    # Generate figures
    print(f"\n{'='*60}")
    print("Generating Figures")
    print(f"{'='*60}")
    plot_erosion_curves(exp1_all)
    plot_margin_analysis(exp2_all)
    plot_noise_robustness(exp3_all)
    plot_targeted_attack(exp4_all)

    # Save summary tables
    save_summary_tables(exp1_all, exp2_all, exp3_all, exp4_all)

    # Save full results as JSON
    all_results = {
        "experiment1_erosion": {k: v for k, v in exp1_all.items()},
        "experiment2_margin": {k: v for k, v in exp2_all.items()},
        "experiment3_noise": {k: v for k, v in exp3_all.items()},
        "experiment4_targeted": {k: v for k, v in exp4_all.items()},
    }
    with open(OUTPUT_DIR / "UUS_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("  Saved UUS_all_results.json")

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results in: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

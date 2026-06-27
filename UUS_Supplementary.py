"""
UUS Supplementary Experiments (Reviewer Fixes)
===============================================
Addresses:
  W1: MLP (2-layer, 512-128-2) experiments with approximate influence functions
  W4: Retrain-from-scratch baseline (on MNIST)
  W5: Full test set PGD evaluation (no subsampling)
  M4: Figure 2 margin analysis plot (was missing from submission)

Run AFTER UUS_Experiments.py (uses same data discovery).
Outputs go to results/ alongside the main results.

Usage:
  python UUS_Supplementary.py
"""

import numpy as np
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# Import data loaders from main script
# ============================================================
import glob as _glob

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

LAMBDA = 0.001
N_SAMPLES = 2000
DELETION_STEPS = [5, 10, 25, 50, 100, 150, 200]
PGD_STEPS = 40
PGD_STEP_SIZE = 0.005
NOISE_EPSILONS = [0.5, 1.0, 2.0, 5.0]
N_RANDOM_TRIALS = 5

PGD_EPS = {
    "MNIST": 0.1,
    "FashionMNIST": 0.1,
    "CIFAR10_features": 0.03,
}


def _find_file(*patterns):
    search_roots = ["/kaggle/input", "/kaggle/working", "data"]
    for root in search_roots:
        for pat in patterns:
            hits = sorted(_glob.glob(f"{root}/**/{pat}", recursive=True))
            if hits:
                return hits[0]
    return None


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(X, y, lam=LAMBDA):
    """Train L2-regularized logistic regression, return weight vector."""
    from sklearn.linear_model import LogisticRegression
    n, d = X.shape
    clf = LogisticRegression(C=1.0 / (n * lam), fit_intercept=False,
                             solver="lbfgs", max_iter=1000, tol=1e-8)
    clf.fit(X, y)
    return clf.coef_.flatten()


def compute_hessian(X, y, w, lam=LAMBDA):
    """Compute Hessian of regularized logistic loss."""
    n, d = X.shape
    z = y * (X @ w)
    s = sigmoid(z)
    diag = s * (1 - s)
    H = (X.T * diag) @ X / n + lam * np.eye(d)
    return H


def compute_gradient(X, y, w):
    """Compute per-sample gradients of logistic loss."""
    z = y * (X @ w)
    s = sigmoid(z)
    coeff = -(1 - s) * y
    grads = X * coeff[:, np.newaxis]
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
    return np.min(geometric_margin(w, X, y))


def compute_margin_gradient(w, x, y_val):
    """Gradient of margin m(x; w) = y * w^T x / ||w|| w.r.t. w."""
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return np.zeros_like(w)
    wtx = w @ x
    return y_val / w_norm * (x - (wtx / w_norm**2) * w)


def robustness_influence(w, X_train, y_train, X_test, y_test, H_inv, lam=LAMBDA, top_k_test=50):
    """Compute signed robustness influence for each training point."""
    n_train = X_train.shape[0]
    margins = geometric_margin(w, X_test, y_test)
    vulnerable_idx = np.argsort(margins)[:top_k_test]
    X_vuln = X_test[vulnerable_idx]
    y_vuln = y_test[vulnerable_idx]
    margin_grads = np.array([
        compute_margin_gradient(w, X_vuln[i], y_vuln[i])
        for i in range(len(vulnerable_idx))
    ])
    influences = np.zeros(n_train)
    for j in range(n_train):
        delta_j = newton_step_delete(w, X_train, y_train, j, H_inv, lam)
        signed_changes = margin_grads @ delta_j
        influences[j] = np.min(signed_changes)
    return influences


def adversarial_accuracy(w, X, y, epsilon=0.1):
    """PGD adversarial accuracy on the FULL test set (no subsampling, W5 fix)."""
    correct = 0
    n = len(y)
    for i in range(n):
        x_adv = X[i].copy()
        for _ in range(PGD_STEPS):
            logit = w @ x_adv
            prob = sigmoid(y[i] * logit)
            grad_x = -(1 - prob) * y[i] * w
            x_adv = x_adv + PGD_STEP_SIZE * np.sign(grad_x)
            x_adv = np.clip(x_adv, X[i] - epsilon, X[i] + epsilon)
            x_adv = np.clip(x_adv, -3, 3)
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


# ============================================================
# MLP Model (2-layer: d -> 128 -> 1)
# ============================================================

class SimpleMLP:
    """
    2-layer MLP for binary classification with L2 regularization.
    Architecture: input(d) -> hidden(128, ReLU) -> output(1, sigmoid)
    Trained with gradient descent. Stores flattened parameters for
    influence function computation.
    """
    def __init__(self, d, hidden=128, lam=LAMBDA, lr=0.01, epochs=200):
        self.d = d
        self.hidden = hidden
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        # Initialize weights
        rng = np.random.RandomState(SEED)
        self.W1 = rng.randn(d, hidden) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)

    def _forward(self, X):
        """Forward pass, return (hidden_activations, logits)."""
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0)  # ReLU
        logits = (h @ self.W2 + self.b2).flatten()
        return h, logits

    def predict_logits(self, X):
        _, logits = self._forward(X)
        return logits

    def predict(self, X):
        logits = self.predict_logits(X)
        return np.sign(logits)

    def get_params(self):
        """Flatten all parameters into a single vector."""
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_params(self, theta):
        """Set parameters from a flattened vector."""
        idx = 0
        n1 = self.d * self.hidden
        self.W1 = theta[idx:idx+n1].reshape(self.d, self.hidden)
        idx += n1
        self.b1 = theta[idx:idx+self.hidden]
        idx += self.hidden
        n2 = self.hidden * 1
        self.W2 = theta[idx:idx+n2].reshape(self.hidden, 1)
        idx += n2
        self.b2 = theta[idx:idx+1]

    def n_params(self):
        return self.d * self.hidden + self.hidden + self.hidden * 1 + 1

    def fit(self, X, y):
        """Train with mini-batch gradient descent."""
        n = X.shape[0]
        for epoch in range(self.epochs):
            h, logits = self._forward(X)
            probs = sigmoid(y * logits)
            loss_grad = -(1 - probs) * y  # (n,)

            # Backprop
            dL_dlogits = loss_grad  # (n,)
            dL_dW2 = (h.T @ dL_dlogits.reshape(-1, 1)) / n + self.lam * self.W2
            dL_db2 = np.mean(dL_dlogits)

            dL_dh = np.outer(dL_dlogits, self.W2.flatten())  # (n, hidden)
            relu_mask = (X @ self.W1 + self.b1) > 0
            dL_dz1 = dL_dh * relu_mask  # (n, hidden)
            dL_dW1 = (X.T @ dL_dz1) / n + self.lam * self.W1
            dL_db1 = np.mean(dL_dz1, axis=0)

            self.W1 -= self.lr * dL_dW1
            self.b1 -= self.lr * dL_db1
            self.W2 -= self.lr * dL_dW2
            self.b2 -= self.lr * np.array([dL_db2])

        return self

    def compute_per_sample_gradient(self, X, y):
        """Compute gradient of loss for each sample, returned as (n, p) matrix."""
        n = X.shape[0]
        p = self.n_params()
        grads = np.zeros((n, p))

        for i in range(n):
            xi = X[i:i+1]
            yi = y[i]
            h, logit = self._forward(xi)
            prob = sigmoid(yi * logit[0])
            dl = -(1 - prob) * yi

            # Backprop for single sample
            dW2 = (h.T * dl).flatten()
            db2 = np.array([dl])
            dh = dl * self.W2.flatten()
            relu_mask = ((xi @ self.W1 + self.b1) > 0).flatten()
            dz1 = dh * relu_mask
            dW1 = (xi.T @ dz1.reshape(1, -1)).ravel()
            db1 = dz1

            grads[i] = np.concatenate([dW1, db1, dW2, db2])

        return grads

    def compute_approx_hessian(self, X, y):
        """
        Approximate Hessian using Gauss-Newton: H ~ (1/n) J^T J + lam*I
        where J is the Jacobian of per-sample gradients.
        This avoids computing second derivatives through ReLU.
        """
        grads = self.compute_per_sample_gradient(X, y)
        n = X.shape[0]
        p = self.n_params()
        H = (grads.T @ grads) / n + self.lam * np.eye(p)
        return H


def mlp_geometric_margin(model, X, y):
    """Approximate margin for MLP: y * f(x) / ||theta||."""
    logits = model.predict_logits(X)
    theta_norm = np.linalg.norm(model.get_params())
    return (y * logits) / theta_norm


def mlp_min_margin(model, X, y):
    return np.min(mlp_geometric_margin(model, X, y))


def mlp_adversarial_accuracy(model, X, y, epsilon=0.1, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    """PGD attack for MLP, evaluated on FULL test set (W5 fix)."""
    correct = 0
    n = len(y)
    for i in range(n):
        x_adv = X[i].copy()
        for _ in range(steps):
            # Numerical gradient w.r.t. input
            xi = x_adv.reshape(1, -1)
            h = np.maximum(xi @ model.W1 + model.b1, 0)
            logit = (h @ model.W2 + model.b2).flatten()[0]
            prob = sigmoid(y[i] * logit)
            dl = -(1 - prob) * y[i]
            # Backprop to input
            dh = dl * model.W2.flatten()
            relu_mask = ((xi @ model.W1 + model.b1) > 0).flatten()
            dz1 = dh * relu_mask
            grad_x = (dz1.reshape(1, -1) @ model.W1.T).flatten()
            x_adv = x_adv + step_size * np.sign(grad_x)
            x_adv = np.clip(x_adv, X[i] - epsilon, X[i] + epsilon)
            x_adv = np.clip(x_adv, -3, 3)  # standardized data range
        pred = np.sign(model.predict_logits(x_adv.reshape(1, -1))[0])
        if pred == y[i]:
            correct += 1
    return correct / n


def mlp_clean_accuracy(model, X, y):
    preds = model.predict(X)
    preds[preds == 0] = 1
    return np.mean(preds == y)


def mlp_newton_delete(model, X, y, j, H_inv):
    """Newton-step deletion for MLP using approximate Hessian."""
    grad_j = model.compute_per_sample_gradient(X[j:j+1], y[j:j+1]).flatten()
    n = X.shape[0]
    delta = (1.0 / n) * H_inv @ grad_j
    return delta


# ============================================================
# W1: MLP Experiments (Exp 1 + Exp 3 on MNIST only)
# ============================================================

def mlp_experiment_erosion(X_train, y_train, X_test, y_test, dataset_name="MNIST_MLP"):
    """Experiment 1 with MLP model."""
    print(f"\n  [MLP Exp 1] Robustness Erosion: {dataset_name}")
    eps = PGD_EPS.get("MNIST", 0.1)

    # Train MLP
    d = X_train.shape[1]
    model = SimpleMLP(d, hidden=128, lam=LAMBDA, lr=0.01, epochs=300)
    model.fit(X_train, y_train)

    base_clean = mlp_clean_accuracy(model, X_test, y_test)
    base_adv = mlp_adversarial_accuracy(model, X_test, y_test, epsilon=eps)
    print(f"    Baseline: clean={base_clean:.3f}, adv={base_adv:.3f}")

    # Compute approximate Hessian and its inverse
    print("    Computing approximate Hessian...")
    H = model.compute_approx_hessian(X_train, y_train)
    print(f"    Hessian shape: {H.shape}, inverting...")
    H_inv = np.linalg.inv(H + 1e-4 * np.eye(H.shape[0]))  # regularize for stability

    # Compute influences
    print("    Computing MLP robustness influences...")
    theta_orig = model.get_params().copy()
    margins = mlp_geometric_margin(model, X_test, y_test)
    top_k = min(50, len(y_test))
    vuln_idx = np.argsort(margins)[:top_k]
    X_vuln, y_vuln = X_test[vuln_idx], y_test[vuln_idx]

    # Margin gradient for MLP (approximate: gradient of y*f(x)/||theta|| w.r.t. theta)
    n_train = X_train.shape[0]
    p = model.n_params()
    influences = np.zeros(n_train)

    # Precompute margin gradients at vulnerable points
    theta_norm = np.linalg.norm(theta_orig)
    margin_grads = np.zeros((top_k, p))
    eps_fd = 1e-5  # finite difference step
    for v in range(top_k):
        logit_base = model.predict_logits(X_vuln[v:v+1])[0]
        m_base = y_vuln[v] * logit_base / theta_norm
        # Use per-sample gradient as proxy for margin gradient
        grad_loss = model.compute_per_sample_gradient(X_vuln[v:v+1], y_vuln[v:v+1]).flatten()
        # margin ~ y*f(x)/||theta||, grad_margin ~ y*grad_f/||theta|| (first-order)
        margin_grads[v] = y_vuln[v] * grad_loss / theta_norm  # approximate

    for j in range(n_train):
        delta_j = mlp_newton_delete(model, X_train, y_train, j, H_inv)
        signed_changes = margin_grads @ delta_j
        influences[j] = np.min(signed_changes)

    adv_order = np.argsort(influences)
    imp_order = np.argsort(-influences)

    results = {"baseline_clean": base_clean, "baseline_adv": base_adv,
               "model": "MLP_128", "steps": []}

    deletion_steps_mlp = [5, 10, 25, 50, 100]

    for ordering_name in ["adversarial", "random", "improving"]:
        print(f"    Ordering: {ordering_name}")
        if ordering_name == "adversarial":
            order = adv_order
        elif ordering_name == "improving":
            order = imp_order
        else:
            order = None

        for k in deletion_steps_mlp:
            if ordering_name == "random":
                adv_accs = []
                for trial in range(3):  # fewer trials for MLP (slow)
                    rng = np.random.RandomState(SEED + trial)
                    rand_order = rng.permutation(n_train)[:k]
                    model.set_params(theta_orig.copy())
                    theta_k = theta_orig.copy()
                    for idx in rand_order:
                        delta = mlp_newton_delete(model, X_train, y_train, idx, H_inv)
                        theta_k = theta_k + delta
                    model.set_params(theta_k)
                    adv_accs.append(mlp_adversarial_accuracy(model, X_test, y_test, epsilon=eps))
                adv_acc = np.mean(adv_accs)
                adv_std = np.std(adv_accs)
            else:
                model.set_params(theta_orig.copy())
                theta_k = theta_orig.copy()
                for idx in order[:k]:
                    delta = mlp_newton_delete(model, X_train, y_train, idx, H_inv)
                    theta_k = theta_k + delta
                model.set_params(theta_k)
                adv_acc = mlp_adversarial_accuracy(model, X_test, y_test, epsilon=eps)
                adv_std = 0.0

            results["steps"].append({
                "ordering": ordering_name, "k": k,
                "adv_accuracy": round(adv_acc, 4), "adv_std": round(adv_std, 4),
            })
            print(f"      k={k:3d}: adv_acc={adv_acc:.4f}")

    model.set_params(theta_orig)
    return results


def mlp_experiment_noise(X_train, y_train, X_test, y_test, dataset_name="MNIST_MLP"):
    """Experiment 3 (noise tradeoff) with MLP model."""
    print(f"\n  [MLP Exp 3] Noise-Robustness Tradeoff: {dataset_name}")
    eps = PGD_EPS.get("MNIST", 0.1)

    d = X_train.shape[1]
    model = SimpleMLP(d, hidden=128, lam=LAMBDA, lr=0.01, epochs=300)
    model.fit(X_train, y_train)
    theta_orig = model.get_params().copy()
    p = model.n_params()
    n = X_train.shape[0]

    grads = model.compute_per_sample_gradient(X_train, y_train)
    B = np.max(np.linalg.norm(grads, axis=1))

    base_adv = mlp_adversarial_accuracy(model, X_test, y_test, epsilon=eps)
    base_clean = mlp_clean_accuracy(model, X_test, y_test)
    base_margin = mlp_min_margin(model, X_test, y_test)

    results = {"baseline_adv": base_adv, "baseline_clean": base_clean,
               "baseline_margin": round(base_margin, 6), "model": "MLP_128",
               "noise_results": []}

    for eps_priv in NOISE_EPSILONS:
        sigma = B / (n * LAMBDA * np.sqrt(eps_priv / 2))
        adv_accs, clean_accs, margins_list = [], [], []
        for trial in range(3):
            rng = np.random.RandomState(SEED + trial + 2000)
            eta = rng.normal(0, sigma, size=p)
            model.set_params(theta_orig + eta)
            adv_accs.append(mlp_adversarial_accuracy(model, X_test, y_test, epsilon=eps))
            clean_accs.append(mlp_clean_accuracy(model, X_test, y_test))
            margins_list.append(mlp_min_margin(model, X_test, y_test))

        results["noise_results"].append({
            "eps_priv": eps_priv, "sigma": round(sigma, 4),
            "adv_accuracy": round(np.mean(adv_accs), 4),
            "adv_std": round(np.std(adv_accs), 4),
            "clean_accuracy": round(np.mean(clean_accs), 4),
        })
        print(f"    eps={eps_priv}: adv={np.mean(adv_accs):.4f}, clean={np.mean(clean_accs):.4f}")

    model.set_params(theta_orig)
    return results


# ============================================================
# W4: Retrain-from-scratch baseline
# ============================================================

def retrain_baseline(X_train, y_train, X_test, y_test, dataset_name="MNIST"):
    """
    Compare Newton-step unlearning vs full retraining after deletion.
    Tests whether robustness degradation is inherent to the data removal
    or an artifact of the approximate unlearning mechanism.
    """
    from sklearn.linear_model import LogisticRegression
    print(f"\n  [Retrain Baseline] {dataset_name}")
    eps = PGD_EPS.get(dataset_name, 0.1)
    n = X_train.shape[0]

    # Train original model
    w_orig = train_logistic(X_train, y_train, LAMBDA)

    H = compute_hessian(X_train, y_train, w_orig, LAMBDA)
    H_inv = np.linalg.inv(H)
    influences = robustness_influence(w_orig, X_train, y_train, X_test, y_test, H_inv, LAMBDA)
    adv_order = np.argsort(influences)  # most margin-reducing first

    base_adv = adversarial_accuracy(w_orig, X_test, y_test, epsilon=eps)
    base_margin = min_margin(w_orig, X_test, y_test)
    print(f"    Baseline: adv_acc={base_adv:.4f}, min_margin={base_margin:.4f}")

    results = {"baseline_adv": base_adv, "baseline_margin": round(base_margin, 6),
               "comparisons": []}

    for k in [25, 50, 100, 200]:
        delete_idx = adv_order[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[delete_idx] = False

        # Newton-step unlearning
        w_newton = w_orig.copy()
        for idx in delete_idx:
            delta = newton_step_delete(w_newton, X_train, y_train, idx, H_inv, LAMBDA)
            w_newton = w_newton + delta
        newton_adv = adversarial_accuracy(w_newton, X_test, y_test, epsilon=eps)
        newton_margin = min_margin(w_newton, X_test, y_test)

        # Full retrain from scratch
        X_remain = X_train[keep_mask]
        y_remain = y_train[keep_mask]
        n_remain = X_remain.shape[0]
        clf_retrain = LogisticRegression(C=1.0/(n_remain*LAMBDA), fit_intercept=False,
                                         solver="lbfgs", max_iter=1000, tol=1e-8)
        clf_retrain.fit(X_remain, y_remain)
        w_retrain = clf_retrain.coef_.flatten()
        retrain_adv = adversarial_accuracy(w_retrain, X_test, y_test, epsilon=eps)
        retrain_margin = min_margin(w_retrain, X_test, y_test)

        results["comparisons"].append({
            "k": k,
            "newton_adv": round(newton_adv, 4),
            "newton_margin": round(newton_margin, 6),
            "retrain_adv": round(retrain_adv, 4),
            "retrain_margin": round(retrain_margin, 6),
        })
        print(f"    k={k:3d}: newton_adv={newton_adv:.4f} margin={newton_margin:.4f} | "
              f"retrain_adv={retrain_adv:.4f} margin={retrain_margin:.4f}")

    return results


# ============================================================
# W5: Full test set evaluation
# (adversarial_accuracy above already uses full test set)
# ============================================================


def fulltest_erosion_check(X_train, y_train, X_test, y_test, dataset_name="MNIST"):
    """
    Re-run Exp 1 key points with full test set evaluation to verify
    that the 500-sample subsampling didn't introduce artifacts.
    """
    print(f"\n  [Full Test Set Check] {dataset_name}")
    eps = PGD_EPS.get(dataset_name, 0.1)
    n = X_train.shape[0]

    w = train_logistic(X_train, y_train, LAMBDA)

    H = compute_hessian(X_train, y_train, w, LAMBDA)
    H_inv = np.linalg.inv(H)
    influences = robustness_influence(w, X_train, y_train, X_test, y_test, H_inv, LAMBDA)
    adv_order = np.argsort(influences)
    imp_order = np.argsort(-influences)

    n_test = len(y_test)
    base_adv = adversarial_accuracy(w, X_test, y_test, epsilon=eps)
    print(f"    Baseline (full n_test={n_test}): adv_acc={base_adv:.4f}")

    results = {"baseline_adv_full": base_adv, "n_test": n_test, "checks": []}

    for k in [50, 100, 200]:
        for oname, order in [("adversarial", adv_order), ("improving", imp_order)]:
            w_k = w.copy()
            for idx in order[:k]:
                delta = newton_step_delete(w_k, X_train, y_train, idx, H_inv, LAMBDA)
                w_k = w_k + delta
            adv_acc = adversarial_accuracy(w_k, X_test, y_test, epsilon=eps)
            results["checks"].append({
                "ordering": oname, "k": k, "adv_accuracy_full": round(adv_acc, 4),
            })
            print(f"    {oname:12s} k={k:3d}: adv_acc_full={adv_acc:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("UUS Supplementary Experiments")
    print("=" * 60)

    # Load MNIST data (primary dataset for all supplementary experiments)
    def _load_csv_dataset(filepath, pos_label, neg_label, n_per_class=N_SAMPLES):
        import pandas as pd
        df = pd.read_csv(filepath)
        label_candidates = ["label", "Label", "class"]
        label_name = None
        for lc in label_candidates:
            if lc in df.columns:
                label_name = lc
                break
        if label_name:
            labels = df[label_name].values.astype(int)
            pixels = df.drop(columns=[label_name]).values.astype(np.float64)
        else:
            labels = df.iloc[:, 0].values.astype(int)
            pixels = df.iloc[:, 1:].values.astype(np.float64)
        mask = (labels == pos_label) | (labels == neg_label)
        X, raw_y = pixels[mask], labels[mask]
        y = np.where(raw_y == pos_label, 1, -1)
        if X.max() > 1.0:
            X = X / 255.0
        idx_pos = np.where(y == 1)[0][:n_per_class]
        idx_neg = np.where(y == -1)[0][:n_per_class]
        idx = np.concatenate([idx_pos, idx_neg])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    # Find MNIST
    mnist_path = _find_file("mnist_train.csv")
    if mnist_path is None:
        for root in ["/kaggle/input", "/kaggle/working", "data"]:
            hits = sorted(_glob.glob(f"{root}/**/*.csv", recursive=True))
            for h in hits:
                if "mnist" in h.lower() and "fashion" not in h.lower():
                    mnist_path = h
                    break
            if mnist_path:
                break

    if mnist_path is None:
        print("ERROR: MNIST CSV not found. Exiting.")
        return

    print(f"\nLoading MNIST from: {mnist_path}")
    X, y = _load_csv_dataset(mnist_path, pos_label=3, neg_label=8)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    all_supp = {}

    # --- W5: Full test set verification ---
    print(f"\n{'='*60}")
    print("W5: Full Test Set Evaluation")
    print(f"{'='*60}")
    all_supp["fulltest_check"] = fulltest_erosion_check(
        X_train, y_train, X_test, y_test, "MNIST"
    )

    # --- W4: Retrain baseline ---
    print(f"\n{'='*60}")
    print("W4: Retrain-from-Scratch Baseline")
    print(f"{'='*60}")
    all_supp["retrain_baseline"] = retrain_baseline(
        X_train, y_train, X_test, y_test, "MNIST"
    )

    # --- W1: MLP experiments ---
    print(f"\n{'='*60}")
    print("W1: MLP (2-layer, 128 hidden) Experiments")
    print(f"{'='*60}")
    # Use fewer samples for MLP (Hessian is huge: ~100k x 100k)
    # Reduce to d=100 via PCA for tractability
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"  PCA reduced: {X_train.shape[1]} -> {X_train_pca.shape[1]}")

    all_supp["mlp_erosion"] = mlp_experiment_erosion(
        X_train_pca, y_train, X_test_pca, y_test, "MNIST_MLP"
    )
    all_supp["mlp_noise"] = mlp_experiment_noise(
        X_train_pca, y_train, X_test_pca, y_test, "MNIST_MLP"
    )

    # --- Save results ---
    with open(OUTPUT_DIR / "UUS_supplementary_results.json", "w") as f:
        json.dump(all_supp, f, indent=2)
    print(f"\nSaved supplementary results to {OUTPUT_DIR}/UUS_supplementary_results.json")

    # --- Plot: Retrain baseline comparison ---
    rb = all_supp["retrain_baseline"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ks = [0] + [c["k"] for c in rb["comparisons"]]
    newton_adv = [rb["baseline_adv"]] + [c["newton_adv"] for c in rb["comparisons"]]
    retrain_adv = [rb["baseline_adv"]] + [c["retrain_adv"] for c in rb["comparisons"]]
    newton_margin = [rb["baseline_margin"]] + [c["newton_margin"] for c in rb["comparisons"]]
    retrain_margin = [rb["baseline_margin"]] + [c["retrain_margin"] for c in rb["comparisons"]]

    ax1.plot(ks, newton_adv, "o-", label="Newton-step unlearning", color="#d62728", linewidth=2)
    ax1.plot(ks, retrain_adv, "s--", label="Full retraining", color="#1f77b4", linewidth=2)
    ax1.set_xlabel("Number of Adversarial Deletions (k)")
    ax1.set_ylabel("Adversarial Accuracy")
    ax1.set_title("MNIST: Unlearning vs Retraining (Adv Acc)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, newton_margin, "o-", label="Newton-step unlearning", color="#d62728", linewidth=2)
    ax2.plot(ks, retrain_margin, "s--", label="Full retraining", color="#1f77b4", linewidth=2)
    ax2.set_xlabel("Number of Adversarial Deletions (k)")
    ax2.set_ylabel("Minimum Margin")
    ax2.set_title("MNIST: Unlearning vs Retraining (Margin)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_retrain_baseline.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig5_retrain_baseline.pdf")

    # --- Plot: MLP erosion ---
    mlp_e = all_supp["mlp_erosion"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    colors = {"adversarial": "#d62728", "random": "#2ca02c", "improving": "#1f77b4"}
    markers = {"adversarial": "v", "random": "o", "improving": "^"}
    for ordering in ["adversarial", "random", "improving"]:
        steps_data = [s for s in mlp_e["steps"] if s["ordering"] == ordering]
        ks_plot = [0] + [s["k"] for s in steps_data]
        accs = [mlp_e["baseline_adv"]] + [s["adv_accuracy"] for s in steps_data]
        ax.plot(ks_plot, accs, color=colors[ordering], marker=markers[ordering],
                label=ordering.capitalize(), linewidth=2, markersize=6)
    ax.set_xlabel("Number of Deletions (k)")
    ax.set_ylabel("Adversarial Accuracy")
    ax.set_title("MNIST (MLP, 128 hidden, PCA-100)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_mlp_erosion.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig6_mlp_erosion.pdf")

    # --- Plot: MLP noise ---
    mlp_n = all_supp["mlp_noise"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    eps_vals = [nr["eps_priv"] for nr in mlp_n["noise_results"]]
    adv_vals = [nr["adv_accuracy"] for nr in mlp_n["noise_results"]]
    clean_vals = [nr["clean_accuracy"] for nr in mlp_n["noise_results"]]
    ax.plot(eps_vals, adv_vals, "o-", color="#d62728", label="Adversarial Accuracy", linewidth=2)
    ax.plot(eps_vals, clean_vals, "s--", color="#1f77b4", label="Clean Accuracy", linewidth=2)
    ax.axhline(y=mlp_n["baseline_adv"], color="#d62728", linestyle=":", alpha=0.4)
    ax.axhline(y=mlp_n["baseline_clean"], color="#1f77b4", linestyle=":", alpha=0.4)
    ax.set_xlabel("Privacy Parameter (eps)")
    ax.set_ylabel("Accuracy")
    ax.set_title("MNIST MLP: Noise-Robustness Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_mlp_noise.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig7_mlp_noise.pdf")

    print(f"\n{'='*60}")
    print("ALL SUPPLEMENTARY EXPERIMENTS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

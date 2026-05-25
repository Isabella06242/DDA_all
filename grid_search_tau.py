"""
Grid search over (tau1, tau2) pairs from 1–50, excluding diagonal (tau1 == tau2).
Uses vectorized batch operations to process all 862 events simultaneously per tau pair.
"""

import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ── helpers ─────────────────────────────────────────────────────────────────

def deriv_all_batch(data_2d, dm):
    """deriv_all vectorized over events. data_2d: (T, N) → returns (T-2*dm, N)."""
    T, N = data_2d.shape
    t = np.arange(dm, T - dm)
    ddata = np.zeros((len(t), N))
    for n1 in range(1, dm + 1):
        ddata += (data_2d[t + n1] - data_2d[t - n1]) / n1
    ddata /= dm
    return ddata


def dda_features_batch(Y, tau1, tau2, WL, WS, dm):
    """
    Compute averaged DDA features for all N events in Y for a fixed (tau1, tau2).
    Returns (N, 4) array: [a1, a2, a3, RMSE].
    """
    N = Y.shape[1]
    TAU = [tau1, tau2]
    TM = max(TAU)
    WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * dm - 1)) / WS))
    if WN < 1:
        return None

    sum_features = np.zeros((N, 4))
    n_valid = 0

    for wn in range(WN):
        anf = wn * WS
        ende = anf + WL + TM + 2 * dm - 1
        if ende + 1 > Y.shape[0]:
            break

        raw = Y[anf:ende + 1, :]        # (WL+TM+2dm, N)
        ddata = deriv_all_batch(raw, dm)  # (WL+TM,    N)
        clipped = raw[dm:-dm, :]          # (WL+TM,    N)

        # normalize each event independently
        mu = clipped.mean(axis=0, keepdims=True)
        sig = clipped.std(axis=0, ddof=1, keepdims=True)
        sig = np.where(sig == 0, 1.0, sig)
        DATA = (clipped - mu) / sig       # (WL+TM, N)
        dDATA = ddata / sig               # (WL+TM, N)

        # delay coordinates  (N, WL)
        xt1 = DATA[TM - TAU[0]: len(DATA) - TAU[0], :].T
        xt2 = DATA[TM - TAU[1]: len(DATA) - TAU[1], :].T
        yd  = dDATA[TM:, :].T

        # M shape: (N, WL, 3)
        M = np.stack([xt1, xt2, xt1 ** 2], axis=2)

        # batch normal equations
        MtM = M.transpose(0, 2, 1) @ M          # (N, 3, 3)
        Mty = (M.transpose(0, 2, 1) @ yd[:, :, np.newaxis]).squeeze(-1)  # (N, 3)

        # add tiny ridge to avoid singular matrices
        MtM += 1e-10 * np.eye(3)[np.newaxis]    # broadcast to (N, 3, 3)
        A = np.linalg.solve(MtM, Mty[:, :, np.newaxis]).squeeze(-1)  # (N, 3)

        # RMSE
        resid = yd - (M @ A[:, :, np.newaxis]).squeeze(-1)
        rmse = np.sqrt(np.mean(resid ** 2, axis=1))

        sum_features += np.column_stack([A, rmse])
        n_valid += 1

    if n_valid == 0:
        return None
    return sum_features / n_valid   # (N, 4)


# ── load data ────────────────────────────────────────────────────────────────

print("Loading data...")
Y_raw = genfromtxt("channel_182_alignment_activity.csv", delimiter=',')
Y_raw = Y_raw[:, 1:]                         # drop time column
Y = Y_raw[Y_raw.shape[0] // 2:, :]          # trim to second half (post-alignment)
print(f"Data shape after trim: {Y.shape}")

# labels: first 431 = full-single-opt (-1), next 431 = single-opt-first (1)
labels = np.array([-1] * 431 + [1] * 431)

# fixed train/test split (same as Classification.py)
train_idx, test_idx = train_test_split(
    np.arange(len(labels)), train_size=30,
    stratify=labels, random_state=33
)
y_train = labels[train_idx]
y_test  = labels[test_idx]

# DDA parameters
WL = 512
WS = 256
dm = 4

# ── grid search ──────────────────────────────────────────────────────────────

tau_range = range(1, 51)
results = []
total = sum(1 for t1 in tau_range for t2 in tau_range if t1 != t2)
done = 0

print(f"Searching {total} (tau1, tau2) pairs...\n")

for tau1 in tau_range:
    for tau2 in tau_range:
        if tau1 == tau2:
            continue

        feats = dda_features_batch(Y, tau1, tau2, WL, WS, dm)
        if feats is None:
            continue

        X_train = feats[train_idx]
        X_test  = feats[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_tr, y_train)

        train_auc = roc_auc_score(y_train, svm.decision_function(X_tr))
        test_auc  = roc_auc_score(y_test,  svm.decision_function(X_te))

        results.append({'tau1': tau1, 'tau2': tau2,
                        'train_auc': train_auc, 'test_auc': test_auc})
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total} done  |  best test AUC so far: "
                  f"{max(r['test_auc'] for r in results):.4f}")

# ── results ──────────────────────────────────────────────────────────────────

df = pd.DataFrame(results)
df.to_csv("grid_search_tau_results.csv", index=False)

best_test  = df.loc[df['test_auc'].idxmax()]
best_train = df.loc[df['train_auc'].idxmax()]

print("\n=== GRID SEARCH COMPLETE ===")
print(f"Best TEST  AUC: {best_test['test_auc']:.4f}  "
      f"(tau1={int(best_test['tau1'])}, tau2={int(best_test['tau2'])}, "
      f"train AUC={best_test['train_auc']:.4f})")
print(f"Best TRAIN AUC: {best_train['train_auc']:.4f}  "
      f"(tau1={int(best_train['tau1'])}, tau2={int(best_train['tau2'])}, "
      f"test AUC={best_train['test_auc']:.4f})")

print("\nTop 10 by TEST AUC:")
print(df.nlargest(10, 'test_auc').to_string(index=False))

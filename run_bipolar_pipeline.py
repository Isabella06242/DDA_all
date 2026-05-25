"""
Run the DDA pipeline on all .mat files in py/bipolar/.
Inlines the three pipeline stages (DDA_singlechannel → new_preprocess_classcification → Classification)
entirely in memory — no intermediate CSVs per channel.

Output: bipolar_results/{channel}_auc.csv  (one per channel)
        bipolar_results/all_channels_summary.csv
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from dda_functions import deriv_all

# ── DDA / classification parameters (same as current pipeline) ──────────────
TAU = [35, 34]
WL  = 512
WS  = 256
DM  = 4
TM  = max(TAU)
TRAIN_SIZE   = 30
RANDOM_STATE = 33

# ── helpers ──────────────────────────────────────────────────────────────────

def run_dda(Y: np.ndarray) -> np.ndarray:
    """
    DDA feature extraction on Y (time × events).
    Returns features array of shape (N_events, 4): [a1, a2, a3, RMSE].
    Mirrors the multi-series loop in DDA_singlechannel.py.
    """
    WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * DM - 1)) / WS))
    N  = Y.shape[1]
    ST = np.full((WN, 4, N), np.nan)

    for n_Y in range(N):
        for wn in range(WN):
            anf  = wn * WS
            ende = anf + WL + TM + 2 * DM - 1

            data  = Y[anf: ende + 1, n_Y]
            ddata = deriv_all(data, DM)
            data  = data[DM:-DM]

            STD   = np.std(data, ddof=1)
            if STD == 0:
                continue
            DATA  = (data - np.mean(data)) / STD
            dDATA = ddata / STD

            M = np.column_stack([
                DATA[TM - TAU[0]: len(DATA) - TAU[0]],
                DATA[TM - TAU[1]: len(DATA) - TAU[1]],
                (DATA[TM - TAU[0]: len(DATA) - TAU[0]]) ** 2,
            ])
            dDATA_sliced = dDATA[TM:]

            try:
                ST[wn, :3, n_Y] = np.linalg.solve(M.T @ M, M.T @ dDATA_sliced)
            except np.linalg.LinAlgError:
                ST[wn, :3, n_Y] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]
            ST[wn, 3, n_Y] = np.sqrt(np.mean((dDATA_sliced - M @ ST[wn, :3, n_Y]) ** 2))

    # average over windows → (N_events, 4)
    features = np.nanmean(ST, axis=0).T   # (N, 4)
    return features


def classify(features: np.ndarray, labels: np.ndarray) -> dict:
    """
    Train/test split → StandardScaler → linear SVM → AUC & accuracy.
    Mirrors Classification.py.
    """
    # adapt train_size if too large for this dataset
    n_min_class = np.min(np.bincount(labels + 1))   # smallest class count
    effective_train = min(TRAIN_SIZE, max(4, n_min_class - 2) * 2)

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        train_size=effective_train,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx],   labels[test_idx]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_tr, y_train)

    train_scores = svm.decision_function(X_tr)
    test_scores  = svm.decision_function(X_te)

    return {
        'train_auc':  roc_auc_score(y_train, train_scores),
        'test_auc':   roc_auc_score(y_test,  test_scores),
        'train_acc':  accuracy_score(y_train, svm.predict(X_tr)),
        'test_acc':   accuracy_score(y_test,  svm.predict(X_te)),
        'n_train':    int(effective_train),
        'n_test':     len(test_idx),
    }


def load_channel(mat_path: Path):
    """
    Load data and class indices from a bipolar .mat file.
    Returns Y (time × events, second-half trimmed) and labels array.
    """
    stem  = mat_path.stem                          # e.g. EMU038_inspection_clipped_bipolar_ch001
    parts = stem.split('_')
    subject   = parts[0]                           # EMU038 / EMU039
    condition = parts[1]                           # inspection / outcome
    ch_str    = parts[-1]                          # ch001

    with h5py.File(mat_path, 'r') as f:
        data_key = f'{condition}_{ch_str}_data'
        raw = f[data_key][:]                       # (N_events, 4096)

        if condition == 'inspection':
            idx1 = f['full_single_opt_info_ind'][:].flatten().astype(int) - 1
            idx2 = f['single_opt_first_inspection_ind'][:].flatten().astype(int) - 1
        else:   # outcome
            idx1 = f['win_ind'][:].flatten().astype(int) - 1
            idx2 = f['loss_ind'][:].flatten().astype(int) - 1

    # (N_events, 4096) → (4096, N_events) then trim to second half
    Y_full = raw.T                                 # (4096, N_events)
    Y_c1   = Y_full[:, idx1]
    Y_c2   = Y_full[:, idx2]
    Y      = np.hstack([Y_c1, Y_c2])
    Y      = Y[Y.shape[0] // 2:, :]               # post-alignment half

    labels = np.array([-1] * len(idx1) + [1] * len(idx2))
    return Y, labels, len(idx1), len(idx2)


# ── main loop ────────────────────────────────────────────────────────────────

bipolar_dir  = Path('bipolar')
results_dir  = Path('bipolar_results')
results_dir.mkdir(exist_ok=True)

mat_files = sorted(bipolar_dir.glob('*.mat'))
print(f"Found {len(mat_files)} .mat files\n")

summary_rows = []

for mat_path in mat_files:
    print(f"Processing {mat_path.name} ...", end=' ', flush=True)
    try:
        Y, labels, n1, n2 = load_channel(mat_path)
        features = run_dda(Y)
        metrics  = classify(features, labels)

        row = {
            'channel':   mat_path.stem,
            'n_class1':  n1,
            'n_class2':  n2,
            'tau1':      TAU[0],
            'tau2':      TAU[1],
            **metrics,
        }
        summary_rows.append(row)

        # per-channel file
        ch_df = pd.DataFrame([
            {'metric': k, 'value': v} for k, v in row.items()
        ])
        ch_df.to_csv(results_dir / f'{mat_path.stem}_auc.csv', index=False)

        print(f"test_auc={metrics['test_auc']:.3f}  train_auc={metrics['train_auc']:.3f}")

    except Exception as e:
        print(f"ERROR: {e}")
        summary_rows.append({'channel': mat_path.stem, 'error': str(e)})

# summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(results_dir / 'all_channels_summary.csv', index=False)

print(f"\nDone. Results in {results_dir}/")
print(summary_df[['channel', 'train_auc', 'test_auc']].to_string(index=False))

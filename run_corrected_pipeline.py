"""
Corrected DDA AUC pipeline for all patients EXCEPT EMU025.

Conditions:
  Class 1 (-1): full_single_opt_info_align_times
  Class 2 (+1): single_opt_first_inspection_align_times

Data source: /media/Data/Human_Intracranial_MAD/analysis/Adam_reformat/bipolar/
  - {patient}_SES{N}_neural_data_bipolar.mat  → lfpdata (n_samples × n_channels), Fs
  - {patient}_SES{N}_align_times.mat          → event time fields (seconds)

All available sessions are combined per patient.
Post-alignment window: POST_SAMPLES = 2048 samples = 1 s at Fs=2048 Hz.
DDA/SVM parameters unchanged from original pipeline: TAU=[35,34], WL=512, WS=256, DM=4.
"""

import sys, os, time, warnings
from typing import Optional
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────
BIPOLAR_DIR  = Path("/media/Data/Human_Intracranial_MAD/analysis/Adam_reformat/bipolar")
RESULTS_DIR  = Path("bipolar_results")
RESULTS_DIR.mkdir(exist_ok=True)

PATIENTS     = ["EMU024", "EMU030", "EMU036", "EMU037", "EMU038", "EMU039", "EMU051"]
SESSIONS     = [1, 2, 3, 4]   # try all; skip if file missing

POST_SAMPLES = 2048    # samples to take AFTER each align time (1 s at 2048 Hz)
TAU          = [35, 34]
WL           = 512
WS           = 256
DM           = 4
TM           = max(TAU)
TRAIN_SIZE   = 30
RANDOM_STATE = 33


# ── DDA helpers (vectorised over events) ────────────────────────────────────

def deriv_all_batch(data_2d: np.ndarray, dm: int) -> np.ndarray:
    """
    Symmetric derivative for all events at once.
    data_2d : (T, N)  →  returns (T - 2*dm, N)
    """
    T, N = data_2d.shape
    t = np.arange(dm, T - dm)
    ddata = np.zeros((len(t), N))
    for n1 in range(1, dm + 1):
        ddata += (data_2d[t + n1] - data_2d[t - n1]) / n1
    ddata /= dm
    return ddata


def dda_features_batch(Y: np.ndarray) -> Optional[np.ndarray]:
    """
    Y : (POST_SAMPLES, N_events)
    Returns (N_events, 4) feature matrix [a1, a2, a3, RMSE], or None if too short.
    """
    N = Y.shape[1]
    WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * DM - 1)) / WS))
    if WN < 1:
        return None

    sum_feat = np.zeros((N, 4))
    n_valid  = 0

    for wn in range(WN):
        anf  = wn * WS
        ende = anf + WL + TM + 2 * DM - 1
        if ende + 1 > Y.shape[0]:
            break

        raw    = Y[anf: ende + 1, :]         # (WL+TM+2dm, N)
        ddata  = deriv_all_batch(raw, DM)    # (WL+TM,     N)
        clipped = raw[DM:-DM, :]             # (WL+TM,     N)

        mu  = clipped.mean(axis=0, keepdims=True)
        sig = clipped.std(axis=0, ddof=1, keepdims=True)
        sig = np.where(sig == 0, 1.0, sig)
        DATA  = (clipped - mu) / sig
        dDATA = ddata / sig

        xt1 = DATA[TM - TAU[0]: len(DATA) - TAU[0], :].T   # (N, WL)
        xt2 = DATA[TM - TAU[1]: len(DATA) - TAU[1], :].T
        yd  = dDATA[TM:, :].T                               # (N, WL)

        M   = np.stack([xt1, xt2, xt1 ** 2], axis=2)       # (N, WL, 3)
        MtM = M.transpose(0, 2, 1) @ M                     # (N, 3, 3)
        Mty = (M.transpose(0, 2, 1) @ yd[:, :, np.newaxis]).squeeze(-1)

        MtM += 1e-10 * np.eye(3)
        try:
            A = np.linalg.solve(MtM, Mty[:, :, np.newaxis]).squeeze(-1)  # (N, 3)
        except np.linalg.LinAlgError:
            A = np.linalg.lstsq(
                MtM.reshape(-1, 3, 3)[0], Mty[0], rcond=None
            )[0][np.newaxis].repeat(N, axis=0)

        resid = yd - (M @ A[:, :, np.newaxis]).squeeze(-1)
        rmse  = np.sqrt(np.mean(resid ** 2, axis=1))

        sum_feat += np.column_stack([A, rmse])
        n_valid  += 1

    return sum_feat / n_valid if n_valid > 0 else None


# ── classification ───────────────────────────────────────────────────────────

def classify(features: np.ndarray, labels: np.ndarray) -> dict:
    n_min = min(np.sum(labels == -1), np.sum(labels == 1))
    effective_train = min(TRAIN_SIZE, max(4, n_min - 2) * 2)

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        train_size=effective_train,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    X_tr, X_te = features[train_idx], features[test_idx]
    y_tr, y_te = labels[train_idx],   labels[test_idx]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    svm = SVC(kernel="linear", C=1.0)
    svm.fit(X_tr, y_tr)

    return {
        "train_auc": roc_auc_score(y_tr, svm.decision_function(X_tr)),
        "test_auc":  roc_auc_score(y_te, svm.decision_function(X_te)),
        "n_train":   int(effective_train),
        "n_test":    len(test_idx),
    }


# ── epoch extraction ─────────────────────────────────────────────────────────

def load_align_times(at_path: Path):
    """Return (c1_times, c2_times) in seconds, NaNs removed."""
    with h5py.File(at_path, "r") as f:
        c1 = np.array(f["full_single_opt_info_align_times"]).ravel()
        c2 = np.array(f["single_opt_first_inspection_align_times"]).ravel()
    return c1[~np.isnan(c1)], c2[~np.isnan(c2)]


def extract_epochs(neural_path: Path, align_times: np.ndarray, fs: float):
    """
    Open the bipolar .mat file and slice POST_SAMPLES after each align time.
    Returns (n_valid_events, POST_SAMPLES, n_channels) float32 array.
    Skips events that would go out of bounds.
    """
    with h5py.File(neural_path, "r") as f:
        n_samples, n_channels = f["lfpdata"].shape
        starts = np.round(align_times * fs).astype(int)
        valid  = (starts >= 0) & (starts + POST_SAMPLES <= n_samples)
        starts = starts[valid]

        if len(starts) == 0:
            return np.empty((0, POST_SAMPLES, n_channels), dtype=np.float32)

        # Load epochs: one contiguous read per event (efficient HDF5 access)
        epochs = np.empty((len(starts), POST_SAMPLES, n_channels), dtype=np.float32)
        for i, s in enumerate(starts):
            epochs[i] = f["lfpdata"][s: s + POST_SAMPLES, :]

    return epochs


# ── main pipeline ─────────────────────────────────────────────────────────────

all_top5_rows = []

for patient in PATIENTS:
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"  {patient}")
    print(f"{'='*65}")

    # ── collect epochs from all available sessions ────────────────────
    c1_epochs_list, c2_epochs_list = [], []
    total_c1, total_c2 = 0, 0

    for ses in SESSIONS:
        at_path  = BIPOLAR_DIR / f"{patient}_SES{ses}_align_times.mat"
        np_path  = BIPOLAR_DIR / f"{patient}_SES{ses}_neural_data_bipolar.mat"

        if not at_path.exists() or not np_path.exists():
            continue

        # get Fs
        with h5py.File(np_path, "r") as f:
            Fs = float(np.squeeze(np.array(f["Fs"])))

        c1_times, c2_times = load_align_times(at_path)

        ep_c1 = extract_epochs(np_path, c1_times, Fs)
        ep_c2 = extract_epochs(np_path, c2_times, Fs)

        c1_epochs_list.append(ep_c1)
        c2_epochs_list.append(ep_c2)

        print(f"  SES{ses}: {len(ep_c1):3d} class-1 epochs,  "
              f"{len(ep_c2):3d} class-2 epochs  (Fs={Fs:.0f} Hz)")
        total_c1 += len(ep_c1)
        total_c2 += len(ep_c2)

    if total_c1 == 0 or total_c2 == 0:
        print(f"  [SKIP] No valid epochs found for {patient}.")
        continue

    # ── concatenate across sessions ───────────────────────────────────
    all_c1 = np.concatenate(c1_epochs_list, axis=0)  # (N_c1, POST, n_ch)
    all_c2 = np.concatenate(c2_epochs_list, axis=0)  # (N_c2, POST, n_ch)

    n_c1, _, n_ch = all_c1.shape
    n_c2           = all_c2.shape[0]

    print(f"  Combined: {n_c1} class-1, {n_c2} class-2, {n_ch} channels")
    print(f"  Memory: ~{(all_c1.nbytes + all_c2.nbytes) / 1e9:.2f} GB epoch buffer")

    # epochs stacked: (total_events, POST_SAMPLES, n_channels)
    all_epochs = np.concatenate([all_c1, all_c2], axis=0)
    labels     = np.array([-1] * n_c1 + [1] * n_c2)

    # free per-session lists
    del all_c1, all_c2, c1_epochs_list, c2_epochs_list

    # ── per-channel DDA + SVM ─────────────────────────────────────────
    pat_rows = []

    for ch in range(n_ch):
        # Y_ch : (POST_SAMPLES, n_events)
        Y_ch = all_epochs[:, :, ch].T.astype(np.float64)

        try:
            feats = dda_features_batch(Y_ch)
            if feats is None:
                raise ValueError("window too short")
            metrics = classify(feats, labels)
        except Exception as e:
            pat_rows.append({
                "patient": patient, "channel": ch + 1,
                "train_auc": np.nan, "test_auc": np.nan,
                "n_c1": n_c1, "n_c2": n_c2,
                "n_train": np.nan, "n_test": np.nan,
                "error": str(e),
            })
            continue

        pat_rows.append({
            "patient":   patient,
            "channel":   ch + 1,
            "train_auc": round(metrics["train_auc"], 6),
            "test_auc":  round(metrics["test_auc"],  6),
            "n_c1":      n_c1,
            "n_c2":      n_c2,
            "n_train":   metrics["n_train"],
            "n_test":    metrics["n_test"],
            "error":     "",
        })

        if (ch + 1) % 25 == 0 or ch == n_ch - 1:
            last = pat_rows[-1]
            print(f"  ch {ch+1:3d}/{n_ch}  test_auc={last['test_auc']:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    del all_epochs

    # ── save per-patient full results ─────────────────────────────────
    pat_df = pd.DataFrame(pat_rows)
    csv_path = RESULTS_DIR / f"{patient}_corrected_all_channels.csv"
    pat_df.to_csv(csv_path, index=False)
    print(f"  Saved all-channel results → {csv_path.name}")

    # ── top 5 by test_auc ────────────────────────────────────────────
    top5 = (pat_df.dropna(subset=["test_auc"])
                  .nlargest(5, "test_auc")
                  [["patient", "channel", "train_auc", "test_auc"]])
    all_top5_rows.append(top5)

    print(f"\n  Top 5 channels for {patient}:")
    print(top5.to_string(index=False))
    print(f"  Total elapsed: {time.time()-t0:.0f}s")

# ── save combined top-5 CSV ───────────────────────────────────────────────────
top5_df = pd.concat(all_top5_rows, ignore_index=True)
top5_path = Path("top5_channels_per_patient_corrected.csv")
top5_df.to_csv(top5_path, index=False)
print(f"\n{'='*65}")
print(f"Done. Combined top-5 → {top5_path}")
print(top5_df.to_string(index=False))

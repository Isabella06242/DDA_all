import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import joblib, json
from datetime import datetime

from classifier import compute_hyperplane_distances

# ======================
# 1. LOAD & SPLIT DATA
# ======================

data = pd.read_csv("classifier_ready_data/merged_dda_features.csv")

train_data, test_data = train_test_split(
    data, train_size=30, stratify=data['event_type'], random_state=33
)

features = ['a1', 'a2', 'a3', 'RMSE']

X_train = train_data[features].values
y_train = np.where(train_data['event_type'] == 'full-single-opt', -1, 1)
X_test  = test_data[features].values
y_test  = np.where(test_data['event_type']  == 'full-single-opt', -1, 1)

# ======================
# 2. TRAIN
# ======================

signed_distances, w, b, svm_model, scaler = compute_hyperplane_distances(
    X_train, y_train, standardize=True
)

# ======================
# 3. PREDICT
# ======================

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred  = svm_model.predict(X_test_scaled)

train_probs = svm_model.decision_function(X_train_scaled)
test_probs  = svm_model.decision_function(X_test_scaled)

# ======================
# 4. RESULTS
# ======================

from sklearn.metrics import balanced_accuracy_score, jaccard_score, hamming_loss, zero_one_loss

train_acc     = accuracy_score(y_train, y_train_pred) * 100
test_acc      = accuracy_score(y_test,  y_test_pred)  * 100
train_bal_acc = balanced_accuracy_score(y_train, y_train_pred) * 100
test_bal_acc  = balanced_accuracy_score(y_test,  y_test_pred)  * 100
train_auc     = roc_auc_score(y_train, train_probs)
test_auc      = roc_auc_score(y_test,  test_probs)
train_jaccard = jaccard_score(y_train, y_train_pred, average='binary', pos_label=1)
test_jaccard  = jaccard_score(y_test,  y_test_pred,  average='binary', pos_label=1)
train_hamming = hamming_loss(y_train, y_train_pred) * 100
test_hamming  = hamming_loss(y_test,  y_test_pred)  * 100
train_zo      = zero_one_loss(y_train, y_train_pred) * 100
test_zo       = zero_one_loss(y_test,  y_test_pred)  * 100

print(f"\nAccuracy          — Train: {train_acc:.1f}%   Test: {test_acc:.1f}%")
print(f"Balanced Accuracy — Train: {train_bal_acc:.1f}%   Test: {test_bal_acc:.1f}%")
print(f"AUC-ROC           — Train: {train_auc:.3f}  Test: {test_auc:.3f}")
print(f"Jaccard           — Train: {train_jaccard:.3f}  Test: {test_jaccard:.3f}")
print(f"Hamming Loss      — Train: {train_hamming:.1f}%   Test: {test_hamming:.1f}%")
print(f"Zero-One Loss     — Train: {train_zo:.1f}%   Test: {test_zo:.1f}%")
print(f"\nFeature weights: { {n: f'{v:.6f}' for n, v in zip(features, w)} }")
print(f"Bias: {b:.6f}")
print("\nDetailed Test Report:")
print(classification_report(y_test, y_test_pred, target_names=['full-single-opt (-1)', 'single-opt-first (1)']))

# ======================
# 5. SAVE MODEL & RESULTS
# ======================

joblib.dump({"svm_model": svm_model, "scaler": scaler, "weights": w, "bias": b,
             "train_accuracy": train_acc, "test_accuracy": test_acc}, "trained_dda_classifier.joblib")

output_file = "classification_results.json"
try:
    with open(output_file, 'r') as f:
        all_results = json.load(f)
        if not isinstance(all_results, list): all_results = [all_results]
except FileNotFoundError:
    all_results = []

results = {
    'timestamp': datetime.now().isoformat(),
    'TAU': f"[50, {len(all_results) - 2500}]",
    'train_accuracy': train_acc,   'test_accuracy': test_acc,
    'train_bal_acc':  train_bal_acc, 'test_bal_acc':  test_bal_acc,
    'train_auc':      train_auc,   'test_auc':      test_auc,
    'train_jaccard':  train_jaccard, 'test_jaccard':  test_jaccard,
    'train_hamming':  train_hamming, 'test_hamming':  test_hamming,
    'train_zero_one': train_zo,    'test_zero_one': test_zo,
    'weights': w.tolist(), 'bias': float(b),
}

all_results.append(results)
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {output_file} (total runs: {len(all_results)})")
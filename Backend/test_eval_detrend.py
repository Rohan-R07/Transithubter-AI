"""Evaluate detrended model on untouched exoTest.csv."""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

df_train = pd.read_csv("exoTrain.csv")
df_test = pd.read_csv("exoTest.csv")

X_train_raw = df_train.drop(columns=["LABEL"]).to_numpy(dtype=np.float64)
y_train = df_train["LABEL"].to_numpy(dtype=np.int64)

X_test_raw = df_test.drop(columns=["LABEL"]).to_numpy(dtype=np.float64)
y_test = df_test["LABEL"].to_numpy(dtype=np.int64)

def detrend_and_normalize(X, smooth_window=101):
    trend = uniform_filter1d(X, size=smooth_window, axis=1)
    resid = X - trend
    median_resid = np.median(resid, axis=1, keepdims=True)
    mad = np.median(np.abs(resid - median_resid), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1.0, mad)
    scaled = resid / (1.4826 * mad)
    return scaled

X_train_proc = detrend_and_normalize(X_train_raw, smooth_window=101)
X_test_proc = detrend_and_normalize(X_test_raw, smooth_window=101)

# Fit on all of exoTrain
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(X_train_proc, y_train)

# Predict probabilities on exoTest
test_probs = clf.predict_proba(X_test_proc)[:, 1]
y_test_bin = (y_test == 2).astype(int)

print("--- Test Set Exoplanet Probabilities (The 5 Exoplanets) ---")
pos_indices = np.where(y_test_bin == 1)[0]
for rank, idx in enumerate(pos_indices):
    print(f"  Exoplanet #{rank+1} (Row {idx}): P(Class 2) = {test_probs[idx]:.4f}")

print("\n--- Test Set Non-Exoplanet Probabilities Distribution ---")
neg_indices = np.where(y_test_bin == 0)[0]
p_neg = test_probs[neg_indices]
print(f"Non-Exoplanets (n={len(p_neg)}): min={p_neg.min():.4f}, median={np.median(p_neg):.4f}, 95%={np.percentile(p_neg, 95):.4f}, max={p_neg.max():.4f}")

print("\n--- Performance at Different Candidate Thresholds on exoTest.csv ---")
for t in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.50]:
    pred = (test_probs >= t).astype(int)
    cm = confusion_matrix(y_test_bin, pred)
    tn, fp, fn, tp = cm.ravel()
    p = precision_score(y_test_bin, pred, zero_division=0.0)
    r = recall_score(y_test_bin, pred, zero_division=0.0)
    f = f1_score(y_test_bin, pred, zero_division=0.0)
    print(f"T={t:.2f} | TP={tp}/5 (Rec={r*100:.1f}%), FP={fp:3d}/565, Precision={p*100:.2f}%, F1={f:.4f}")

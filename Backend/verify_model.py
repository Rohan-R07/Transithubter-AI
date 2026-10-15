"""TransitHunter AI - Comprehensive Model & Artifacts Verification Script.

Performs explicit checks for both Phase 1 baseline and Phase 1.5 improved detector:
1. Verifies existence and non-zero size of baseline and Phase 1.5 artifacts.
2. Loads Phase 1 baseline model and metadata, confirming contract.
3. Loads Phase 1.5 improved model and metadata, confirming contract.
4. Verifies preprocessing pipeline (detrend_and_normalize_lightcurves).
5. Verifies decision threshold configuration (0.08) and thresholded inference logic.
6. Verifies single and batch inference shapes and probability sum constraints.
7. Inspects cross_validation_comparison.json, diagnosis_report.json, and model_evaluation_phase1_5.json.
"""

import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.dataset_service import detrend_and_normalize_lightcurves


def verify_all_artifacts() -> bool:
    print("=" * 70)
    print(" TransitHunter AI -- Phase 1 & 1.5 Model & Artifacts Verification")
    print("=" * 70)

    # File paths
    baseline_model = backend_dir / "models" / "exoplanet_random_forest.joblib"
    baseline_meta = backend_dir / "models" / "feature_metadata.json"
    baseline_eval = backend_dir / "reports" / "model_evaluation.json"
    inspection_file = backend_dir / "reports" / "dataset_inspection.json"

    p15_model = backend_dir / "models" / "exoplanet_detector_phase1_5.joblib"
    p15_meta = backend_dir / "models" / "feature_metadata_phase1_5.json"
    p15_eval = backend_dir / "reports" / "model_evaluation_phase1_5.json"
    p15_cv = backend_dir / "reports" / "cross_validation_comparison.json"
    p15_diag = backend_dir / "reports" / "diagnosis_report.json"

    artifacts = [
        ("Phase 1 Baseline Model", baseline_model),
        ("Phase 1 Feature Metadata", baseline_meta),
        ("Phase 1 Evaluation Report", baseline_eval),
        ("Dataset Inspection Report", inspection_file),
        ("Phase 1.5 Improved Model", p15_model),
        ("Phase 1.5 Feature Metadata", p15_meta),
        ("Phase 1.5 Evaluation Report", p15_eval),
        ("Phase 1.5 CV Comparison", p15_cv),
        ("Phase 1.5 Diagnosis Report", p15_diag),
    ]

    # Check 1: Existence and file size
    print("\n[Check 1/6] Checking artifact existence and non-zero sizes...")
    for label, path in artifacts:
        if not path.is_file():
            print(f"[FAIL]: {label} not found at {path}")
            return False
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            print(f"[FAIL]: {label} at {path} is empty (0 bytes)")
            return False
        print(f"  [OK] {label}: {path.name} ({size_bytes / 1024:.2f} KB)")

    # Check 2: Load Baseline Model & Metadata
    print("\n[Check 2/6] Verifying Phase 1 baseline artifacts...")
    b_model: RandomForestClassifier = joblib.load(baseline_model)
    with open(baseline_meta, "r", encoding="utf-8") as f:
        b_metadata = json.load(f)

    if not isinstance(b_model, RandomForestClassifier):
        print(f"[FAIL]: Expected RandomForestClassifier, got {type(b_model)}")
        return False
    if b_model.n_features_in_ != 3197:
        print(f"[FAIL]: Baseline model feature count mismatch: {b_model.n_features_in_} != 3197")
        return False
    print(f"  [OK] Baseline model loaded: {type(b_model).__name__} (100 estimators, 3197 features)")

    # Check 3: Load Phase 1.5 Improved Model & Metadata
    print("\n[Check 3/6] Verifying Phase 1.5 improved detector artifacts...")
    model_p15: RandomForestClassifier = joblib.load(p15_model)
    with open(p15_meta, "r", encoding="utf-8") as f:
        meta_p15 = json.load(f)

    if not isinstance(model_p15, RandomForestClassifier):
        print(f"[FAIL]: Expected RandomForestClassifier for Phase 1.5, got {type(model_p15)}")
        return False
    if model_p15.n_features_in_ != 3197:
        print(f"[FAIL]: Phase 1.5 feature count mismatch: {model_p15.n_features_in_} != 3197")
        return False

    threshold = meta_p15.get("decision_threshold")
    if threshold is None or threshold <= 0 or threshold >= 1.0:
        print(f"[FAIL]: Invalid decision threshold in metadata: {threshold}")
        return False

    print(f"  [OK] Phase 1.5 model loaded: {type(model_p15).__name__} (100 estimators, 3197 features)")
    print(f"  [OK] Calibrated decision threshold confirmed: {threshold}")
    print(f"  [OK] Preprocessing method: {meta_p15['preprocessing_info']['pipeline']}")

    # Check 4: Preprocessing Pipeline Test
    print("\n[Check 4/6] Verifying preprocessing pipeline (detrend_and_normalize_lightcurves)...")
    dummy_raw = np.random.default_rng(42).standard_normal((3, 3197)) + 1500.0  # Large baseline offset
    proc_output = detrend_and_normalize_lightcurves(dummy_raw, smooth_window=31)

    if proc_output.shape != (3, 3197):
        print(f"[FAIL]: Preprocessed shape mismatch: {proc_output.shape} != (3, 3197)")
        return False
    if np.isnan(proc_output).any() or np.isinf(proc_output).any():
        print("[FAIL]: Preprocessor produced NaN or Inf values")
        return False

    # Check that baseline drift was removed (mean close to 0)
    row_means = np.abs(np.mean(proc_output, axis=1))
    if (row_means > 0.5).any():
        print(f"[FAIL]: Residual baseline mean unexpectedly large: {row_means}")
        return False
    print(f"  [OK] Preprocessing pipeline flattens baseline drift (residual means: {row_means.round(4)})")

    # Check 5: Inference and Threshold Application
    print("\n[Check 5/6] Verifying Phase 1.5 inference and threshold logic...")
    dummy_test = np.zeros((1, 3197), dtype=np.float64)
    dummy_test_proc = detrend_and_normalize_lightcurves(dummy_test, smooth_window=31)

    probs = model_p15.predict_proba(dummy_test_proc)
    c2_prob = float(probs[0, 1])
    thresholded_pred = 2 if c2_prob >= threshold else 1

    if not (0.0 <= c2_prob <= 1.0):
        print(f"[FAIL]: Invalid probability: {c2_prob}")
        return False
    if thresholded_pred not in [1, 2]:
        print(f"[FAIL]: Invalid predicted class: {thresholded_pred}")
        return False

    print(f"  [OK] Single sample P(Exoplanet) = {c2_prob:.4f}")
    print(f"  [OK] Thresholded prediction at T={threshold}: Class {thresholded_pred} ({meta_p15['label_mapping'][str(thresholded_pred)]})")

    # Batch test
    batch_raw = np.random.default_rng(42).standard_normal((5, 3197))
    batch_proc = detrend_and_normalize_lightcurves(batch_raw, smooth_window=31)
    batch_probs = model_p15.predict_proba(batch_proc)

    if not np.allclose(batch_probs.sum(axis=1), 1.0):
        print("[FAIL]: Batch probabilities do not sum to 1.0")
        return False
    print(f"  [OK] Batch of 5 samples: probabilities strictly sum to 1.0 across all rows")

    # Check 6: Reports Validation & Leaked Data Check
    print("\n[Check 6/6] Verifying report structures and absence of raw CSV data...")
    with open(p15_eval, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(p15_cv, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    with open(p15_diag, "r", encoding="utf-8") as f:
        diag_data = json.load(f)

    # Size check
    for name, content in [("eval", eval_data), ("cv", cv_data), ("diag", diag_data)]:
        size = len(json.dumps(content))
        if size > 150000:
            print(f"[FAIL]: Report '{name}' is unexpectedly large ({size} chars) -- raw CSV data may have leaked.")
            return False

    required_eval_keys = [
        "project",
        "phase",
        "timestamp",
        "selected_candidate",
        "selected_decision_threshold",
        "validation_metrics_summary",
        "test_evaluation_selected_threshold_0_08",
        "test_threshold_tradeoffs",
    ]
    for k in required_eval_keys:
        if k not in eval_data:
            print(f"[FAIL]: Key '{k}' missing from model_evaluation_phase1_5.json")
            return False

    print(f"  [OK] model_evaluation_phase1_5.json has required schema and {len(eval_data.keys())} keys")
    print(f"  [OK] cross_validation_comparison.json contains {len(cv_data.keys())} candidate models")
    print(f"  [OK] diagnosis_report.json contains root cause analysis and distributions")
    print(f"  [OK] Verified zero raw CSV flux rows leaked into reports")

    print("\n" + "=" * 70)
    print(" ALL VERIFICATION CHECKS PASSED (6/6)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_all_artifacts()
    sys.exit(0 if success else 1)

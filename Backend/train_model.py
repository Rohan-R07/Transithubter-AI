"""TransitHunter AI - Model Training and Evaluation Pipeline.

Phase 1:
- Dataset inspection & validation
- Genuine Random Forest training from scratch
- Test set evaluation (un-tampered exoTest.csv)
- Serialization of model and metadata artifacts
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

# Ensure backend root is in Python module search path
# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.dataset_service import (
    LABEL_MAPPING,
    inspect_dataset,
    load_and_preprocess_dataset,
    validate_datasets,
)
from app.services.training_service import (
    evaluate_model,
    load_and_verify_artifacts,
    save_artifacts,
    train_random_forest,
)


def find_dataset(filename: str, backend_path: Path) -> Path:
    """Search for dataset in workspace root, backend/data, or current working directory."""
    candidates = [
        backend_path.parent / filename,       # Workspace root (e.g., ../exoTrain.csv)
        backend_path / "data" / filename,     # backend/data/exoTrain.csv
        Path.cwd() / filename,                # Current working directory
        backend_path / filename,              # backend/exoTrain.csv
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Unable to locate '{filename}'. Looked in:\n" +
        "\n".join(f" - {c.resolve()}" for c in candidates)
    )


def run_pipeline(
    train_path: Path,
    test_path: Path,
    models_dir: Path,
    reports_dir: Path,
    n_estimators: int = 100,
    random_seed: int = 42,
    class_weight: str = "balanced",
) -> dict:
    """Execute the end-to-end Phase 1 training and evaluation pipeline."""
    print("=" * 70)
    print(" TransitHunter AI -- Phase 1 Dataset Validation & Model Training")
    print("=" * 70)
    print(f"Training dataset : {train_path}")
    print(f"Testing dataset  : {test_path}")
    print(f"Models directory : {models_dir}")
    print(f"Reports directory: {reports_dir}")
    print("-" * 70)

    # 1. Dataset Inspection & Validation
    print("\n[Step 1/5] Inspecting and validating datasets...")
    validation_results = validate_datasets(train_path, test_path)
    train_inspection = validation_results["train_inspection"]
    test_inspection = validation_results["test_inspection"]

    print(f"[OK] Training set: {train_inspection['row_count']} rows, {train_inspection['column_count']} columns")
    print(f"     Class 1 (Non-exoplanet): {train_inspection['label_info']['class_counts']['1']['count']} "
          f"({train_inspection['label_info']['class_counts']['1']['percentage']}%)")
    print(f"     Class 2 (Exoplanet)    : {train_inspection['label_info']['class_counts']['2']['count']} "
          f"({train_inspection['label_info']['class_counts']['2']['percentage']}%)")
    print(f"     Missing values: {train_inspection['missing_values_count']}, "
          f"Non-finite values: {train_inspection['non_finite_values_count']}, "
          f"Duplicates: {train_inspection['duplicate_rows_count']}")

    print(f"\n[OK] Testing set: {test_inspection['row_count']} rows, {test_inspection['column_count']} columns")
    print(f"     Class 1 (Non-exoplanet): {test_inspection['label_info']['class_counts']['1']['count']} "
          f"({test_inspection['label_info']['class_counts']['1']['percentage']}%)")
    print(f"     Class 2 (Exoplanet)    : {test_inspection['label_info']['class_counts']['2']['count']} "
          f"({test_inspection['label_info']['class_counts']['2']['percentage']}%)")
    print(f"     Missing values: {test_inspection['missing_values_count']}, "
          f"Non-finite values: {test_inspection['non_finite_values_count']}, "
          f"Duplicates: {test_inspection['duplicate_rows_count']}")

    print("[OK] Schema compatibility verified: Feature columns, counts, and ordering are identical.")

    # 2. Load and Preprocess Training Data
    print("\n[Step 2/5] Loading training data and preparing feature matrices...")
    X_train, y_train, feature_names = load_and_preprocess_dataset(train_path, is_training=True)
    print(f"[OK] Features matrix X_train shape: {X_train.shape}, dtype={X_train.dtype}")
    print(f"[OK] Labels vector y_train shape  : {y_train.shape}, dtype={y_train.dtype}")

    # 3. Train Model
    print(f"\n[Step 3/5] Training RandomForestClassifier from scratch...")
    print(f"     Hyperparameters: n_estimators={n_estimators}, class_weight='{class_weight}', "
          f"random_state={random_seed}, n_jobs=-1")
    model, train_config = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        n_estimators=n_estimators,
        random_state=random_seed,
        class_weight=class_weight,
    )
    print("[OK] Model training completed successfully.")

    # 4. Evaluate Model on Test Data
    print("\n[Step 4/5] Evaluating model on untouched test dataset (exoTest.csv)...")
    X_test, y_test, _ = load_and_preprocess_dataset(
        test_path,
        expected_features=feature_names,
        is_training=True,
    )
    eval_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_labels=[1, 2],
        class_names=LABEL_MAPPING,
    )
    eval_report = {
        "project": "TransitHunter AI",
        "phase": "Phase 1 - Dataset Validation and ML Model Training",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_configuration": train_config,
        "metrics": eval_metrics,
    }

    cm = eval_metrics["confusion_matrix"]
    print(f"[OK] Accuracy         : {eval_metrics['accuracy']:.4f}")
    print(f"[OK] Macro Precision  : {eval_metrics['precision_macro']:.4f}")
    print(f"[OK] Macro Recall     : {eval_metrics['recall_macro']:.4f}")
    print(f"[OK] Macro F1-Score   : {eval_metrics['f1_score_macro']:.4f}")
    print(f"[OK] Weighted F1-Score: {eval_metrics['f1_score_weighted']:.4f}")
    print("\nConfusion Matrix (labels: [1 = Non-Exoplanet, 2 = Exoplanet]):")
    print(f"  [TN: {cm['true_negative_class_1']:>4}, FP: {cm['false_positive_class_1_as_2']:>4}]")
    print(f"  [FN: {cm['false_negative_class_2_as_1']:>4}, TP: {cm['true_positive_class_2']:>4}]")
    print(f"\nPer-Class Breakdown:")
    for lbl, data in eval_metrics["per_class_metrics"].items():
        print(f"  Class {lbl} ({data['name']}): "
              f"Precision={data['precision']:.4f}, "
              f"Recall={data['recall']:.4f}, "
              f"F1={data['f1_score']:.4f}, "
              f"Support={data['support']}")

    # 5. Save Artifacts
    print("\n[Step 5/5] Saving model and reports artifacts...")
    feature_metadata = {
        "model_name": "exoplanet_random_forest",
        "model_type": "RandomForestClassifier",
        "library": "scikit-learn",
        "library_version": "1.9.0",
        "random_seed": random_seed,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "label_column": "LABEL",
        "label_mapping": {
            "1": "non_exoplanet",
            "2": "exoplanet"
        },
        "preprocessing_info": {
            "approach": "raw_flux_features_validated",
            "missing_values_imputed": False,
            "synthetic_sampling_applied": False,
            "feature_dtype": "float64",
            "normalization": "none"
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    dataset_inspection_report = {
        "project": "TransitHunter AI",
        "phase": "Phase 1 - Dataset Inspection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_inspection": train_inspection,
        "test_inspection": test_inspection,
        "validation_status": validation_results["status"],
        "schema_compatibility": validation_results["schemas_compatible"],
    }

    saved_paths = save_artifacts(
        model=model,
        feature_metadata=feature_metadata,
        evaluation_report=eval_report,
        dataset_inspection=dataset_inspection_report,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    for artifact_name, artifact_path in saved_paths.items():
        size_kb = Path(artifact_path).stat().st_size / 1024
        print(f"[OK] Saved {artifact_name}: {artifact_path} ({size_kb:.2f} KB)")

    # Post-training verification
    print("\n[Verification] Testing loaded artifacts contract and inference...")
    verification = load_and_verify_artifacts(
        model_path=saved_paths["model_file"],
        metadata_path=saved_paths["feature_metadata_file"],
    )
    print(f"[OK] Verification status: {verification['status']}")
    print(f"[OK] Model accepts shape (1, {verification['n_features_in']}) and output prediction = {verification['single_prediction_sample']}")
    print(f"[OK] Probabilities output for classes [1, 2] = {verification['single_probability_sample']}")

    print("\n" + "=" * 70)
    print(" Pipeline execution finished successfully!")
    print("=" * 70)

    return {
        "saved_paths": saved_paths,
        "validation": validation_results,
        "evaluation": eval_report,
        "verification": verification,
    }


def main():
    parser = argparse.ArgumentParser(description="TransitHunter AI - Model Training Pipeline")
    parser.add_argument("--train-path", type=str, default=None, help="Path to exoTrain.csv")
    parser.add_argument("--test-path", type=str, default=None, help="Path to exoTest.csv")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory to save models")
    parser.add_argument("--reports-dir", type=str, default=None, help="Directory to save reports")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--class-weight", type=str, default="balanced", help="Class weighting strategy")

    args = parser.parse_args()

    # Determine default paths
    train_path = Path(args.train_path) if args.train_path else find_dataset("exoTrain.csv", backend_dir)
    test_path = Path(args.test_path) if args.test_path else find_dataset("exoTest.csv", backend_dir)
    models_dir = Path(args.models_dir) if args.models_dir else backend_dir / "models"
    reports_dir = Path(args.reports_dir) if args.reports_dir else backend_dir / "reports"

    run_pipeline(
        train_path=train_path,
        test_path=test_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
        n_estimators=args.n_estimators,
        random_seed=args.random_seed,
        class_weight=args.class_weight,
    )


if __name__ == "__main__":
    main()

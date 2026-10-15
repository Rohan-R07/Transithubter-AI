"""Dataset service for TransitHunter AI.

Provides reusable functions for:
- Dataset inspection (schema, types, shapes, label distributions, missing values, duplicates)
- Dataset validation (feature alignment, non-finite checks, label verification)
- Dataset loading and preprocessing (separating labels, ensuring strict feature ordering)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


LABEL_COLUMN: str = "LABEL"
LABEL_MAPPING: Dict[int, str] = {
    1: "non_exoplanet",
    2: "exoplanet"
}
EXPECTED_LABELS: set = {1, 2}


def inspect_dataset(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect a CSV dataset and return comprehensive summary metadata.

    Args:
        file_path: Path to the CSV file to inspect.

    Returns:
        Dictionary containing:
        - file_name and file_path
        - row_count, column_count
        - label_column and unique label values
        - class_distribution (counts and percentages)
        - missing_values_count
        - non_finite_values_count (inf, -inf)
        - duplicate_rows_count
        - all_feature_columns_numeric (bool)
        - dtypes_summary
        - feature_count
        - feature_names_sample (first 5 and last 5)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist at expected path: {path.resolve()}")

    df = pd.read_csv(path)
    total_rows, total_cols = df.shape

    # Check for LABEL column
    has_label = LABEL_COLUMN in df.columns
    label_info: Dict[str, Any] = {
        "present": has_label,
        "column_name": LABEL_COLUMN if has_label else None,
        "unique_values": [],
        "class_counts": {},
        "class_percentages": {},
    }

    if has_label:
        unique_labels = sorted(df[LABEL_COLUMN].unique().tolist())
        label_info["unique_values"] = unique_labels

        counts = df[LABEL_COLUMN].value_counts().to_dict()
        proportions = (df[LABEL_COLUMN].value_counts(normalize=True) * 100).to_dict()

        for lbl, count in counts.items():
            str_lbl = str(lbl)
            label_name = LABEL_MAPPING.get(lbl, f"class_{lbl}")
            label_info["class_counts"][str_lbl] = {
                "name": label_name,
                "count": int(count),
                "percentage": round(float(proportions.get(lbl, 0.0)), 4)
            }
            label_info["class_percentages"][str_lbl] = round(float(proportions.get(lbl, 0.0)), 4)

    # Features
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    feature_count = len(feature_cols)

    # Missing & non-finite checks
    missing_count = int(df.isna().sum().sum())

    # Check non-finite values (inf / -inf) on numeric columns
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    all_features_numeric = len(numeric_feature_cols) == feature_count

    if all_features_numeric and feature_count > 0:
        non_finite_count = int(np.isinf(df[feature_cols].to_numpy(dtype=np.float64)).sum())
    else:
        non_finite_count = 0

    # Duplicates check
    duplicate_rows_count = int(df.duplicated().sum())

    # Data types breakdown
    dtypes_summary = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}

    # Sample feature column names
    first_five = feature_cols[:5] if feature_count >= 5 else feature_cols
    last_five = feature_cols[-5:] if feature_count >= 5 else feature_cols

    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "row_count": total_rows,
        "column_count": total_cols,
        "feature_count": feature_count,
        "dtypes_summary": dtypes_summary,
        "label_info": label_info,
        "missing_values_count": missing_count,
        "non_finite_values_count": non_finite_count,
        "duplicate_rows_count": duplicate_rows_count,
        "all_feature_columns_numeric": all_features_numeric,
        "feature_names_head": first_five,
        "feature_names_tail": last_five,
    }


def validate_datasets(
    train_path: Union[str, Path],
    test_path: Union[str, Path]
) -> Dict[str, Any]:
    """Validate that training and testing CSV files exist, are well-formed, and share identical schemas.

    Args:
        train_path: Path to training CSV (e.g., exoTrain.csv).
        test_path: Path to test CSV (e.g., exoTest.csv).

    Returns:
        Dictionary with validation results and flags.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If schema validation fails critically.
    """
    train_info = inspect_dataset(train_path)
    test_info = inspect_dataset(test_path)

    # Read column lists to verify exact ordering
    train_df = pd.read_csv(train_path, nrows=1)
    test_df = pd.read_csv(test_path, nrows=1)

    train_feature_cols = [c for c in train_df.columns if c != LABEL_COLUMN]
    test_feature_cols = [c for c in test_df.columns if c != LABEL_COLUMN]

    # Check 1: Feature count compatibility
    if len(train_feature_cols) != len(test_feature_cols):
        raise ValueError(
            f"Feature count mismatch: Train has {len(train_feature_cols)} features, "
            f"but Test has {len(test_feature_cols)} features."
        )

    # Check 2: Exact feature ordering compatibility
    if train_feature_cols != test_feature_cols:
        raise ValueError(
            "Feature names or ordering mismatch between train and test datasets."
        )

    # Check 3: Check that all features are numeric
    if not train_info["all_feature_columns_numeric"]:
        raise ValueError("Non-numeric feature columns detected in training dataset.")
    if not test_info["all_feature_columns_numeric"]:
        raise ValueError("Non-numeric feature columns detected in testing dataset.")

    # Check 4: Check that labels are valid {1, 2}
    train_labels = set(train_info["label_info"]["unique_values"])
    if not train_labels.issubset(EXPECTED_LABELS):
        raise ValueError(
            f"Unexpected labels in training dataset: {train_labels}. Expected subset of {EXPECTED_LABELS}."
        )

    test_labels = set(test_info["label_info"]["unique_values"])
    if not test_labels.issubset(EXPECTED_LABELS):
        raise ValueError(
            f"Unexpected labels in testing dataset: {test_labels}. Expected subset of {EXPECTED_LABELS}."
        )

    # Check 5: Non-finite and missing values check
    if train_info["missing_values_count"] > 0 or train_info["non_finite_values_count"] > 0:
        raise ValueError(
            f"Training dataset has invalid entries: {train_info['missing_values_count']} nulls, "
            f"{train_info['non_finite_values_count']} non-finite values."
        )

    if test_info["missing_values_count"] > 0 or test_info["non_finite_values_count"] > 0:
        raise ValueError(
            f"Testing dataset has invalid entries: {test_info['missing_values_count']} nulls, "
            f"{test_info['non_finite_values_count']} non-finite values."
        )

    return {
        "status": "VALID",
        "schemas_compatible": True,
        "feature_count": len(train_feature_cols),
        "train_rows": train_info["row_count"],
        "test_rows": test_info["row_count"],
        "labels_verified": True,
        "label_mapping": LABEL_MAPPING,
        "train_inspection": train_info,
        "test_inspection": test_info,
    }


def load_and_preprocess_dataset(
    file_path: Union[str, Path],
    expected_features: Optional[List[str]] = None,
    is_training: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Load a dataset from CSV, separate the label column, and prepare numeric feature arrays.

    Args:
        file_path: Path to the CSV file.
        expected_features: Optional expected feature column list for inference order enforcement.
        is_training: Whether the dataset is for training (requires LABEL column).

    Returns:
        Tuple of (X, y, feature_names):
        - X: 2D numpy array of float64 flux values, shape (n_samples, n_features)
        - y: 1D numpy array of int64 labels, shape (n_samples,) or None if unlabeled
        - feature_names: Ordered list of feature column names
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)

    # Handle labels
    y: Optional[np.ndarray] = None
    if LABEL_COLUMN in df.columns:
        y = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
        feature_df = df.drop(columns=[LABEL_COLUMN])
    else:
        if is_training:
            raise ValueError(f"Training dataset must contain the '{LABEL_COLUMN}' column.")
        feature_df = df

    # Feature ordering and alignment
    feature_names = list(feature_df.columns)
    if expected_features is not None:
        if set(feature_names) != set(expected_features):
            raise ValueError("Provided dataset features do not match expected features set.")
        # Ensure exact column ordering matching metadata
        feature_df = feature_df[expected_features]
        feature_names = expected_features

    # Convert to float64 numpy array
    X = feature_df.to_numpy(dtype=np.float64)

    # Verify no NaN or Inf
    if np.isnan(X).any():
        raise ValueError("NaN values encountered in feature matrix during preprocessing.")
    if np.isinf(X).any():
        raise ValueError("Inf/-Inf values encountered in feature matrix during preprocessing.")

    return X, y, feature_names


def detrend_and_normalize_lightcurves(
    X: np.ndarray,
    smooth_window: int = 31,
) -> np.ndarray:
    """Preprocess photometric flux light curves using continuum detrending and robust MAD scaling.

    Astronomical Rationale:
    1. Stellar rotation, variability, and telescope instrument drift introduce large baseline shifts
       across thousands of timestamps that dominate decision tree split criteria.
    2. Uniform moving-average smoothing estimates the stellar continuum baseline.
    3. Subtracting the continuum isolates localized, transit-induced photometric dips.
    4. Dividing by the Median Absolute Deviation (1.4826 * MAD) normalizes photometric noise
       to standard Gaussian units without being distorted by the transits themselves.

    Args:
        X: 2D numpy array of shape (n_samples, n_timestamps), float64 flux measurements.
        smooth_window: Window size for moving-average continuum estimation (default 31).

    Returns:
        2D numpy array of shape (n_samples, n_timestamps), zero-centered and noise-normalized.
    """
    from scipy.ndimage import uniform_filter1d

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of lightcurves, got shape {X.shape}")

    # 1. Estimate continuum trend across time axis (axis 1)
    trend = uniform_filter1d(X, size=smooth_window, axis=1, mode="nearest")

    # 2. Subtract continuum to isolate transit residuals
    residuals = X - trend

    # 3. Compute Median Absolute Deviation (MAD) row-wise
    median_residuals = np.median(residuals, axis=1, keepdims=True)
    mad = np.median(np.abs(residuals - median_residuals), axis=1, keepdims=True)

    # 4. Standardize: sigma ~ 1.4826 * MAD
    scale = 1.4826 * mad
    scale = np.where(scale == 0, 1.0, scale)

    return residuals / scale

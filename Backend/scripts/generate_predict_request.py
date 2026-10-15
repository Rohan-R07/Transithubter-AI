#!/usr/bin/env python3
"""TransitHunter AI - Prediction Request Generator Utility.

Extracts a single light-curve sample from exoTest.csv, validates the 3,197 numeric
flux features, and generates a valid request payload JSON for POST /predict.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EXPECTED_FEATURE_COUNT: int = 3197
LABEL_COLUMN: str = "LABEL"
DEFAULT_MODE: str = "balanced"
ALLOWED_MODES: Tuple[str, ...] = ("balanced", "high_recall")


def find_dataset(filename: str = "exoTest.csv", explicit_path: Optional[str] = None) -> Path:
    """Locate the dataset file without moving or duplicating it.

    Searches explicit path, workspace root, backend/data, current working directory,
    and backend root.
    """
    if explicit_path:
        p = Path(explicit_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified dataset file not found at: {p}")

    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    workspace_root = backend_dir.parent

    search_candidates = [
        workspace_root / filename,
        backend_dir / "data" / filename,
        Path.cwd() / filename,
        backend_dir / filename,
    ]

    for candidate in search_candidates:
        if candidate.is_file():
            return candidate.resolve()

    candidate_str = "\n".join(f"  - {c.resolve()}" for c in search_candidates)
    raise FileNotFoundError(
        f"Could not locate '{filename}'. Looked in:\n{candidate_str}\n"
        f"Please provide the dataset path using --dataset-path <path_to_{filename}>."
    )


def extract_flux_sample(
    csv_path: Path,
    row_index: int = 0,
    label_filter: Optional[int] = None,
) -> Tuple[List[float], Optional[int], int]:
    """Read dataset, select a row, validate features, and return flux values.

    Args:
        csv_path: Path to the dataset CSV file.
        row_index: Row index to select (ignored if label_filter is specified).
        label_filter: Optional label (1 or 2) to select the first matching row.

    Returns:
        Tuple of (flux_values, true_label, selected_row_index).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    try:
        # Read dataset with pandas
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV dataset '{csv_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset '{csv_path}' is empty.")

    selected_idx = row_index
    true_label: Optional[int] = None

    if label_filter is not None:
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"Cannot filter by label: column '{LABEL_COLUMN}' not found in CSV.")
        matching = df.index[df[LABEL_COLUMN] == label_filter].tolist()
        if not matching:
            raise ValueError(f"No rows found with {LABEL_COLUMN} == {label_filter}.")
        selected_idx = matching[0]

    if selected_idx < 0 or selected_idx >= len(df):
        raise IndexError(
            f"Row index {selected_idx} out of bounds for dataset with {len(df)} rows "
            f"(valid indices: 0 to {len(df) - 1})."
        )

    row_series = df.iloc[selected_idx]

    # Handle LABEL column
    if LABEL_COLUMN in row_series.index:
        true_label = int(row_series[LABEL_COLUMN])
        features_series = row_series.drop(labels=[LABEL_COLUMN])
    else:
        features_series = row_series

    # Validate feature count
    received_count = len(features_series)
    if received_count != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Invalid feature count: expected exactly {EXPECTED_FEATURE_COUNT} flux values, "
            f"but found {received_count} columns in row {selected_idx}."
        )

    # Validate all values are finite numeric values
    flux_values: List[float] = []
    invalid_entries: List[Tuple[int, Any, str]] = []

    for idx, (col_name, raw_val) in enumerate(features_series.items()):
        try:
            val_float = float(raw_val)
            if not math.isfinite(val_float):
                invalid_entries.append((idx, raw_val, "non-finite (NaN or Inf)"))
            else:
                flux_values.append(val_float)
        except (ValueError, TypeError) as conv_err:
            invalid_entries.append((idx, raw_val, f"conversion error: {conv_err}"))

    if invalid_entries:
        sample_bad = invalid_entries[:5]
        details = ", ".join(f"idx {i} ('{c}')={v!r} ({reason})" for i, v, reason in sample_bad)
        raise ValueError(
            f"Row {selected_idx} contains {len(invalid_entries)} invalid or non-finite flux value(s). "
            f"First invalid entries: {details}"
        )

    return flux_values, true_label, selected_idx


def build_predict_request(
    flux_values: List[float],
    mode: str = DEFAULT_MODE,
) -> Dict[str, Any]:
    """Construct and validate the request dictionary matching PredictRequest schema."""
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Allowed modes are: {ALLOWED_MODES}")

    if len(flux_values) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Cannot build request: expected {EXPECTED_FEATURE_COUNT} flux values, "
            f"got {len(flux_values)}."
        )

    return {
        "flux_values": flux_values,
        "mode": mode,
    }


def generate_predict_request_file(
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    row_index: int = 0,
    label_filter: Optional[int] = None,
    mode: str = DEFAULT_MODE,
) -> Path:
    """Generate the prediction request JSON file and return destination path."""
    csv_file = find_dataset("exoTest.csv", explicit_path=dataset_path)
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent

    if output_path:
        out_file = Path(output_path).resolve()
    else:
        out_file = backend_dir / "predict_request.json"

    flux_values, true_label, selected_idx = extract_flux_sample(
        csv_path=csv_file,
        row_index=row_index,
        label_filter=label_filter,
    )

    payload = build_predict_request(flux_values=flux_values, mode=mode)

    # Ensure parent directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    label_desc = f" (LABEL={true_label}: {'exoplanet' if true_label == 2 else 'non-exoplanet'})" if true_label else ""
    print(f"[OK] Read sample from: {csv_file}")
    print(f"[OK] Selected row index: {selected_idx}{label_desc}")
    print(f"[OK] Validated {len(flux_values)} numeric flux values (all finite floats)")
    print(f"[OK] Mode: '{mode}'")
    print(f"[OK] Generated request JSON: {out_file} ({out_file.stat().st_size:,} bytes)")

    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a POST /predict JSON request from exoTest.csv."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to exoTest.csv (searches workspace root by default)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for predict_request.json (default: backend/predict_request.json)",
    )
    parser.add_argument(
        "--row-index",
        "-r",
        type=int,
        default=0,
        help="Zero-based row index in exoTest.csv to extract (default: 0)",
    )
    parser.add_argument(
        "--label",
        "-l",
        type=int,
        choices=[1, 2],
        default=None,
        help="Filter first row matching label: 2=exoplanet candidate, 1=non-exoplanet",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=ALLOWED_MODES,
        default=DEFAULT_MODE,
        help="Prediction mode: 'balanced' (default, T=0.08) or 'high_recall' (T=0.05)",
    )

    args = parser.parse_args()

    try:
        out_path = generate_predict_request_file(
            dataset_path=args.dataset_path,
            output_path=args.output,
            row_index=args.row_index,
            label_filter=args.label,
            mode=args.mode,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

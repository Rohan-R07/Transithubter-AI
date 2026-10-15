"""Model loader — loads the Phase 1.5 model exactly once at application startup.

Responsibilities:
- Deserialize exoplanet_detector_phase1_5.joblib
- Load feature_metadata_phase1_5.json
- Provide a ModelState singleton to the rest of the application
- Raise ModelNotReadyError cleanly when the model is unavailable
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to the Backend/ directory)
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # Backend/
_MODEL_PATH = _BACKEND_ROOT / "models" / "exoplanet_detector_phase1_5.joblib"
_METADATA_PATH = _BACKEND_ROOT / "models" / "feature_metadata_phase1_5.json"
_EVAL_REPORT_PATH = _BACKEND_ROOT / "reports" / "model_evaluation_phase1_5.json"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelNotReadyError(RuntimeError):
    """Raised when prediction is attempted but the model is not loaded."""


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class ModelState:
    """Holds the loaded model and all associated metadata."""

    model: Optional[RandomForestClassifier] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    eval_report: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    feature_count: int = 0
    decision_threshold: float = 0.08
    high_recall_threshold: float = 0.05
    label_mapping: Dict[str, str] = field(default_factory=lambda: {"1": "non_exoplanet", "2": "exoplanet"})
    classes: List[int] = field(default_factory=lambda: [1, 2])
    smooth_window: int = 31
    load_error: Optional[str] = None
    is_ready: bool = False


# Module-level singleton
_state = ModelState()


def load_model() -> ModelState:
    """Load model and metadata from disk into the global singleton.

    Called once during FastAPI lifespan startup.  Never retrains the model.
    On any error the state is left with ``is_ready=False`` and the error
    message stored in ``load_error``.
    """
    global _state

    logger.info("Loading Phase 1.5 exoplanet detector …")

    # ---- validate paths ----
    missing = [p for p in (_MODEL_PATH, _METADATA_PATH) if not p.exists()]
    if missing:
        msg = f"Required model artifacts not found: {[str(p) for p in missing]}"
        logger.error(msg)
        _state.load_error = msg
        _state.is_ready = False
        return _state

    try:
        model: RandomForestClassifier = joblib.load(_MODEL_PATH)
        with _METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata: Dict[str, Any] = json.load(f)

        # Optional eval report — load if present, otherwise skip silently
        eval_report: Dict[str, Any] = {}
        if _EVAL_REPORT_PATH.exists():
            with _EVAL_REPORT_PATH.open("r", encoding="utf-8") as f:
                eval_report = json.load(f)

        # ---- validate compatibility ----
        expected_features = int(metadata["feature_count"])
        if model.n_features_in_ != expected_features:
            raise ValueError(
                f"Model expects {model.n_features_in_} features but metadata says {expected_features}."
            )

        classes_in_model = [int(c) for c in model.classes_]
        if set(classes_in_model) != {1, 2}:
            raise ValueError(f"Unexpected model classes: {classes_in_model}. Expected [1, 2].")

        # ---- run a single dummy inference to catch serialisation issues ----
        dummy = np.zeros((1, expected_features), dtype=np.float64)
        _ = model.predict_proba(dummy)

        preprocessing = metadata.get("preprocessing_info", {})
        smooth_window = int(preprocessing.get("smooth_window", 31))

        _state.model = model
        _state.metadata = metadata
        _state.eval_report = eval_report
        _state.feature_names = list(metadata["feature_names"])
        _state.feature_count = expected_features
        _state.decision_threshold = float(metadata.get("decision_threshold", 0.08))
        _state.high_recall_threshold = float(
            metadata.get("decision_threshold_options", {}).get("high_recall_sensitive", 0.05)
        )
        _state.label_mapping = {str(k): v for k, v in metadata.get("label_mapping", {"1": "non_exoplanet", "2": "exoplanet"}).items()}
        _state.classes = classes_in_model
        _state.smooth_window = smooth_window
        _state.load_error = None
        _state.is_ready = True

        logger.info(
            "Model loaded: %s  |  features=%d  |  threshold=%.2f  |  window=%d",
            metadata.get("model_name", "exoplanet_detector_phase1_5"),
            expected_features,
            _state.decision_threshold,
            smooth_window,
        )

    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load model: {type(exc).__name__}: {exc}"
        logger.error(msg)
        _state.model = None
        _state.load_error = msg
        _state.is_ready = False

    return _state


def get_state() -> ModelState:
    """Return the global model state singleton."""
    return _state


def require_model() -> RandomForestClassifier:
    """Return the loaded model or raise ModelNotReadyError."""
    if not _state.is_ready or _state.model is None:
        detail = _state.load_error or "Model has not been loaded."
        raise ModelNotReadyError(detail)
    return _state.model

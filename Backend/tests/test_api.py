"""Test suite for TransitHunter AI FastAPI endpoints.

Run from the Backend/ directory:
    pytest tests/ -v

Tests:
    1.  GET /health — model loaded
    2.  GET /model/metadata — schema and key fields
    3.  POST /predict — valid 3,197-feature input (balanced mode)
    4.  POST /predict — valid input in high_recall mode
    5.  POST /predict — wrong feature count (too few)
    6.  POST /predict — wrong feature count (too many)
    7.  POST /predict — non-numeric / Pydantic coercion failure
    8.  POST /predict — NaN value in flux_values
    9.  POST /predict — Inf value in flux_values
    10. POST /predict — model not loaded (mocked failure)
    11. Preprocessing consistency: detrended output has correct shape / no NaN
    12. Class-label mapping: class 1 → non_exoplanet, class 2 → exoplanet
"""

from __future__ import annotations

import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App import — triggers lifespan which loads the model
# ---------------------------------------------------------------------------
from app.main import app
from app.core import model_loader
from app.services.dataset_service import detrend_and_normalize_lightcurves

EXPECTED_FEATURES = 3197
_VALID_FLUX = [0.5] * EXPECTED_FEATURES  # all identical (non-zero but same value)
_VALID_FLUX_VARIED = list(np.random.default_rng(42).normal(0, 500, EXPECTED_FEATURES))


@pytest.fixture(scope="module")
def client():
    """Shared test client — model is loaded once."""
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1. Health endpoint
# ===========================================================================


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_health_model_loaded(client):
    """Model should be loaded if artifact files are present."""
    response = client.get("/health")
    body = response.json()
    # If model_loaded is False, print load_error for debugging
    if not body["model_loaded"]:
        pytest.skip(f"Model not loaded: {body.get('load_error')}")
    assert body["model_loaded"] is True
    assert body["model_name"] is not None


# ===========================================================================
# 2. Model metadata endpoint
# ===========================================================================


def test_model_metadata_schema(client):
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    response = client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()

    # Required top-level keys
    for key in (
        "model_name", "phase", "model_type", "library",
        "feature_count", "expected_input", "label_mapping",
        "default_threshold", "threshold_options", "preprocessing",
    ):
        assert key in body, f"Missing key: {key}"

    assert body["feature_count"] == EXPECTED_FEATURES
    assert body["default_threshold"] == pytest.approx(0.08)
    assert body["label_mapping"]["1"] == "non_exoplanet"
    assert body["label_mapping"]["2"] == "exoplanet"


def test_model_metadata_threshold_options(client):
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    response = client.get("/model/metadata")
    body = response.json()
    opts = body["threshold_options"]
    assert opts["balanced_recommended"] == pytest.approx(0.08)
    assert opts["high_recall_sensitive"] == pytest.approx(0.05)


def test_model_metadata_preprocessing(client):
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    response = client.get("/model/metadata")
    body = response.json()
    preproc = body["preprocessing"]
    assert preproc["smooth_window"] == 31
    assert "MAD" in preproc["algorithm"] or "mad" in preproc["algorithm"].lower()


# ===========================================================================
# 3. Predict — valid input, balanced mode
# ===========================================================================


def test_predict_valid_balanced(client):
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    response = client.post("/predict", json={"flux_values": _VALID_FLUX_VARIED, "mode": "balanced"})
    assert response.status_code == 200
    body = response.json()

    assert body["predicted_class"] in (1, 2)
    assert body["predicted_label"] in ("non_exoplanet", "exoplanet")
    assert isinstance(body["is_exoplanet_candidate"], bool)
    assert 0.0 <= body["probability_exoplanet"] <= 1.0
    assert 0.0 <= body["probability_non_exoplanet"] <= 1.0
    assert math.isclose(
        body["probability_exoplanet"] + body["probability_non_exoplanet"], 1.0, abs_tol=1e-5
    )
    assert body["decision_threshold"] == pytest.approx(0.08)
    assert body["mode"] == "balanced"
    assert len(body["explanation"]) > 10


# ===========================================================================
# 4. Predict — valid input, high_recall mode
# ===========================================================================


def test_predict_valid_high_recall(client):
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    response = client.post("/predict", json={"flux_values": _VALID_FLUX_VARIED, "mode": "high_recall"})
    assert response.status_code == 200
    body = response.json()
    assert body["decision_threshold"] == pytest.approx(0.05)
    assert body["mode"] == "high_recall"


# ===========================================================================
# 5. Wrong feature count — too few
# ===========================================================================


def test_predict_too_few_features(client):
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    payload = {"flux_values": [1.0] * 100}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "100" in str(body) or "feature" in str(body).lower() or "detail" in body


# ===========================================================================
# 6. Wrong feature count — too many
# ===========================================================================


def test_predict_too_many_features(client):
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    payload = {"flux_values": [1.0] * (EXPECTED_FEATURES + 100)}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ===========================================================================
# 7. Non-numeric input (Pydantic rejects strings)
# ===========================================================================


def test_predict_non_numeric_input(client):
    bad_payload = {"flux_values": ["not_a_number"] + [0.0] * (EXPECTED_FEATURES - 1)}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ===========================================================================
# 8. NaN value
# ===========================================================================


def test_predict_nan_value(client):
    # Python's json module rejects float('nan') — send as raw JSON bytes instead
    # JSON spec does not support NaN; the server must reject it as a parse error (422).
    import json as _json
    good_part = _json.dumps([0.5] * (EXPECTED_FEATURES - 1))[1:-1]  # strip outer brackets
    raw = f'{{"flux_values": [NaN, {good_part}]}}'
    response = client.post("/predict", content=raw.encode(), headers={"Content-Type": "application/json"})
    # NaN is not valid JSON → server returns 422 (unprocessable)
    assert response.status_code == 422


# ===========================================================================
# 9. Inf value
# ===========================================================================


def test_predict_inf_value(client):
    # Infinity is not valid JSON — server must reject it (422).
    import json as _json
    good_part = _json.dumps([0.5] * (EXPECTED_FEATURES - 1))[1:-1]
    raw = f'{{"flux_values": [Infinity, {good_part}]}}'
    response = client.post("/predict", content=raw.encode(), headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# ===========================================================================
# 10. Model not loaded (mocked)
# ===========================================================================


def test_predict_model_not_loaded(client):
    """When the model is not available, predict must return 503 — never a fake result."""
    from app.core.model_loader import ModelNotReadyError
    # Patch require_model in the router's own namespace (where it was imported)
    with patch("app.routers.predict.require_model", side_effect=ModelNotReadyError("Mocked failure")):
        response = client.post("/predict", json={"flux_values": _VALID_FLUX_VARIED})

    assert response.status_code == 503
    body = response.json()
    assert "detail" in body


# ===========================================================================
# 11. Preprocessing consistency
# ===========================================================================


def test_preprocessing_shape_and_finite():
    """detrend_and_normalize_lightcurves must return same shape with all-finite values."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1000, (5, EXPECTED_FEATURES)).astype(np.float64)
    result = detrend_and_normalize_lightcurves(X, smooth_window=31)

    assert result.shape == X.shape, "Output shape must match input shape."
    assert np.isfinite(result).all(), "All output values must be finite after detrending."


def test_preprocessing_reduces_baseline():
    """After detrending, row-wise median should be near zero."""
    rng = np.random.default_rng(1)
    # Simulate strong linear drift
    baseline = np.linspace(-5000, 5000, EXPECTED_FEATURES)
    noise = rng.normal(0, 10, (3, EXPECTED_FEATURES))
    X = noise + baseline

    result = detrend_and_normalize_lightcurves(X, smooth_window=31)
    row_medians = np.median(result, axis=1)
    # Medians should be much closer to 0 than the raw baseline spread
    assert np.all(np.abs(row_medians) < 1.0), (
        f"Row medians after detrending should be near zero, got {row_medians}"
    )


# ===========================================================================
# 12. Class-label mapping interpretation
# ===========================================================================


def test_class_label_mapping():
    """Confirm model state maps class 1 → non_exoplanet, class 2 → exoplanet."""
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    assert state.label_mapping["1"] == "non_exoplanet"
    assert state.label_mapping["2"] == "exoplanet"
    assert state.classes == [1, 2]


def test_predict_consistency_class_label(client):
    """is_exoplanet_candidate must match (predicted_class == 2)."""
    state = model_loader.get_state()
    if not state.is_ready:
        pytest.skip("Model not loaded.")

    response = client.post("/predict", json={"flux_values": _VALID_FLUX_VARIED})
    assert response.status_code == 200
    body = response.json()
    assert body["is_exoplanet_candidate"] == (body["predicted_class"] == 2)
    assert body["predicted_label"] == ("exoplanet" if body["is_exoplanet_candidate"] else "non_exoplanet")

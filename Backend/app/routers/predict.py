"""POST /predict router.

Preprocessing mirrors the Phase 1.5 training pipeline exactly:
    detrend_and_normalize_lightcurves(smooth_window=31)
as implemented in app/services/dataset_service.py.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, status

from app.core.model_loader import ModelNotReadyError, get_state, require_model
from app.schemas import PredictRequest, PredictResponse, ThresholdMode
from app.services.dataset_service import detrend_and_normalize_lightcurves

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

# The class index for "exoplanet" in model.classes_ = [1, 2]
_CLASS2_LABEL = 2
_CLASS1_LABEL = 1


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict exoplanet transit",
    description=(
        "Accepts exactly 3,197 photometric flux values from a single stellar light curve "
        "and returns the exoplanet transit probability and classification decision.\n\n"
        "**Modes:**\n"
        "- `balanced` (default): threshold 0.08 — CV-selected, low false-positive rate.\n"
        "- `high_recall`: threshold 0.05 — catches more candidates, higher false-positive rate (~1.9%).\n\n"
        "**Preprocessing**: Flux values are detrended using a uniform moving-average "
        "(window=31) and normalised with robust 1.4826×MAD scaling — identical to training."
    ),
    responses={
        422: {"description": "Validation error — wrong feature count, NaN/Inf values, or bad JSON."},
        503: {"description": "Model not loaded — check /health."},
    },
)
def predict(request: PredictRequest) -> PredictResponse:
    # ---- model availability ----
    try:
        model = require_model()
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available for inference. Check /health for details.",
        ) from exc

    state = get_state()

    # ---- feature count validation ----
    expected = state.feature_count
    received = len(request.flux_values)
    if received != expected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Feature count mismatch: expected {expected} flux values "
                f"(FLUX.1 … FLUX.{expected}), received {received}."
            ),
        )

    # ---- convert to numpy and validate (NaN/Inf already checked by Pydantic) ----
    try:
        X_raw = np.array(request.flux_values, dtype=np.float64).reshape(1, -1)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not convert flux_values to a numeric array: {exc}",
        ) from exc

    # ---- preprocessing — exact replica of training pipeline ----
    try:
        X_processed = detrend_and_normalize_lightcurves(X_raw, smooth_window=state.smooth_window)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preprocessing failed.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Preprocessing failed: {type(exc).__name__}. Ensure all flux values are valid finite numbers.",
        ) from exc

    # ---- check for zero-MAD edge case after preprocessing ----
    if not np.isfinite(X_processed).all():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "After detrending, the light curve contains non-finite values. "
                "This can occur when all flux values are identical (zero variance). "
                "Please provide a valid variable-star light curve."
            ),
        )

    # ---- resolve threshold ----
    if request.mode == ThresholdMode.high_recall:
        threshold = state.high_recall_threshold
    else:
        threshold = state.decision_threshold

    # ---- inference ----
    try:
        proba = model.predict_proba(X_processed)  # shape (1, 2) — [P(class1), P(class2)]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model inference failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference encountered an unexpected error.",
        ) from exc

    # model.classes_ = [1, 2] — confirmed at load time
    c2_idx = list(model.classes_).index(_CLASS2_LABEL)
    c1_idx = list(model.classes_).index(_CLASS1_LABEL)

    prob_exoplanet = float(proba[0, c2_idx])
    prob_non_exoplanet = float(proba[0, c1_idx])

    is_exoplanet = prob_exoplanet >= threshold
    predicted_class = _CLASS2_LABEL if is_exoplanet else _CLASS1_LABEL
    predicted_label = state.label_mapping.get(str(predicted_class), str(predicted_class))

    # ---- human-readable explanation ----
    mode_label = "balanced (T=0.08)" if request.mode == ThresholdMode.balanced else "high-recall (T=0.05)"
    if is_exoplanet:
        explanation = (
            f"The light curve shows a photometric dip pattern consistent with a transit signal. "
            f"The model assigned P(exoplanet) = {prob_exoplanet:.4f}, which exceeds the "
            f"{mode_label} decision threshold of {threshold}. "
            f"This star is flagged as an exoplanet candidate for follow-up analysis."
        )
    else:
        explanation = (
            f"No significant transit signal was detected. "
            f"The model assigned P(exoplanet) = {prob_exoplanet:.4f}, which is below the "
            f"{mode_label} decision threshold of {threshold}. "
            f"The light curve is consistent with a non-transiting star."
        )

    return PredictResponse(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        is_exoplanet_candidate=is_exoplanet,
        probability_exoplanet=round(prob_exoplanet, 6),
        probability_non_exoplanet=round(prob_non_exoplanet, 6),
        decision_threshold=threshold,
        mode=request.mode.value,
        explanation=explanation,
    )

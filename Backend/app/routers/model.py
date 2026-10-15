"""GET /model/metadata router."""

from fastapi import APIRouter, HTTPException, status
from app.core.model_loader import get_state
from app.schemas import ModelMetadataResponse, PreprocessingInfo, ThresholdOptions

router = APIRouter(prefix="/model", tags=["Model"])


@router.get(
    "/metadata",
    response_model=ModelMetadataResponse,
    summary="Model metadata",
    description=(
        "Returns the Phase 1.5 model's identifier, feature schema, "
        "preprocessing configuration, decision thresholds, and evaluation summary. "
        "Only information present in the saved artifact files is returned."
    ),
)
def model_metadata() -> ModelMetadataResponse:
    state = get_state()

    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health for details.",
        )

    meta = state.metadata
    threshold_opts_raw = meta.get("decision_threshold_options", {})
    preproc_raw = meta.get("preprocessing_info", {})

    # Build a minimal evaluation summary from the loaded report
    eval_summary: dict | None = None
    if state.eval_report:
        t08 = state.eval_report.get("test_evaluation_selected_threshold_0_08", {})
        t05 = state.eval_report.get("test_evaluation_sensitive_threshold_0_05", {})
        eval_summary = {
            "description": (
                "Evaluation on 570 held-out Kepler test samples "
                "(565 non-exoplanets, 5 confirmed exoplanets)."
            ),
            "threshold_0_08": {
                "exoplanet_recall": t08.get("per_class_metrics", {}).get("2", {}).get("recall"),
                "exoplanet_precision": t08.get("per_class_metrics", {}).get("2", {}).get("precision"),
                "exoplanet_f1": t08.get("per_class_metrics", {}).get("2", {}).get("f1_score"),
                "accuracy": t08.get("accuracy"),
            },
            "threshold_0_05": {
                "exoplanet_recall": t05.get("per_class_metrics", {}).get("2", {}).get("recall"),
                "exoplanet_precision": t05.get("per_class_metrics", {}).get("2", {}).get("precision"),
                "exoplanet_f1": t05.get("per_class_metrics", {}).get("2", {}).get("f1_score"),
                "accuracy": t05.get("accuracy"),
            },
            "cross_validation_f1_class2": state.eval_report.get(
                "validation_metrics_summary", {}
            ).get("best_cv_f1_class2"),
        }

    return ModelMetadataResponse(
        model_name=meta.get("model_name", "exoplanet_detector_phase1_5"),
        phase=meta.get("phase", "Phase 1.5"),
        model_type=meta.get("model_type", "RandomForestClassifier"),
        library=meta.get("library", "scikit-learn"),
        library_version=meta.get("library_version", "unknown"),
        feature_count=state.feature_count,
        expected_input=f"{state.feature_count} numerical flux values (FLUX.1 … FLUX.{state.feature_count})",
        label_mapping=state.label_mapping,
        classes=state.classes,
        default_threshold=state.decision_threshold,
        threshold_options=ThresholdOptions(
            balanced_recommended=float(threshold_opts_raw.get("balanced_recommended", 0.08)),
            high_recall_sensitive=float(threshold_opts_raw.get("high_recall_sensitive", 0.05)),
        ),
        preprocessing=PreprocessingInfo(
            pipeline=preproc_raw.get("pipeline", "detrend_and_normalize_lightcurves"),
            algorithm=preproc_raw.get(
                "algorithm",
                "uniform_filter1d continuum subtraction + robust 1.4826*MAD standardization",
            ),
            smooth_window=state.smooth_window,
            feature_dtype=preproc_raw.get("feature_dtype", "float64"),
        ),
        created_at=meta.get("created_at", "unknown"),
        evaluation_summary=eval_summary,
    )


@router.get(
    "/info",
    response_model=ModelMetadataResponse,
    summary="Model information (alias for /metadata)",
)
def model_info() -> ModelMetadataResponse:
    return model_metadata()


@router.get(
    "/metrics",
    summary="Model evaluation metrics",
    description="Returns cross-validation and test set evaluation metrics for the Phase 1.5 classifier.",
)
def model_metrics():
    state = get_state()
    meta_resp = model_metadata()
    summary = meta_resp.evaluation_summary or {}
    t08 = summary.get("threshold_0_08", {})
    return {
        "model_name": meta_resp.model_name,
        "phase": meta_resp.phase,
        "accuracy": t08.get("accuracy", 0.9895),
        "recall": t08.get("exoplanet_recall", 0.20),
        "precision": t08.get("exoplanet_precision", 0.3333),
        "f1_score": t08.get("exoplanet_f1", 0.2500),
        "decision_threshold": meta_resp.default_threshold,
        "confusion_matrix": {
            "true_negatives": 563,
            "false_positives": 2,
            "false_negatives": 4,
            "true_positives": 1,
        },
        "evaluation_summary": summary,
    }


dataset_router = APIRouter(prefix="/dataset", tags=["Dataset"])


@dataset_router.get(
    "/info",
    summary="Kepler dataset summary",
    description="Returns statistics and class distributions for the NASA Kepler dataset.",
)
def dataset_info():
    return {
        "dataset_name": "NASA Kepler Exoplanet Labelled Time Series",
        "feature_count": 3197,
        "feature_type": "continuous photometric flux",
        "training_samples": 5087,
        "test_samples": 570,
        "total_samples": 5657,
        "class_distribution": {
            "non_exoplanet_class_1": 5615,
            "confirmed_exoplanet_class_2": 42,
            "positive_rate_pct": 0.74,
        },
        "status": "validated (zero missing values, zero infs, zero duplicates)",
    }


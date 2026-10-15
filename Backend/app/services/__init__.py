"""Services package for dataset processing and ML training."""

from .dataset_service import (
    inspect_dataset,
    validate_datasets,
    load_and_preprocess_dataset,
    detrend_and_normalize_lightcurves,
)
from .training_service import (
    train_random_forest,
    evaluate_model,
    evaluate_threshold_tradeoffs,
    compute_small_sample_uncertainty,
    save_artifacts,
    load_and_verify_artifacts,
)

from .network_service import (
    NetworkService,
    NetworkNode,
    NetworkLink,
    RouteResult,
    NodeNotFoundError,
    LinkNotFoundError,
    InvalidRouteRequestError,
    get_network_service,
)

__all__ = [
    "inspect_dataset",
    "validate_datasets",
    "load_and_preprocess_dataset",
    "detrend_and_normalize_lightcurves",
    "train_random_forest",
    "evaluate_model",
    "evaluate_threshold_tradeoffs",
    "compute_small_sample_uncertainty",
    "save_artifacts",
    "load_and_verify_artifacts",
    "NetworkService",
    "NetworkNode",
    "NetworkLink",
    "RouteResult",
    "NodeNotFoundError",
    "LinkNotFoundError",
    "InvalidRouteRequestError",
    "get_network_service",
]

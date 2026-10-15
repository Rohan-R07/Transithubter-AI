"""Pydantic schemas for request / response validation.

All external API shapes are defined here — never inline in the routers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ThresholdMode(str, Enum):
    """Named decision-threshold operating modes."""

    balanced = "balanced"         # T = 0.08  (default, selected by cross-validation)
    high_recall = "high_recall"   # T = 0.05  (catches more exoplanets, more false positives)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Prediction request payload.

    Attributes:
        flux_values: Exactly 3,197 numerical photometric flux values in the
            same column order as the training dataset (FLUX.1 … FLUX.3197).
        mode: Optional operating mode.  Defaults to ``balanced`` (T=0.08).
    """

    flux_values: List[float] = Field(
        ...,
        description="Exactly 3,197 numerical photometric flux values (FLUX.1 … FLUX.3197).",
        min_length=1,
    )
    mode: ThresholdMode = Field(
        default=ThresholdMode.balanced,
        description=(
            "'balanced' uses threshold 0.08 (CV-selected, fewer false positives). "
            "'high_recall' uses threshold 0.05 (catches more exoplanets at the cost "
            "of more false positives — ~1.9% false-positive rate on the Kepler test set)."
        ),
    )

    @field_validator("flux_values")
    @classmethod
    def validate_flux_values(cls, values: List[float]) -> List[float]:
        """Reject incorrect length, NaN, or Inf values before they reach the model."""
        # Length checked post-model-load to include the exact expected count in the error,
        # but we can catch obvious non-numeric sentinels here via pydantic's float coercion.
        import math

        bad_indices = [i for i, v in enumerate(values) if not math.isfinite(v)]
        if bad_indices:
            shown = bad_indices[:5]
            raise ValueError(
                f"flux_values contains {len(bad_indices)} non-finite value(s) "
                f"(NaN or Inf) at indices {shown}{'…' if len(bad_indices) > 5 else ''}. "
                "All flux values must be finite real numbers."
            )
        return values


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = Field(description="'ok' when the API is running.")
    model_loaded: bool = Field(description="True when the Phase 1.5 model is ready for inference.")
    model_name: Optional[str] = Field(default=None, description="Model identifier from metadata.")
    model_phase: Optional[str] = Field(default=None, description="Training phase description.")
    load_error: Optional[str] = Field(default=None, description="Non-null if model loading failed.")


class PreprocessingInfo(BaseModel):
    """Preprocessing pipeline description."""

    pipeline: str
    algorithm: str
    smooth_window: int
    feature_dtype: str


class ThresholdOptions(BaseModel):
    """Available decision thresholds."""

    balanced_recommended: float
    high_recall_sensitive: float


class ModelMetadataResponse(BaseModel):
    """GET /model/metadata response."""

    model_name: str
    phase: str
    model_type: str
    library: str
    library_version: str
    feature_count: int
    expected_input: str
    label_mapping: Dict[str, str]
    classes: List[int]
    default_threshold: float
    threshold_options: ThresholdOptions
    preprocessing: PreprocessingInfo
    created_at: str
    evaluation_summary: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    """POST /predict response."""

    predicted_class: int = Field(description="Predicted integer class label (1=non-exoplanet, 2=exoplanet).")
    predicted_label: str = Field(description="Human-readable class label.")
    is_exoplanet_candidate: bool = Field(description="True when the model predicts an exoplanet transit signal.")
    probability_exoplanet: float = Field(description="Model probability assigned to the exoplanet class (class 2).")
    probability_non_exoplanet: float = Field(description="Model probability assigned to the non-exoplanet class (class 1).")
    decision_threshold: float = Field(description="Threshold applied: predicted exoplanet when P(class 2) >= threshold.")
    mode: str = Field(description="Operating mode used ('balanced' or 'high_recall').")
    explanation: str = Field(description="Human-readable explanation of the prediction.")
    disclaimer: str = Field(
        default=(
            "A positive prediction reflects statistical similarity to confirmed Kepler transit "
            "flux patterns. It does not constitute physical proof that an exoplanet exists."
        )
    )


# ---------------------------------------------------------------------------
# Phase 3: Communication Network Simulation Schemas
# ---------------------------------------------------------------------------


class NetworkNodeSchema(BaseModel):
    """A node in the space communication network."""

    id: str = Field(description="Unique node identifier.")
    name: str = Field(description="Human-readable node label.")
    node_type: str = Field(description="Node category: ground_station, orbital_relay, deep_space_relay, observatory.")
    description: str = Field(description="Mission role and technical description.")
    location: Optional[str] = Field(default=None, description="Spatial coordinates or orbital location.")


class NetworkLinkSchema(BaseModel):
    """A communication link between two network nodes."""

    id: str = Field(description="Unique link identifier.")
    source: str = Field(description="Source node ID.")
    destination: str = Field(description="Destination node ID.")
    latency_ms: float = Field(description="One-way propagation latency in milliseconds (Dijkstra weight).")
    distance_km: float = Field(description="Physical link distance in kilometers.")
    available: bool = Field(default=True, description="True if link is operational and can carry traffic.")
    status: str = Field(default="operational", description="Link health status ('operational', 'offline (...)').")


class TopologyResponse(BaseModel):
    """GET /network/topology response."""

    nodes: List[NetworkNodeSchema] = Field(description="All space network nodes.")
    links: List[NetworkLinkSchema] = Field(description="All communication links.")
    total_nodes: int = Field(description="Total count of nodes in the network.")
    total_links: int = Field(description="Total count of communication links.")
    active_links: int = Field(description="Number of currently operational links.")
    disabled_links: int = Field(description="Number of disabled/interfered links.")


class RouteRequest(BaseModel):
    """POST /network/route request payload."""

    source: str = Field(
        ...,
        description="Source node ID or Name (e.g. 'earth_ground_station' or 'Earth Ground Station').",
        min_length=1,
    )
    destination: str = Field(
        ...,
        description="Destination node ID or Name (e.g. 'exoplanet_observatory' or 'Exoplanet Observatory').",
        min_length=1,
    )


class RouteResponse(BaseModel):
    """POST /network/route response."""

    reachable: bool = Field(description="True if an operational route exists between source and destination.")
    source: str = Field(description="Resolved source node name.")
    destination: str = Field(description="Resolved destination node name.")
    source_id: str = Field(description="Resolved source node ID.")
    destination_id: str = Field(description="Resolved destination node ID.")
    path: List[str] = Field(description="Ordered list of node names along the optimal path.")
    path_node_ids: List[str] = Field(description="Ordered list of node IDs along the optimal path.")
    links: List[NetworkLinkSchema] = Field(description="List of active links traversed along the route.")
    total_latency_ms: Optional[float] = Field(
        default=None,
        description="Cumulative latency in milliseconds (shortest path edge weight sum). Null if unreachable.",
    )
    total_distance_km: Optional[float] = Field(
        default=None,
        description="Cumulative physical distance in kilometers along the path. Null if unreachable.",
    )
    hops: int = Field(description="Number of link hops in the path (0 if unreachable or same node).")
    message: str = Field(description="Routing summary or unreachability diagnostic message.")


class InterferenceRequest(BaseModel):
    """POST /network/interference request payload."""

    link_id: Optional[str] = Field(
        default=None,
        description="Unique ID of the link to disable (e.g. 'link_earth_relay_a').",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source node ID or Name (used if link_id is omitted).",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Destination node ID or Name (used if link_id is omitted).",
    )
    reason: Optional[str] = Field(
        default="Solar flare electromagnetic interference",
        description="Simulated cause of link failure.",
    )


class InterferenceResponse(BaseModel):
    """POST /network/interference response."""

    success: bool = Field(description="True if the link was successfully disabled.")
    link: NetworkLinkSchema = Field(description="The updated link with available=False.")
    message: str = Field(description="Description of the interference event.")


class RestoreRequest(BaseModel):
    """POST /network/restore request payload."""

    link_id: Optional[str] = Field(
        default=None,
        description="Unique ID of the link to restore (e.g. 'link_earth_relay_a').",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source node ID or Name (used if link_id is omitted).",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Destination node ID or Name (used if link_id is omitted).",
    )
    restore_all: bool = Field(
        default=False,
        description="If True, restores all disabled links in the network to operational status.",
    )


class RestoreResponse(BaseModel):
    """POST /network/restore response."""

    success: bool = Field(description="True if link(s) were successfully restored.")
    restored_links: List[NetworkLinkSchema] = Field(description="List of links that were restored.")
    message: str = Field(description="Summary of restoration action.")


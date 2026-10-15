"""GET /health router."""

from fastapi import APIRouter
from app.core.model_loader import get_state
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description="Returns API status and whether the Phase 1.5 exoplanet detector is loaded and ready.",
)
def health() -> HealthResponse:
    state = get_state()
    return HealthResponse(
        status="ok",
        model_loaded=state.is_ready,
        model_name=state.metadata.get("model_name") if state.is_ready else None,
        model_phase=state.metadata.get("phase") if state.is_ready else None,
        load_error=state.load_error,
    )

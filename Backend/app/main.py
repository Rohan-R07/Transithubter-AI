"""TransitHunter AI — FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.model_loader import load_model
from app.routers import health, model, network, predict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS — configurable; safe defaults for local Vite/React dev
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = [
    "*",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:3000",
]



# ---------------------------------------------------------------------------
# Lifespan — model loads ONCE here, never retrained
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the Phase 1.5 model at startup; release nothing (stateless model)."""
    state = load_model()
    if state.is_ready:
        logger.info("TransitHunter AI ready — model loaded successfully.")
    else:
        logger.warning(
            "TransitHunter AI started WITHOUT a loaded model. "
            "Prediction endpoints will return 503. Error: %s",
            state.load_error,
        )
    yield
    logger.info("TransitHunter AI shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TransitHunter AI",
    description=(
        "Exoplanet transit detection API using a Random Forest classifier "
        "trained on NASA Kepler photometric flux data (Phase 1.5 model). "
        "Provide 3,197 flux values from a stellar light curve to receive "
        "a transit probability and exoplanet candidate classification."
    ),
    version="1.5.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------------------------------------------------------------------
# Global exception handler — never leaks stack traces or paths
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred. Please try again."},
    )


def _sanitize_for_json(obj: object) -> object:
    """Recursively replace non-JSON-serializable values so json.dumps never crashes.

    Handles:
    - float('inf'), float('nan') → string representations
    - Exception objects and other non-primitive types → str()
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # Catch-all for Exception objects, PydanticUndefined, etc.
    return str(obj)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with sanitised detail — prevents json.dumps crash on inf/nan/Exception objects."""
    detail = _sanitize_for_json(exc.errors())
    return JSONResponse(status_code=422, content={"detail": detail})


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(model.router)
app.include_router(model.dataset_router)
app.include_router(predict.router)
app.include_router(network.router)



# ---------------------------------------------------------------------------
# Root redirect to docs
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(
        content={
            "name": "TransitHunter AI API",
            "version": "1.5.0",
            "docs": "/docs",
            "health": "/health",
        }
    )

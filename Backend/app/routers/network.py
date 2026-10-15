"""Router for Phase 3: Communication Network Simulation & Dynamic Dijkstra Routing.

Endpoints:
- GET /network/topology: Current nodes and links.
- POST /network/route: Calculate optimal path with Dijkstra latency weights.
- POST /network/interference: Disable link(s) simulating space signal disruption.
- POST /network/restore: Restore disabled link(s).
- POST /network/reset: Reset network topology back to initial operational state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    InterferenceRequest,
    InterferenceResponse,
    RestoreRequest,
    RestoreResponse,
    RouteRequest,
    RouteResponse,
    TopologyResponse,
)
from app.services.network_service import (
    InvalidRouteRequestError,
    LinkNotFoundError,
    NodeNotFoundError,
    get_network_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/network", tags=["Communication Network"])


@router.get(
    "/topology",
    response_model=TopologyResponse,
    summary="Get network topology",
    description=(
        "Returns the complete simulated deep space communication network topology, "
        "including all ground stations, orbital relays, deep space relays, observatories, "
        "and communication links with their operational status and latencies."
    ),
)
def get_topology() -> Dict[str, Any]:
    svc = get_network_service()
    return svc.get_topology()


@router.post(
    "/route",
    response_model=RouteResponse,
    summary="Calculate optimal route with Dijkstra's algorithm",
    description=(
        "Computes the shortest communication path between source and destination nodes "
        "using Dijkstra's algorithm with one-way communication latency (ms) as the edge weight. "
        "Only operational, non-interfered links are used. If no path exists, returns a clear "
        "unreachable status."
    ),
    responses={
        404: {"description": "Source or destination node not recognized."},
        422: {"description": "Invalid input parameters."},
    },
)
def calculate_route(request: RouteRequest) -> Dict[str, Any]:
    svc = get_network_service()

    if not request.source or not request.source.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'source' cannot be empty.",
        )
    if not request.destination or not request.destination.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'destination' cannot be empty.",
        )

    try:
        route_result = svc.calculate_route(
            source_input=request.source,
            destination_input=request.destination,
        )
    except NodeNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except InvalidRouteRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return route_result.to_dict()


@router.post(
    "/interference",
    response_model=InterferenceResponse,
    summary="Simulate communication link interference",
    description=(
        "Disables a specified communication link to simulate solar storms, cosmic radiation, "
        "or orbital occultation. Subsequent route queries will dynamically route around this link."
    ),
    responses={
        404: {"description": "Specified link or node not found."},
        422: {"description": "Missing link identifier and endpoints."},
    },
)
def simulate_interference(request: InterferenceRequest) -> Dict[str, Any]:
    svc = get_network_service()

    if not request.link_id and not (request.source and request.destination):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Must provide either 'link_id' or both 'source' and 'destination' to apply interference.",
        )

    try:
        updated_link = svc.set_interference(
            link_id=request.link_id,
            source=request.source,
            destination=request.destination,
            reason=request.reason or "Signal interference",
        )
    except (LinkNotFoundError, NodeNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return {
        "success": True,
        "link": updated_link.to_dict(),
        "message": f"Link '{updated_link.id}' disabled due to {request.reason}.",
    }


@router.post(
    "/restore",
    response_model=RestoreResponse,
    summary="Restore communication link",
    description=(
        "Restores a disabled communication link back to operational status. "
        "Set 'restore_all=true' to restore all disabled links across the network."
    ),
    responses={
        404: {"description": "Specified link or node not found."},
        422: {"description": "Missing link identifier and endpoints."},
    },
)
def restore_link(request: RestoreRequest) -> Dict[str, Any]:
    svc = get_network_service()

    if not request.restore_all and not request.link_id and not (request.source and request.destination):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Must provide 'link_id', both 'source' and 'destination', or set 'restore_all=true'.",
        )

    try:
        restored = svc.restore_link(
            link_id=request.link_id,
            source=request.source,
            destination=request.destination,
            restore_all=request.restore_all,
        )
    except (LinkNotFoundError, NodeNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if request.restore_all:
        msg = f"Restored {len(restored)} communication link(s) to operational status."
    else:
        link_id = restored[0].id if restored else "unknown"
        msg = f"Link '{link_id}' restored to operational status."

    return {
        "success": True,
        "restored_links": [l.to_dict() for l in restored],
        "message": msg,
    }


@router.post(
    "/reset",
    response_model=TopologyResponse,
    summary="Reset network to default operational state",
    description="Resets all nodes and links back to the factory-default operational topology.",
)
def reset_network() -> Dict[str, Any]:
    svc = get_network_service()
    svc.reset()
    return svc.get_topology()

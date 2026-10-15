"""Network simulation service with Dynamic Dijkstra Routing for TransitHunter AI.

Represents space communication network nodes and communication links.
Provides deterministic Dijkstra shortest-path calculations using communication
latency as the edge weight, dynamic interference simulation, and link restoration.
"""

from __future__ import annotations

import heapq
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class NodeNotFoundError(ValueError):
    """Raised when a requested network node cannot be identified."""


class LinkNotFoundError(ValueError):
    """Raised when a requested communication link cannot be identified."""


class InvalidRouteRequestError(ValueError):
    """Raised when route inputs are invalid (e.g. empty or malformed)."""


@dataclass
class NetworkNode:
    """A node in the space communication network."""

    id: str
    name: str
    node_type: str
    description: str
    location: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "description": self.description,
            "location": self.location,
        }


@dataclass
class NetworkLink:
    """A bidirectional communication link between two network nodes."""

    id: str
    source: str
    destination: str
    latency_ms: float
    distance_km: float
    available: bool = True
    status: str = "operational"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "destination": self.destination,
            "latency_ms": self.latency_ms,
            "distance_km": self.distance_km,
            "available": self.available,
            "status": self.status,
        }

    def connects(self, node_a: str, node_b: str) -> bool:
        """Check if this link connects node_a and node_b in either direction."""
        return (self.source == node_a and self.destination == node_b) or (
            self.source == node_b and self.destination == node_a
        )


@dataclass
class RouteResult:
    """Result of a Dijkstra shortest-path calculation."""

    reachable: bool
    source_id: str
    source_name: str
    destination_id: str
    destination_name: str
    path_names: List[str]
    path_ids: List[str]
    links: List[NetworkLink]
    total_latency_ms: Optional[float]
    total_distance_km: Optional[float]
    hops: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reachable": self.reachable,
            "source": self.source_name,
            "destination": self.destination_name,
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "path": self.path_names,
            "path_node_ids": self.path_ids,
            "links": [link.to_dict() for link in self.links],
            "total_latency_ms": self.total_latency_ms,
            "total_distance_km": self.total_distance_km,
            "hops": self.hops,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Default Topology Definition
# ---------------------------------------------------------------------------

DEFAULT_NODES: List[NetworkNode] = [
    NetworkNode(
        id="earth_ground_station",
        name="Earth Ground Station",
        node_type="ground_station",
        description="Primary Earth mission operations center and deep space transceiver facility.",
        location="Canberra Deep Space Communication Complex, Earth",
    ),
    NetworkNode(
        id="orbital_relay_a",
        name="Orbital Relay A",
        node_type="orbital_relay",
        description="Primary geostationary laser/RF communications satellite.",
        location="GEO Sector Alpha (altitude ~35,786 km)",
    ),
    NetworkNode(
        id="orbital_relay_b",
        name="Orbital Relay B",
        node_type="orbital_relay",
        description="Secondary geostationary optical relay providing orbital redundancy.",
        location="GEO Sector Beta (altitude ~35,786 km)",
    ),
    NetworkNode(
        id="deep_space_relay",
        name="Deep Space Relay",
        node_type="deep_space_relay",
        description="Lagrange point L1 communications bridge and deep-space transceiver.",
        location="Sun-Earth Lagrange Point L1 (~1.5M km)",
    ),
    NetworkNode(
        id="exoplanet_observatory",
        name="Exoplanet Observatory",
        node_type="observatory",
        description="High-precision spaceborne transit photometer and exoplanet survey telescope.",
        location="Earth-Trailing Heliocentric Orbit",
    ),
]

DEFAULT_LINKS: List[NetworkLink] = [
    NetworkLink(
        id="link_earth_relay_a",
        source="earth_ground_station",
        destination="orbital_relay_a",
        latency_ms=25.0,
        distance_km=2000.0,
        available=True,
        status="operational",
    ),
    NetworkLink(
        id="link_earth_relay_b",
        source="earth_ground_station",
        destination="orbital_relay_b",
        latency_ms=35.0,
        distance_km=3500.0,
        available=True,
        status="operational",
    ),
    NetworkLink(
        id="link_relay_a_relay_b",
        source="orbital_relay_a",
        destination="orbital_relay_b",
        latency_ms=15.0,
        distance_km=5000.0,
        available=True,
        status="operational",
    ),
    NetworkLink(
        id="link_relay_a_deep_space",
        source="orbital_relay_a",
        destination="deep_space_relay",
        latency_ms=120.0,
        distance_km=500000.0,
        available=True,
        status="operational",
    ),
    NetworkLink(
        id="link_relay_b_deep_space",
        source="orbital_relay_b",
        destination="deep_space_relay",
        latency_ms=140.0,
        distance_km=600000.0,
        available=True,
        status="operational",
    ),
    NetworkLink(
        id="link_deep_space_observatory",
        source="deep_space_relay",
        destination="exoplanet_observatory",
        latency_ms=200.0,
        distance_km=1500000.0,
        available=True,
        status="operational",
    ),
]


class NetworkService:
    """Manages space network topology, link states, and Dijkstra routing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: Dict[str, NetworkNode] = {}
        self._links: Dict[str, NetworkLink] = {}
        self.reset()

    def reset(self) -> None:
        """Reset topology to the default operational state."""
        with self._lock:
            self._nodes = {n.id: deepcopy(n) for n in DEFAULT_NODES}
            self._links = {l.id: deepcopy(l) for l in DEFAULT_LINKS}

    # -----------------------------------------------------------------------
    # Node Lookup
    # -----------------------------------------------------------------------

    def resolve_node(self, identifier: str) -> NetworkNode:
        """Resolve a node by its ID or Name (case-insensitive)."""
        if not identifier or not isinstance(identifier, str):
            raise InvalidRouteRequestError("Node identifier cannot be empty.")

        clean = identifier.strip().lower()
        with self._lock:
            # 1. Exact ID match
            if clean in self._nodes:
                return self._nodes[clean]

            # 2. Match by normalized name
            for node in self._nodes.values():
                if node.name.strip().lower() == clean:
                    return node

            # 3. Match by fuzzy name without punctuation/underscores
            norm_clean = clean.replace("_", " ").replace("-", " ")
            for node in self._nodes.values():
                if node.name.strip().lower() == norm_clean:
                    return node

            valid_names = [f"'{n.name}' ({n.id})" for n in self._nodes.values()]
            raise NodeNotFoundError(
                f"Unknown node '{identifier}'. Valid network nodes: {', '.join(valid_names)}."
            )

    # -----------------------------------------------------------------------
    # Link Lookup
    # -----------------------------------------------------------------------

    def resolve_link(
        self,
        link_id: Optional[str] = None,
        source: Optional[str] = None,
        destination: Optional[str] = None,
    ) -> NetworkLink:
        """Resolve a link by either its unique link_id or source & destination nodes."""
        with self._lock:
            if link_id:
                clean_id = link_id.strip()
                if clean_id in self._links:
                    return self._links[clean_id]
                valid_ids = list(self._links.keys())
                raise LinkNotFoundError(
                    f"Link '{link_id}' not found in network. Available links: {valid_ids}."
                )

        # Fallback to source & destination resolution
        if source and destination:
            src_node = self.resolve_node(source)
            dst_node = self.resolve_node(destination)

            with self._lock:
                for link in self._links.values():
                    if link.connects(src_node.id, dst_node.id):
                        return link

                raise LinkNotFoundError(
                    f"No direct link exists between '{src_node.name}' and '{dst_node.name}'."
                )

        raise LinkNotFoundError(
            "Must provide either 'link_id' or both 'source' and 'destination' to locate a link."
        )

    # -----------------------------------------------------------------------
    # Topology Inspection
    # -----------------------------------------------------------------------

    def get_topology(self) -> Dict[str, Any]:
        """Return full network topology including all nodes and links."""
        with self._lock:
            nodes_list = [n.to_dict() for n in self._nodes.values()]
            links_list = [l.to_dict() for l in self._links.values()]
            active_count = sum(1 for l in self._links.values() if l.available)
            disabled_count = len(links_list) - active_count

            return {
                "nodes": nodes_list,
                "links": links_list,
                "total_nodes": len(nodes_list),
                "total_links": len(links_list),
                "active_links": active_count,
                "disabled_links": disabled_count,
            }

    # -----------------------------------------------------------------------
    # Dynamic Link Interference & Restoration
    # -----------------------------------------------------------------------

    def set_interference(
        self,
        link_id: Optional[str] = None,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        reason: str = "Signal interference / solar flare",
    ) -> NetworkLink:
        """Disable a link due to simulated interference."""
        link = self.resolve_link(link_id=link_id, source=source, destination=destination)
        with self._lock:
            link.available = False
            link.status = f"offline ({reason})"
            return deepcopy(link)

    def restore_link(
        self,
        link_id: Optional[str] = None,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        restore_all: bool = False,
    ) -> List[NetworkLink]:
        """Restore a disabled link or all links back to operational status."""
        with self._lock:
            if restore_all:
                restored = []
                for link in self._links.values():
                    if not link.available:
                        link.available = True
                        link.status = "operational"
                        restored.append(deepcopy(link))
                return restored

        link = self.resolve_link(link_id=link_id, source=source, destination=destination)
        with self._lock:
            link.available = True
            link.status = "operational"
            return [deepcopy(link)]

    # -----------------------------------------------------------------------
    # Dijkstra Shortest-Path Calculation
    # -----------------------------------------------------------------------

    def calculate_route(self, source_input: str, destination_input: str) -> RouteResult:
        """Calculate optimal communication route using Dijkstra's algorithm.

        Uses communication latency (latency_ms) as the edge weight and routes
        only over available links.
        """
        src = self.resolve_node(source_input)
        dst = self.resolve_node(destination_input)

        if src.id == dst.id:
            return RouteResult(
                reachable=True,
                source_id=src.id,
                source_name=src.name,
                destination_id=dst.id,
                destination_name=dst.name,
                path_names=[src.name],
                path_ids=[src.id],
                links=[],
                total_latency_ms=0.0,
                total_distance_km=0.0,
                hops=0,
                message=f"Source and destination are the same node ('{src.name}'). No transmission required.",
            )

        with self._lock:
            # Build adjacency list with currently available links
            adj: Dict[str, List[Tuple[str, NetworkLink]]] = {nid: [] for nid in self._nodes}
            for link in self._links.values():
                if link.available:
                    adj[link.source].append((link.destination, link))
                    adj[link.destination].append((link.source, link))

            # Dijkstra data structures
            distances: Dict[str, float] = {nid: float("inf") for nid in self._nodes}
            distances[src.id] = 0.0
            predecessors: Dict[str, Tuple[str, NetworkLink]] = {}  # v -> (u, link)

            # Priority queue entries: (cumulative_latency, node_id)
            pq: List[Tuple[float, str]] = [(0.0, src.id)]

            while pq:
                curr_dist, u = heapq.heappop(pq)

                if curr_dist > distances[u]:
                    continue

                if u == dst.id:
                    break

                for v, link in adj[u]:
                    weight = link.latency_ms
                    cand_dist = curr_dist + weight

                    if cand_dist < distances[v]:
                        distances[v] = cand_dist
                        predecessors[v] = (u, link)
                        heapq.heappush(pq, (cand_dist, v))

            # Check if destination was reached
            if distances[dst.id] == float("inf"):
                return RouteResult(
                    reachable=False,
                    source_id=src.id,
                    source_name=src.name,
                    destination_id=dst.id,
                    destination_name=dst.name,
                    path_names=[],
                    path_ids=[],
                    links=[],
                    total_latency_ms=None,
                    total_distance_km=None,
                    hops=0,
                    message=(
                        f"Destination '{dst.name}' is unreachable from '{src.name}'. "
                        "All available communication paths are disrupted or offline."
                    ),
                )

            # Reconstruct shortest path from predecessors
            path_ids: List[str] = [dst.id]
            path_links: List[NetworkLink] = []
            curr = dst.id

            while curr != src.id:
                prev_id, link = predecessors[curr]
                path_links.append(deepcopy(link))
                path_ids.append(prev_id)
                curr = prev_id

            path_ids.reverse()
            path_links.reverse()
            path_names = [self._nodes[nid].name for nid in path_ids]
            total_distance = sum(l.distance_km for l in path_links)
            total_latency = distances[dst.id]

            hops = len(path_links)
            path_str = " -> ".join(path_names)

            return RouteResult(
                reachable=True,
                source_id=src.id,
                source_name=src.name,
                destination_id=dst.id,
                destination_name=dst.name,
                path_names=path_names,
                path_ids=path_ids,
                links=path_links,
                total_latency_ms=round(total_latency, 2),
                total_distance_km=round(total_distance, 2),
                hops=hops,
                message=(
                    f"Optimal route found ({hops} hops, {total_latency:.1f} ms total latency): "
                    f"{path_str}"
                ),
            )


# Singleton instance for the application
_network_service_instance: Optional[NetworkService] = None


def get_network_service() -> NetworkService:
    """Retrieve the singleton NetworkService instance."""
    global _network_service_instance
    if _network_service_instance is None:
        _network_service_instance = NetworkService()
    return _network_service_instance

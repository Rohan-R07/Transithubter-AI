"""Tests for Phase 3 Communication Network Simulation and Dijkstra Routing."""

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.services.network_service import get_network_service


@pytest.fixture(autouse=True)
def reset_network_state():
    """Ensure every test runs against a clean, operational network topology."""
    svc = get_network_service()
    svc.reset()
    yield
    svc.reset()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Topology Tests
# ---------------------------------------------------------------------------


def test_topology_returns_all_nodes_and_links(client):
    """GET /network/topology returns all 5 space nodes and 6 communication links."""
    response = client.get("/network/topology")
    assert response.status_code == 200

    data = response.json()
    assert data["total_nodes"] == 5
    assert data["total_links"] == 6
    assert data["active_links"] == 6
    assert data["disabled_links"] == 0

    node_names = [n["name"] for n in data["nodes"]]
    expected_nodes = [
        "Earth Ground Station",
        "Orbital Relay A",
        "Orbital Relay B",
        "Deep Space Relay",
        "Exoplanet Observatory",
    ]
    for expected in expected_nodes:
        assert expected in node_names

    link_ids = [l["id"] for l in data["links"]]
    assert "link_earth_relay_a" in link_ids
    assert "link_deep_space_observatory" in link_ids


# ---------------------------------------------------------------------------
# Routing Tests (Dijkstra Shortest Path)
# ---------------------------------------------------------------------------


def test_initial_optimal_route(client):
    """Initial route from Earth to Observatory takes the fastest path (345 ms)."""
    response = client.post(
        "/network/route",
        json={
            "source": "Earth Ground Station",
            "destination": "Exoplanet Observatory",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["reachable"] is True
    assert data["path"] == [
        "Earth Ground Station",
        "Orbital Relay A",
        "Deep Space Relay",
        "Exoplanet Observatory",
    ]
    assert data["total_latency_ms"] == 345.0
    assert data["hops"] == 3
    assert len(data["links"]) == 3
    assert data["total_distance_km"] == 2002000.0


def test_route_using_node_ids(client):
    """Route calculation works seamlessly when passing node IDs instead of names."""
    response = client.post(
        "/network/route",
        json={
            "source": "earth_ground_station",
            "destination": "exoplanet_observatory",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["reachable"] is True
    assert data["source"] == "Earth Ground Station"
    assert data["destination"] == "Exoplanet Observatory"
    assert data["total_latency_ms"] == 345.0


def test_same_source_and_destination(client):
    """Routing from a node to itself yields 0 hops, 0 ms latency, and reachable=True."""
    response = client.post(
        "/network/route",
        json={
            "source": "Orbital Relay A",
            "destination": "Orbital Relay A",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["reachable"] is True
    assert data["hops"] == 0
    assert data["total_latency_ms"] == 0.0
    assert data["path"] == ["Orbital Relay A"]


# ---------------------------------------------------------------------------
# Interference & Dynamic Rerouting Tests
# ---------------------------------------------------------------------------


def test_dynamic_rerouting_after_interference(client):
    """When a primary link is disabled, Dijkstra recalculates an alternate route dynamically."""
    # 1. Disable Earth -> Relay A link
    int_resp = client.post(
        "/network/interference",
        json={
            "link_id": "link_earth_relay_a",
            "reason": "Coronal mass ejection",
        },
    )
    assert int_resp.status_code == 200
    assert int_resp.json()["link"]["available"] is False

    # 2. Query route again - should dynamically route via Orbital Relay B
    reroute_resp = client.post(
        "/network/route",
        json={
            "source": "Earth Ground Station",
            "destination": "Exoplanet Observatory",
        },
    )
    assert reroute_resp.status_code == 200
    reroute_data = reroute_resp.json()

    assert reroute_data["reachable"] is True
    # Does not traverse link_earth_relay_a
    traversed_link_ids = [l["id"] for l in reroute_data["links"]]
    assert "link_earth_relay_a" not in traversed_link_ids

    # Dijkstra finds optimal bypass through Orbital Relay B
    assert "Orbital Relay B" in reroute_data["path"]
    assert reroute_data["total_latency_ms"] == 370.0


def test_interference_by_endpoint_names(client):
    """Can disable a link by specifying source and destination node names."""
    resp = client.post(
        "/network/interference",
        json={
            "source": "Earth Ground Station",
            "destination": "Orbital Relay B",
            "reason": "Antenna pointing misalignment",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["link"]["id"] == "link_earth_relay_b"
    assert resp.json()["link"]["available"] is False


# ---------------------------------------------------------------------------
# Link Restoration Tests
# ---------------------------------------------------------------------------


def test_link_restoration_reverts_to_primary_path(client):
    """Restoring a disabled link enables Dijkstra to immediately return to the faster route."""
    # 1. Disable primary link
    client.post("/network/interference", json={"link_id": "link_earth_relay_a"})

    # 2. Restore primary link
    rest_resp = client.post("/network/restore", json={"link_id": "link_earth_relay_a"})
    assert rest_resp.status_code == 200
    assert rest_resp.json()["success"] is True

    # 3. Verify route immediately reverts to original optimal latency (345 ms)
    route_resp = client.post(
        "/network/route",
        json={
            "source": "Earth Ground Station",
            "destination": "Exoplanet Observatory",
        },
    )
    data = route_resp.json()
    assert data["path"] == [
        "Earth Ground Station",
        "Orbital Relay A",
        "Deep Space Relay",
        "Exoplanet Observatory",
    ]
    assert data["total_latency_ms"] == 345.0


def test_restore_all_links(client):
    """Restoring all links brings the whole network back to operational status."""
    # Disable two links
    client.post("/network/interference", json={"link_id": "link_earth_relay_a"})
    client.post("/network/interference", json={"link_id": "link_relay_b_deep_space"})

    # Check topology shows 2 disabled
    topo = client.get("/network/topology").json()
    assert topo["disabled_links"] == 2

    # Restore all
    rest_resp = client.post("/network/restore", json={"restore_all": True})
    assert rest_resp.status_code == 200
    assert len(rest_resp.json()["restored_links"]) == 2

    # Verify topology is fully active
    topo_after = client.get("/network/topology").json()
    assert topo_after["disabled_links"] == 0
    assert topo_after["active_links"] == 6


# ---------------------------------------------------------------------------
# Unreachable Destination Handling
# ---------------------------------------------------------------------------


def test_unreachable_destination(client):
    """If all links to the destination are disabled, response clearly states unreachable."""
    # Disable the only link connecting to Exoplanet Observatory
    client.post("/network/interference", json={"link_id": "link_deep_space_observatory"})

    response = client.post(
        "/network/route",
        json={
            "source": "Earth Ground Station",
            "destination": "Exoplanet Observatory",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["reachable"] is False
    assert data["path"] == []
    assert data["links"] == []
    assert data["hops"] == 0
    assert data["total_latency_ms"] is None
    assert data["total_distance_km"] is None
    assert "unreachable" in data["message"].lower()


# ---------------------------------------------------------------------------
# Input Validation & Error Handling
# ---------------------------------------------------------------------------


def test_unknown_source_node(client):
    """Route request with unknown source node returns 404 with helpful error."""
    response = client.post(
        "/network/route",
        json={"source": "Mars Base Alpha", "destination": "Exoplanet Observatory"},
    )
    assert response.status_code == 404
    assert "Unknown node 'Mars Base Alpha'" in response.json()["detail"]


def test_unknown_destination_node(client):
    """Route request with unknown destination node returns 404 with helpful error."""
    response = client.post(
        "/network/route",
        json={"source": "Earth Ground Station", "destination": "Andromeda Hub"},
    )
    assert response.status_code == 404
    assert "Unknown node 'Andromeda Hub'" in response.json()["detail"]


def test_empty_node_names(client):
    """Empty strings in route request return 422 validation error."""
    response = client.post(
        "/network/route",
        json={"source": "", "destination": "Exoplanet Observatory"},
    )
    assert response.status_code == 422


def test_unknown_link_id_interference(client):
    """Interference on non-existent link_id returns 404 with valid link choices."""
    response = client.post(
        "/network/interference",
        json={"link_id": "link_nonexistent"},
    )
    assert response.status_code == 404
    assert "Link 'link_nonexistent' not found" in response.json()["detail"]


def test_interference_missing_identifiers(client):
    """Interference without link_id or endpoints returns 422."""
    response = client.post("/network/interference", json={})
    assert response.status_code == 422
    assert "Must provide either 'link_id' or both" in response.json()["detail"]


def test_interference_no_direct_link(client):
    """Interference between two nodes with no direct link returns 404."""
    response = client.post(
        "/network/interference",
        json={
            "source": "Earth Ground Station",
            "destination": "Exoplanet Observatory",
        },
    )
    assert response.status_code == 404
    assert "No direct link exists" in response.json()["detail"]


def test_reset_network_endpoint(client):
    """POST /network/reset resets all links back to operational."""
    client.post("/network/interference", json={"link_id": "link_earth_relay_a"})
    client.post("/network/interference", json={"link_id": "link_earth_relay_b"})

    reset_resp = client.post("/network/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["disabled_links"] == 0
    assert reset_resp.json()["active_links"] == 6

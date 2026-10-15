#!/usr/bin/env python3
"""TransitHunter AI - Live API Endpoint Verification Script.

Tests all live FastAPI endpoints against a running server at http://127.0.0.1:8000.
Validates:
- GET  /health
- GET  /model/metadata
- POST /predict (balanced, high_recall, invalid feature count, NaN/Inf, non-numeric, invalid mode)
- GET  /network/topology
- POST /network/route (initial optimal path, same node, by node IDs)
- POST /network/interference (disabling links, dynamic Dijkstra rerouting)
- POST /network/restore (single link restoration, restore all)
- POST /network/reset (reset topology)
- Unreachable destinations handling
- 4xx client error validation (unknown nodes, unknown links, malformed inputs)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

# Default server settings
DEFAULT_BASE_URL: str = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SEC: float = 10.0


@dataclass
class TestResult:
    """Outcome of an individual endpoint test."""

    name: str
    passed: bool
    status_code: int
    duration_ms: float
    error_message: Optional[str] = None
    endpoint: Optional[str] = None


class LiveApiVerifier:
    """Verifies all endpoints on the live running TransitHunter AI server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        verbose: bool = False,
        request_json_path: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

        # Locate predict_request.json
        self.predict_request_file = self._resolve_predict_request(request_json_path)

    def _resolve_predict_request(self, explicit_path: Optional[str]) -> Path:
        """Locate predict_request.json or attempt to auto-generate it."""
        if explicit_path:
            p = Path(explicit_path).resolve()
            if p.is_file():
                return p
            raise FileNotFoundError(f"Specified predict request JSON not found: {p}")

        script_dir = Path(__file__).resolve().parent
        backend_dir = script_dir.parent
        workspace_root = backend_dir.parent

        candidates = [
            backend_dir / "predict_request.json",
            workspace_root / "predict_request.json",
            Path.cwd() / "predict_request.json",
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()

        # If missing, try running generate_predict_request
        gen_script = script_dir / "generate_predict_request.py"
        if gen_script.is_file():
            try:
                import subprocess

                out_target = backend_dir / "predict_request.json"
                subprocess.run(
                    [sys.executable, str(gen_script), "--output", str(out_target)],
                    check=True,
                    capture_output=True,
                )
                if out_target.is_file():
                    return out_target
            except Exception as exc:
                print(f"[WARN] Could not auto-generate predict_request.json: {exc}")

        return backend_dir / "predict_request.json"

    def check_server_online(self) -> bool:
        """Check if server is reachable before executing the test suite."""
        try:
            r = self.client.get("/health", timeout=3.0)
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return False

    def record_test(
        self,
        name: str,
        endpoint: str,
        test_fn: Callable[[], Tuple[int, Optional[str]]],
    ) -> None:
        """Execute a test function and record result with execution time."""
        t0 = time.perf_counter()
        status_code = 0
        error_msg = None
        passed = False

        try:
            status_code, error_msg = test_fn()
            passed = error_msg is None
        except Exception as exc:
            error_msg = f"Unexpected exception: {type(exc).__name__}: {exc}"
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0

        res = TestResult(
            name=name,
            passed=passed,
            status_code=status_code,
            duration_ms=duration_ms,
            error_message=error_msg,
            endpoint=endpoint,
        )
        self.results.append(res)

        badge = "[PASS]" if passed else "[FAIL]"
        print(
            f"  {badge} {name:<46} "
            f"HTTP {status_code:<3} ({duration_ms:5.1f} ms)"
        )
        if not passed and error_msg:
            print(f"         |-- Error: {error_msg}")

    # -----------------------------------------------------------------------
    # Test Implementations: Health & Model Metadata
    # -----------------------------------------------------------------------

    def verify_health(self) -> Tuple[int, Optional[str]]:
        resp = self.client.get("/health")
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        if data.get("status") != "ok":
            return resp.status_code, f"Expected status='ok', got {data.get('status')}"
        if data.get("model_loaded") is not True:
            return resp.status_code, f"Expected model_loaded=True, got {data.get('model_loaded')}"
        if not data.get("model_name"):
            return resp.status_code, "Missing 'model_name' in health response"
        return resp.status_code, None

    def verify_model_metadata(self) -> Tuple[int, Optional[str]]:
        resp = self.client.get("/model/metadata")
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        if data.get("feature_count") != 3197:
            return resp.status_code, f"Expected feature_count=3197, got {data.get('feature_count')}"
        if data.get("classes") != [1, 2]:
            return resp.status_code, f"Expected classes=[1, 2], got {data.get('classes')}"
        thresholds = data.get("threshold_options", {})
        if thresholds.get("balanced_recommended") != 0.08:
            return resp.status_code, f"Expected balanced_recommended=0.08, got {thresholds.get('balanced_recommended')}"
        if thresholds.get("high_recall_sensitive") != 0.05:
            return resp.status_code, f"Expected high_recall_sensitive=0.05, got {thresholds.get('high_recall_sensitive')}"
        pre = data.get("preprocessing", {})
        if pre.get("smooth_window") != 31:
            return resp.status_code, f"Expected smooth_window=31, got {pre.get('smooth_window')}"
        return resp.status_code, None

    # -----------------------------------------------------------------------
    # Test Implementations: Prediction (POST /predict)
    # -----------------------------------------------------------------------

    def verify_predict_balanced(self) -> Tuple[int, Optional[str]]:
        if not self.predict_request_file.is_file():
            return 0, f"Request file not found at {self.predict_request_file}"
        with open(self.predict_request_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["mode"] = "balanced"
        resp = self.client.post("/predict", json=payload)
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        req_keys = [
            "predicted_class",
            "predicted_label",
            "is_exoplanet_candidate",
            "probability_exoplanet",
            "probability_non_exoplanet",
            "decision_threshold",
            "mode",
            "explanation",
            "disclaimer",
        ]
        for k in req_keys:
            if k not in data:
                return resp.status_code, f"Missing required response field '{k}'"

        if data["predicted_class"] not in (1, 2):
            return resp.status_code, f"Invalid predicted_class: {data['predicted_class']}"
        if data["decision_threshold"] != 0.08:
            return resp.status_code, f"Expected threshold=0.08, got {data['decision_threshold']}"
        if data["mode"] != "balanced":
            return resp.status_code, f"Expected mode='balanced', got {data['mode']}"

        p_sum = data["probability_exoplanet"] + data["probability_non_exoplanet"]
        if not (0.98 <= p_sum <= 1.02):
            return resp.status_code, f"Probabilities do not sum to ~1.0: {p_sum}"

        return resp.status_code, None

    def verify_predict_high_recall(self) -> Tuple[int, Optional[str]]:
        if not self.predict_request_file.is_file():
            return 0, f"Request file not found at {self.predict_request_file}"
        with open(self.predict_request_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["mode"] = "high_recall"
        resp = self.client.post("/predict", json=payload)
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        if data["decision_threshold"] != 0.05:
            return resp.status_code, f"Expected threshold=0.05, got {data['decision_threshold']}"
        if data["mode"] != "high_recall":
            return resp.status_code, f"Expected mode='high_recall', got {data['mode']}"
        return resp.status_code, None

    def verify_predict_invalid_feature_count(self) -> Tuple[int, Optional[str]]:
        bad_payload = {"flux_values": [1.0, 2.0, 3.0], "mode": "balanced"}
        resp = self.client.post("/predict", json=bad_payload)
        if resp.status_code != 422:
            return resp.status_code, f"Expected 422 for feature count mismatch, got {resp.status_code}"
        return resp.status_code, None

    def verify_predict_invalid_nan_values(self) -> Tuple[int, Optional[str]]:
        good_part = json.dumps([0.5] * 3196)[1:-1]
        raw = f'{{"flux_values": [NaN, {good_part}]}}'
        resp = self.client.post(
            "/predict",
            content=raw.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 422:
            return resp.status_code, f"Expected 422 for NaN values, got {resp.status_code}"
        return resp.status_code, None

    def verify_predict_non_numeric(self) -> Tuple[int, Optional[str]]:
        bad_payload = {"flux_values": ["not_a_number"] + [0.5] * 3196, "mode": "balanced"}
        resp = self.client.post("/predict", json=bad_payload)
        if resp.status_code != 422:
            return resp.status_code, f"Expected 422 for non-numeric flux, got {resp.status_code}"
        return resp.status_code, None

    def verify_predict_invalid_mode(self) -> Tuple[int, Optional[str]]:
        bad_payload = {"flux_values": [1.0] * 3197, "mode": "ultra_sensitive"}
        resp = self.client.post("/predict", json=bad_payload)
        if resp.status_code != 422:
            return resp.status_code, f"Expected 422 for invalid mode, got {resp.status_code}"
        return resp.status_code, None

    # -----------------------------------------------------------------------
    # Test Implementations: Network Topology & Dijkstra Routing
    # -----------------------------------------------------------------------

    def verify_network_topology(self) -> Tuple[int, Optional[str]]:
        resp = self.client.get("/network/topology")
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        if data.get("total_nodes") != 5:
            return resp.status_code, f"Expected 5 nodes, got {data.get('total_nodes')}"
        if data.get("total_links") != 6:
            return resp.status_code, f"Expected 6 links, got {data.get('total_links')}"
        if data.get("disabled_links") != 0:
            return resp.status_code, f"Expected 0 disabled links, got {data.get('disabled_links')}"
        return resp.status_code, None

    def verify_network_initial_route(self) -> Tuple[int, Optional[str]]:
        payload = {"source": "Earth Ground Station", "destination": "Exoplanet Observatory"}
        resp = self.client.post("/network/route", json=payload)
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        if data.get("reachable") is not True:
            return resp.status_code, f"Expected reachable=True, got {data.get('reachable')}"
        expected_path = [
            "Earth Ground Station",
            "Orbital Relay A",
            "Deep Space Relay",
            "Exoplanet Observatory",
        ]
        if data.get("path") != expected_path:
            return resp.status_code, f"Expected path {expected_path}, got {data.get('path')}"
        if data.get("total_latency_ms") != 345.0:
            return resp.status_code, f"Expected latency=345.0 ms, got {data.get('total_latency_ms')}"
        if data.get("hops") != 3:
            return resp.status_code, f"Expected hops=3, got {data.get('hops')}"
        return resp.status_code, None

    def verify_network_same_node_route(self) -> Tuple[int, Optional[str]]:
        payload = {"source": "Earth Ground Station", "destination": "Earth Ground Station"}
        resp = self.client.post("/network/route", json=payload)
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        if data.get("reachable") is not True or data.get("hops") != 0 or data.get("total_latency_ms") != 0.0:
            return resp.status_code, f"Unexpected same-node result: {data}"
        return resp.status_code, None

    def verify_network_route_by_node_ids(self) -> Tuple[int, Optional[str]]:
        payload = {"source": "earth_ground_station", "destination": "exoplanet_observatory"}
        resp = self.client.post("/network/route", json=payload)
        if resp.status_code != 200:
            return resp.status_code, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        if data.get("total_latency_ms") != 345.0:
            return resp.status_code, f"Expected latency=345.0, got {data.get('total_latency_ms')}"
        return resp.status_code, None

    def verify_network_interference_reroute(self) -> Tuple[int, Optional[str]]:
        # 1. Disable link_earth_relay_a
        int_resp = self.client.post(
            "/network/interference",
            json={"link_id": "link_earth_relay_a", "reason": "Solar flare radiation"},
        )
        if int_resp.status_code != 200:
            return int_resp.status_code, f"Interference failed with {int_resp.status_code}"
        if int_resp.json()["link"]["available"] is not False:
            return int_resp.status_code, "Link availability did not update to False"

        # 2. Query route - should dynamically reroute via Orbital Relay B
        route_resp = self.client.post(
            "/network/route",
            json={"source": "Earth Ground Station", "destination": "Exoplanet Observatory"},
        )
        if route_resp.status_code != 200:
            return route_resp.status_code, f"Reroute query failed with {route_resp.status_code}"
        data = route_resp.json()
        if not data.get("reachable"):
            return route_resp.status_code, "Path became unreachable instead of rerouting"

        # Ensure disabled link is not used
        traversed_ids = [l["id"] for l in data.get("links", [])]
        if "link_earth_relay_a" in traversed_ids:
            return route_resp.status_code, "Reroute still traversed disabled link_earth_relay_a"

        # Dijkstra selects lowest latency bypass: Earth -> Relay B -> Relay A -> Deep Space -> Observatory (370.0 ms)
        if data.get("total_latency_ms") != 370.0:
            return route_resp.status_code, f"Expected recalculated latency=370.0 ms, got {data.get('total_latency_ms')}"

        return route_resp.status_code, None

    def verify_network_restore_and_reversion(self) -> Tuple[int, Optional[str]]:
        # 1. Restore link_earth_relay_a
        rest_resp = self.client.post(
            "/network/restore",
            json={"link_id": "link_earth_relay_a"},
        )
        if rest_resp.status_code != 200:
            return rest_resp.status_code, f"Restore failed with {rest_resp.status_code}"

        # 2. Verify route reverts to optimal 345.0 ms
        route_resp = self.client.post(
            "/network/route",
            json={"source": "Earth Ground Station", "destination": "Exoplanet Observatory"},
        )
        data = route_resp.json()
        if data.get("total_latency_ms") != 345.0:
            return route_resp.status_code, f"Expected reverted latency=345.0 ms, got {data.get('total_latency_ms')}"
        return route_resp.status_code, None

    def verify_network_unreachable_destination(self) -> Tuple[int, Optional[str]]:
        # Disable the only link connecting to Exoplanet Observatory
        self.client.post(
            "/network/interference",
            json={"link_id": "link_deep_space_observatory"},
        )

        route_resp = self.client.post(
            "/network/route",
            json={"source": "Earth Ground Station", "destination": "Exoplanet Observatory"},
        )
        data = route_resp.json()
        if data.get("reachable") is not False:
            return route_resp.status_code, "Expected reachable=False for severed destination"
        if data.get("path") != []:
            return route_resp.status_code, f"Expected empty path, got {data.get('path')}"
        if data.get("hops") != 0:
            return route_resp.status_code, f"Expected hops=0, got {data.get('hops')}"
        if "unreachable" not in data.get("message", "").lower():
            return route_resp.status_code, f"Message should explain unreachability: {data.get('message')}"

        # Clean up
        self.client.post("/network/restore", json={"link_id": "link_deep_space_observatory"})
        return route_resp.status_code, None

    def verify_network_restore_all(self) -> Tuple[int, Optional[str]]:
        # Disable multiple links
        self.client.post("/network/interference", json={"link_id": "link_earth_relay_a"})
        self.client.post("/network/interference", json={"link_id": "link_earth_relay_b"})

        # Restore all
        rest_resp = self.client.post("/network/restore", json={"restore_all": True})
        if rest_resp.status_code != 200:
            return rest_resp.status_code, f"Restore all failed with {rest_resp.status_code}"

        # Check topology is fully restored
        topo_resp = self.client.get("/network/topology")
        data = topo_resp.json()
        if data.get("disabled_links") != 0 or data.get("active_links") != 6:
            return topo_resp.status_code, f"Topology not fully active after restore_all: {data}"
        return rest_resp.status_code, None

    def verify_network_reset_endpoint(self) -> Tuple[int, Optional[str]]:
        resp = self.client.post("/network/reset")
        if resp.status_code != 200:
            return resp.status_code, f"Reset failed with {resp.status_code}"
        data = resp.json()
        if data.get("disabled_links") != 0 or data.get("active_links") != 6:
            return resp.status_code, f"Topology not default after reset: {data}"
        return resp.status_code, None

    # -----------------------------------------------------------------------
    # Test Implementations: 4xx Client Error Handling
    # -----------------------------------------------------------------------

    def verify_error_unknown_node(self) -> Tuple[int, Optional[str]]:
        resp = self.client.post(
            "/network/route",
            json={"source": "Mars Outpost 9", "destination": "Exoplanet Observatory"},
        )
        if resp.status_code != 404:
            return resp.status_code, f"Expected 404 for unknown node, got {resp.status_code}"
        if "Unknown node 'Mars Outpost 9'" not in resp.json().get("detail", ""):
            return resp.status_code, f"Detail message did not identify unknown node: {resp.text}"
        return resp.status_code, None

    def verify_error_empty_node_names(self) -> Tuple[int, Optional[str]]:
        resp = self.client.post(
            "/network/route",
            json={"source": "", "destination": "Exoplanet Observatory"},
        )
        if resp.status_code != 422:
            return resp.status_code, f"Expected 422 for empty source, got {resp.status_code}"
        return resp.status_code, None

    def verify_error_unknown_link_id(self) -> Tuple[int, Optional[str]]:
        resp = self.client.post(
            "/network/interference",
            json={"link_id": "link_alien_corridor"},
        )
        if resp.status_code != 404:
            return resp.status_code, f"Expected 404 for unknown link, got {resp.status_code}"
        return resp.status_code, None

    def verify_error_interference_missing_identifiers(self) -> Tuple[int, Optional[str]]:
        resp = self.client.post("/network/interference", json={})
        if resp.status_code != 422:
            return resp.status_code, f"Expected 422 for missing link identifiers, got {resp.status_code}"
        return resp.status_code, None

    def verify_error_interference_no_direct_link(self) -> Tuple[int, Optional[str]]:
        resp = self.client.post(
            "/network/interference",
            json={"source": "Earth Ground Station", "destination": "Exoplanet Observatory"},
        )
        if resp.status_code != 404:
            return resp.status_code, f"Expected 404 for non-existent direct link, got {resp.status_code}"
        return resp.status_code, None

    # -----------------------------------------------------------------------
    # Test Runner
    # -----------------------------------------------------------------------

    def run_all(self) -> int:
        """Run all test suites against the live API and output report."""
        print("=" * 75)
        print(" TransitHunter AI - Live API Endpoint Verification")
        print(f" Target Server: {self.base_url}")
        print("=" * 75)

        # 1. Pre-flight connectivity check
        print("\n[Step 1/4] Checking server connectivity...")
        if not self.check_server_online():
            print(
                f"\n[ERROR] Could not connect to TransitHunter AI API at {self.base_url}.\n"
                "The FastAPI server is not responding.\n\n"
                "To start the server, open a terminal and run:\n"
                "  cd Backend\n"
                "  uvicorn app.main:app --reload --host 127.0.0.1 --port 8000\n"
            )
            return 2

        print(f"  [OK] Server is online and responding at {self.base_url}")

        # 2. System & Model Inspection Endpoints
        print("\n[Step 2/4] Testing System Health & Model Metadata Endpoints...")
        self.record_test("GET /health returns healthy & loaded", "/health", self.verify_health)
        self.record_test("GET /model/metadata schema & parameters", "/model/metadata", self.verify_model_metadata)

        # 3. Prediction Endpoints
        print("\n[Step 3/4] Testing Model Inference (POST /predict)...")
        self.record_test("POST /predict (balanced mode, real flux)", "/predict", self.verify_predict_balanced)
        self.record_test("POST /predict (high_recall mode, T=0.05)", "/predict", self.verify_predict_high_recall)
        self.record_test("POST /predict validation: wrong feature count", "/predict", self.verify_predict_invalid_feature_count)
        self.record_test("POST /predict validation: NaN/Inf sentinels", "/predict", self.verify_predict_invalid_nan_values)
        self.record_test("POST /predict validation: non-numeric string", "/predict", self.verify_predict_non_numeric)
        self.record_test("POST /predict validation: invalid mode", "/predict", self.verify_predict_invalid_mode)

        # 4. Network Simulation & Routing Endpoints
        print("\n[Step 4/4] Testing Communication Network & Dijkstra Routing...")
        try:
            # Topology & Initial Routing
            self.record_test("GET /network/topology active links & nodes", "/network/topology", self.verify_network_topology)
            self.record_test("POST /network/route initial optimal path (345ms)", "/network/route", self.verify_network_initial_route)
            self.record_test("POST /network/route same-node (0 hops, 0ms)", "/network/route", self.verify_network_same_node_route)
            self.record_test("POST /network/route resolution by node IDs", "/network/route", self.verify_network_route_by_node_ids)

            # Interference, Dynamic Dijkstra Rerouting & Restoration
            self.record_test("POST /network/interference dynamic reroute", "/network/interference", self.verify_network_interference_reroute)
            self.record_test("POST /network/restore reverts to primary path", "/network/restore", self.verify_network_restore_and_reversion)
            self.record_test("POST /network/route unreachable destination", "/network/route", self.verify_network_unreachable_destination)
            self.record_test("POST /network/restore restore_all=true", "/network/restore", self.verify_network_restore_all)
            self.record_test("POST /network/reset restores factory state", "/network/reset", self.verify_network_reset_endpoint)

            # Client Error Validation (4xx)
            self.record_test("Validation: unknown source node (404)", "/network/route", self.verify_error_unknown_node)
            self.record_test("Validation: empty node parameter (422)", "/network/route", self.verify_error_empty_node_names)
            self.record_test("Validation: unknown link ID (404)", "/network/interference", self.verify_error_unknown_link_id)
            self.record_test("Validation: missing link info (422)", "/network/interference", self.verify_error_interference_missing_identifiers)
            self.record_test("Validation: no direct link (404)", "/network/interference", self.verify_error_interference_no_direct_link)
        finally:
            # Defensive teardown: ALWAYS ensure network topology is reset back to operational
            try:
                self.client.post("/network/reset", timeout=3.0)
            except Exception:
                pass

        # Final Summary Report
        total = len(self.results)
        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = total - passed_count

        print("\n" + "=" * 75)
        print(" Verification Summary Report")
        print("=" * 75)
        print(f" Total Endpoints & Scenarios Tested: {total}")
        print(f" Passed: {passed_count}")
        print(f" Failed: {failed_count}")

        if failed_count == 0:
            print("\n [SUCCESS] All live API endpoints verified and operational!")
            return 0
        else:
            print("\n [FAILURE] One or more live endpoint tests failed.")
            print(" Failed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name} ({r.endpoint}): {r.error_message}")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify TransitHunter AI live FastAPI endpoints."
    )
    parser.add_argument(
        "--base-url",
        "-u",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Target FastAPI server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--request-json",
        "-r",
        type=str,
        default=None,
        help="Path to predict_request.json for testing POST /predict",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose diagnostics",
    )

    args = parser.parse_args()

    verifier = LiveApiVerifier(
        base_url=args.base_url,
        timeout=args.timeout,
        verbose=args.verbose,
        request_json_path=args.request_json,
    )

    return verifier.run_all()


if __name__ == "__main__":
    sys.exit(main())

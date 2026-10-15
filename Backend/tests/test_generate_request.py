"""Tests for generate_predict_request utility."""

import json
from pathlib import Path

import pytest
from app.schemas import PredictRequest
from scripts.generate_predict_request import (
    EXPECTED_FEATURE_COUNT,
    build_predict_request,
    extract_flux_sample,
    find_dataset,
    generate_predict_request_file,
)


def test_find_dataset():
    csv_path = find_dataset("exoTest.csv")
    assert csv_path.is_file()
    assert csv_path.name == "exoTest.csv"


def test_extract_flux_sample():
    csv_path = find_dataset("exoTest.csv")
    flux_values, true_label, selected_idx = extract_flux_sample(csv_path, row_index=0)
    assert len(flux_values) == EXPECTED_FEATURE_COUNT
    assert true_label == 2  # First row in Kepler exoTest.csv is exoplanet class
    assert selected_idx == 0
    assert all(isinstance(v, float) for v in flux_values)


def test_build_predict_request():
    mock_fluxes = [1.0] * EXPECTED_FEATURE_COUNT
    payload = build_predict_request(mock_fluxes, mode="balanced")
    assert len(payload["flux_values"]) == EXPECTED_FEATURE_COUNT
    assert payload["mode"] == "balanced"

    # Schema validation
    req = PredictRequest(**payload)
    assert req.mode.value == "balanced"


def test_generate_predict_request_file(tmp_path):
    out_file = tmp_path / "test_req.json"
    generated_path = generate_predict_request_file(output_path=str(out_file), row_index=0)
    assert generated_path.is_file()

    with open(generated_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "flux_values" in data
    assert len(data["flux_values"]) == EXPECTED_FEATURE_COUNT
    assert data["mode"] == "balanced"

"""
Unit & Integration Tests for NEO Guardian
Verifies:
1. Model artifact integrity and feature schema
2. Astronomical physics formulas (IAU Diameter)
3. FastAPI endpoints (/health, /model-info, /predict)
4. Extreme edge case handling and data validation
"""
import pytest
import numpy as np
import joblib
import os
from fastapi.testclient import TestClient
from api import app, get_model

client = TestClient(app)

# ----------------- 1. MODEL ARTIFACT TESTS -----------------
def test_model_loading_and_features():
    """Verify that asteroid_guardian_v2.pkl exists and has the 6 clean features."""
    model = get_model()
    assert model is not None, "Model failed to load."
    expected_features = [
        "Absolute Magnitude",
        "Orbit Uncertainity",
        "Relative Velocity km per sec",
        "Eccentricity",
        "Miss Dist.(Astronomical)",
        "Minimum Orbit Intersection"
    ]
    assert list(model.feature_names_in_) == expected_features, "Model features do not match expected clean schema."
    assert len(model.classes_) == 2, "Binary classification expected."

# ----------------- 2. PHYSICS & FORMULA TESTS -----------------
def test_iau_diameter_calculation():
    """
    Test IAU Diameter formula: D(km) = (1329 / sqrt(albedo)) * 10^(-0.2 * H)
    For H = 18.0 and albedo = 0.15:
    D(km) = (1329 / 0.387298) * 10^(-3.6) = 3431.467 * 0.000251188 = ~0.8619 km = ~861.9 meters
    """
    albedo = 0.15
    H = 18.0
    diam_km = (1329 / np.sqrt(albedo)) * (10 ** (-0.2 * H))
    diam_meters = diam_km * 1000
    assert 850.0 < diam_meters < 870.0, f"Expected diameter near 862m, got {diam_meters}m"

# ----------------- 3. API ENDPOINT TESTS -----------------
def test_api_health_endpoint():
    """Verify /health returns 200 with online status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert len(data["expected_features"]) == 6

def test_api_model_info_endpoint():
    """Verify /model-info returns valid feature importances summing to ~100%."""
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["n_features"] == 6
    assert "Minimum Orbit Intersection" in data["feature_importances_pct"]
    total_pct = sum(data["feature_importances_pct"].values())
    assert 99.0 <= total_pct <= 101.0

def test_api_prediction_hazardous_scenario():
    """High-risk asteroid: low magnitude (large), low MOID, close miss distance."""
    payload = {
        "absolute_magnitude": 18.0,
        "orbit_uncertainty": 1,
        "relative_velocity": 24.5,
        "eccentricity": 0.65,
        "miss_distance": 0.015,
        "minimum_orbit_intersection": 0.004
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_hazardous"] is True
    assert data["status"] == "HAZARDOUS OBJECT"
    assert data["hazard_probability"] >= 0.80
    assert data["estimated_diameter_meters"] > 500

def test_api_prediction_safe_scenario():
    """Nominal asteroid: high magnitude (tiny rock), large MOID, large miss distance."""
    payload = {
        "absolute_magnitude": 27.5,
        "orbit_uncertainty": 8,
        "relative_velocity": 8.0,
        "eccentricity": 0.15,
        "miss_distance": 0.45,
        "minimum_orbit_intersection": 0.25
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_hazardous"] is False
    assert data["status"] == "SECURE / NOMINAL"
    assert data["hazard_probability"] < 0.15
    assert data["estimated_diameter_meters"] < 25
    assert "Meteoroid" in data["threat_classification"]

def test_api_validation_error_on_invalid_input():
    """Pydantic must catch negative velocity or out-of-range magnitude."""
    bad_payload = {
        "absolute_magnitude": -5.0,  # Invalid: ge=5.0
        "orbit_uncertainty": 15,     # Invalid: le=9
        "relative_velocity": -10.0,  # Invalid: ge=0.0
        "eccentricity": 1.5,         # Invalid: le=1.0
        "miss_distance": -1.0,
        "minimum_orbit_intersection": 0.01
    }
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422, "API should return HTTP 422 Unprocessable Entity for invalid physics input."

"""
NEO Guardian - REST API Backend (FastAPI)
Decouples Machine Learning inference and physics calculations from UI presentation.
"""
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

MODEL_PATH = os.path.join(os.path.dirname(__file__), "asteroid_guardian_v2.pkl")

from contextlib import asynccontextmanager

# Load model globally
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Please ensure asteroid_guardian_v2.pkl is present.")
        model = joblib.load(MODEL_PATH)
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

# Initialize FastAPI application
app = FastAPI(
    title="NEO Guardian - Planetary Defense API",
    description="High-reliability REST API for Near-Earth Object (NEO) hazard classification and orbital trajectory assessment.",
    version="2.0.0",
    lifespan=lifespan
)

# Pydantic Schemas for Request & Response
class AsteroidScanRequest(BaseModel):
    absolute_magnitude: float = Field(
        ..., ge=5.0, le=35.0,
        description="Absolute Magnitude (H). Lower H means a brighter and larger asteroid.",
        json_schema_extra={"example": 22.0}
    )
    orbit_uncertainty: int = Field(
        ..., ge=0, le=9,
        description="Orbit Uncertainty (U) from 0 (certain orbit) to 9 (highly uncertain).",
        json_schema_extra={"example": 5}
    )
    relative_velocity: float = Field(
        ..., ge=0.0, le=100.0,
        description="Relative velocity in km/s.",
        json_schema_extra={"example": 15.0}
    )
    eccentricity: float = Field(
        ..., ge=0.0, le=1.0,
        description="Orbital eccentricity (0 is circular, ~0.9 is elongated comet-like).",
        json_schema_extra={"example": 0.5}
    )
    miss_distance: float = Field(
        ..., ge=0.0, le=10.0,
        description="Miss distance in Astronomical Units (AU).",
        json_schema_extra={"example": 0.05}
    )
    minimum_orbit_intersection: float = Field(
        ..., ge=0.0, le=2.0,
        description="Minimum Orbit Intersection Distance (MOID) in AU.",
        json_schema_extra={"example": 0.01}
    )

class AsteroidScanResponse(BaseModel):
    is_hazardous: bool
    status: str
    confidence_score: float
    hazard_probability: float
    estimated_diameter_meters: float
    threat_classification: str
    model_version: str

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Docker / orchestration liveness probes."""
    m = get_model()
    return {
        "status": "online",
        "service": "NEO Guardian API",
        "model_loaded": m is not None,
        "expected_features": list(m.feature_names_in_) if m else []
    }

@app.get("/model-info", status_code=status.HTTP_200_OK)
def model_info():
    """Returns metadata and live feature importance percentages."""
    m = get_model()
    importances = {}
    if hasattr(m, "feature_importances_") and hasattr(m, "feature_names_in_"):
        for name, imp in zip(m.feature_names_in_, m.feature_importances_):
            importances[name] = round(float(imp) * 100, 2)
    return {
        "model_type": type(m).__name__,
        "n_features": len(m.feature_names_in_),
        "features": list(m.feature_names_in_),
        "feature_importances_pct": importances
    }

@app.post("/predict", response_model=AsteroidScanResponse, status_code=status.HTTP_200_OK)
def predict_hazard(payload: AsteroidScanRequest):
    """
    Evaluates planetary hazard probability and physical diameter using the trained AI model and IAU formula.
    """
    m = get_model()
    
    # Construct input dataframe matching exact model feature names
    input_data = pd.DataFrame([{
        "Absolute Magnitude": payload.absolute_magnitude,
        "Orbit Uncertainity": payload.orbit_uncertainty,
        "Relative Velocity km per sec": payload.relative_velocity,
        "Eccentricity": payload.eccentricity,
        "Miss Dist.(Astronomical)": payload.miss_distance,
        "Minimum Orbit Intersection": payload.minimum_orbit_intersection
    }])[m.feature_names_in_]
    
    try:
        prediction = int(m.predict(input_data)[0])
        probability = float(m.predict_proba(input_data)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    # Astronomical diameter formula: D (km) = (1329 / sqrt(albedo)) * 10^(-0.2 * H)
    albedo = 0.15
    try:
        diameter_km = (1329 / np.sqrt(albedo)) * (10 ** (-0.2 * payload.absolute_magnitude))
        diameter_meters = round(float(diameter_km * 1000), 2)
    except Exception:
        diameter_meters = 0.0

    # Categorize threat tier
    if diameter_meters < 25:
        threat_class = "Meteoroid (Atmospheric Burn-up Likely)"
    elif diameter_meters < 140:
        threat_class = "City-Level Threat (Local Impact)"
    elif diameter_meters < 1000:
        threat_class = "Potentially Hazardous (Regional Impact)"
    else:
        threat_class = "Planet-Killer (Extinction Event)"

    return AsteroidScanResponse(
        is_hazardous=(prediction == 1),
        status="HAZARDOUS OBJECT" if prediction == 1 else "SECURE / NOMINAL",
        confidence_score=round(probability if prediction == 1 else (1.0 - probability), 4),
        hazard_probability=round(probability, 4),
        estimated_diameter_meters=diameter_meters,
        threat_classification=threat_class,
        model_version="v2.0-balanced"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

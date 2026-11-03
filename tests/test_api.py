"""Unit and Integration Tests for AgroScope FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient
from api.app.config import settings
from api.app.main import app

client = TestClient(app)

# Models and frontend builds are not in the repo, so a fresh clone or CI run won't have them.
needs_model = pytest.mark.skipif(
    not settings.MODEL_PATH.exists(), reason="Model artifacts missing; run `dvc pull` or `dvc repro`"
)
needs_frontend_build = pytest.mark.skipif(
    not (settings.BASE_DIR / "frontend" / "dist").exists(), reason="Frontend build missing; run `npm run build`"
)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "AgroScope" in data["system"]
    assert data["health"] == "/health"


@needs_model
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert data["model"] == "LOADED"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["uptime_seconds"] >= 0
    assert data["total_inferences"] >= 0
    assert data["memory_rss_mb"] > 0


def test_study_area_geojson():
    response = client.get("/study-area")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0


def test_available_dates():
    response = client.get("/available-dates")
    assert response.status_code == 200
    dates = response.json()
    assert isinstance(dates, list)
    assert len(dates) > 0


def test_statistics():
    response = client.get("/statistics")
    assert response.status_code == 200
    stats = response.json()
    assert "total_cultivated_hectares" in stats
    assert "stress_breakdown" in stats


def test_stress_map_sampled():
    response = client.get("/stress-map?sample=true&limit=50")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) <= 50


@needs_model
def test_on_demand_prediction():
    payload = {
        "ndvi_mean": 0.18,
        "ndwi_mean": -0.05,
        "temp_max_c": 36.5,
        "rainfall_8day": 0.0,
        "et0_8day_sum": 38.0,
        "crop_type": "Sugarcane",
        "lang": "en"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["LOW", "MODERATE", "HIGH"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "advisory" in data
    assert data["advisory"]["priority"] in [1, 2, 3]
    assert len(data["key_drivers"]) > 0


def test_drift_monitoring():
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY (NO DRIFT DETECTED)"
    assert data["data_drift_detected"] is False


@needs_frontend_build
def test_dashboard_endpoint():
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "AgroScope" in response.text


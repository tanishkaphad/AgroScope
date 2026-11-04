"""Unit and Integration Tests for AgroScope FastAPI REST API."""

import json
import random

import pytest
from fastapi.testclient import TestClient
from api.app.config import settings
from api.app.main import app
from api.app.services.geo_service import geo_service
from api.app.services.model_service import model_service
from api.app.services.retrain_service import retrain_service

client = TestClient(app)

# Models, results and frontend builds are not in the repo, so a fresh clone or CI run won't have them.
needs_model = pytest.mark.skipif(
    not settings.MODEL_PATH.exists(), reason="Model artifacts missing; run `dvc pull` or `dvc repro`"
)
needs_results = pytest.mark.skipif(
    not settings.STRESS_MAP_GEOJSON.exists(), reason="Prediction results missing; run `dvc pull` or `dvc repro`"
)
needs_frontend_build = pytest.mark.skipif(
    not (settings.BASE_DIR / "frontend" / "dist").exists(), reason="Frontend build missing; run `npm run build`"
)


@pytest.fixture
def no_results(monkeypatch):
    """Simulate the API running without any prediction result files."""
    monkeypatch.setattr(geo_service, "stress_map", None)
    monkeypatch.setattr(geo_service, "irrigation_map", None)
    monkeypatch.setattr(geo_service, "cycle_stats", None)
    monkeypatch.setattr(geo_service, "cell_lookup", {})
    monkeypatch.setattr(geo_service, "irrigation_lookup", {})
    monkeypatch.setattr(geo_service, "available_dates", [])


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
    assert data["model"] == "LOADED"
    assert data["model_version"].split("-")[0] in ("random_forest", "xgboost", "lightgbm")


def test_health_is_degraded_without_results(no_results):
    data = client.get("/health").json()
    assert data["status"] == "DEGRADED"


def test_health_reports_database_is_not_connected(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_KEY", "example-key")
    data = client.get("/health").json()
    assert data["database"].startswith("NOT_CONNECTED")


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["uptime_seconds"] >= 0
    assert data["total_inferences"] >= 0
    assert data["memory_rss_mb"] > 0
    assert "formula-based labels" in data["score_note"]


def test_metrics_without_model_reports_no_score(monkeypatch):
    monkeypatch.setattr(settings, "METADATA_PATH", settings.MODELS_DIR / "does_not_exist.json")
    monkeypatch.setattr(model_service, "predictor", None)
    data = client.get("/metrics").json()
    assert data["active_model"] is None
    assert data["macro_f1_score"] is None


def test_study_area_geojson():
    response = client.get("/study-area")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0


@needs_results
def test_available_dates_match_published_results():
    response = client.get("/available-dates")
    assert response.status_code == 200
    published = {feat["properties"]["date"] for feat in geo_service.stress_map["features"]}
    assert set(response.json()) == published


def test_available_dates_empty_without_results(no_results):
    assert client.get("/available-dates").json() == []


@needs_results
def test_statistics():
    response = client.get("/statistics")
    assert response.status_code == 200
    stats = response.json()
    assert "total_cultivated_hectares" in stats
    assert "stress_breakdown" in stats


def test_statistics_unavailable_without_results(no_results):
    assert client.get("/statistics").status_code == 503


@needs_results
def test_stress_map_sampled():
    response = client.get("/stress-map?sample=true&limit=50")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) <= 50


@needs_results
def test_stress_map_rejects_unavailable_date():
    assert client.get("/stress-map?date=1999-01-01").status_code == 404


def test_maps_unavailable_without_results(no_results):
    assert client.get("/stress-map").status_code == 503
    assert client.get("/irrigation-map").status_code == 503


@needs_results
def test_area_detail_matches_published_maps():
    features = random.Random(0).sample(geo_service.stress_map["features"], 100)
    for feat in features:
        props = feat["properties"]
        response = client.get(f"/area/{props['cell_id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["water_stress"]["class"] == props["stress_class"]
        assert data["water_stress"]["confidence"] == props["confidence"]
        assert data["date"] == props["date"]
        irrigation = geo_service.get_irrigation_detail(props["cell_id"])["properties"]
        assert data["irrigation_advisory"]["priority"] == irrigation["priority"]


@needs_results
def test_area_detail_does_not_invent_crop_or_soil():
    cell_id = geo_service.stress_map["features"][0]["properties"]["cell_id"]
    data = client.get(f"/area/{cell_id}").json()
    assert data["crop_type"] is None
    assert data["dominant_soil"] is None


@needs_results
def test_area_detail_uses_requested_language():
    cell_id = geo_service.stress_map["features"][0]["properties"]["cell_id"]
    data = client.get(f"/area/{cell_id}?lang=mr").json()
    irrigation = geo_service.get_irrigation_detail(cell_id)["properties"]
    assert data["irrigation_advisory"]["title"] == irrigation["title_mr"]


@needs_results
def test_area_detail_unknown_cell_returns_404():
    assert client.get("/area/DOES_NOT_EXIST").status_code == 404


def test_area_detail_unavailable_without_results(no_results):
    assert client.get("/area/NLBC_CELL_0008").status_code == 503


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
    assert "Simulation only" in data["note"]


def test_prediction_unavailable_without_model(monkeypatch):
    monkeypatch.setattr(model_service, "predictor", None)
    assert client.post("/predict", json={}).status_code == 503


def test_drift_summary_is_served_from_its_file(tmp_path, monkeypatch):
    path = tmp_path / "drift_summary.json"
    path.write_text(json.dumps({"drift_detected": True, "drifted_share": 0.6}), encoding="utf-8")
    monkeypatch.setattr(settings, "DRIFT_SUMMARY_JSON", path)
    assert client.get("/monitoring/drift").json()["drifted_share"] == 0.6


def test_drift_unavailable_without_report(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DRIFT_SUMMARY_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(settings, "DRIFT_REPORT_HTML", tmp_path / "missing.html")
    assert client.get("/monitoring/drift").status_code == 503
    assert client.get("/monitoring/drift/report").status_code == 503


def test_promotion_unavailable_without_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROMOTION_DECISION_JSON", tmp_path / "missing.json")
    assert client.get("/monitoring/promotion").status_code == 503


def test_retraining_disabled_without_configured_token(monkeypatch):
    monkeypatch.setattr(settings, "RETRAIN_TOKEN", "")
    assert client.post("/retrain", headers={"X-Admin-Token": "anything"}).status_code == 503


def test_retraining_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(settings, "RETRAIN_TOKEN", "correct-token")
    assert client.post("/retrain", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.post("/retrain").status_code == 403


def test_retraining_starts_with_valid_token(monkeypatch):
    monkeypatch.setattr(settings, "RETRAIN_TOKEN", "correct-token")
    monkeypatch.setattr(retrain_service, "start", lambda: True)
    response = client.post("/retrain", headers={"X-Admin-Token": "correct-token"})
    assert response.status_code == 202


@needs_frontend_build
def test_dashboard_endpoint():
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "AgroScope" in response.text


def test_prometheus_endpoint_records_requests_by_route_template():
    client.get("/area/DOES_NOT_EXIST")
    body = client.get("/metrics/prometheus").text
    assert 'route="/area/{cell_id}"' in body
    assert "DOES_NOT_EXIST" not in body
    assert "agroscope_model_loaded" in body


def test_json_metrics_endpoint_stays_json():
    assert client.get("/metrics").json()["uptime_seconds"] >= 0


@needs_model
def test_prediction_is_counted_by_class_and_timed():
    client.post("/predict", json={"ndvi_mean": 0.18, "ndwi_mean": -0.05})
    body = client.get("/metrics/prometheus").text
    assert "agroscope_predictions_total{stress_class=" in body
    assert "agroscope_inference_duration_seconds_count" in body


def test_drift_gauge_reads_both_scenarios(tmp_path, monkeypatch):
    published, simulated = tmp_path / "published.json", tmp_path / "simulated.json"
    published.write_text(json.dumps({"drifted_share": 0.25}), encoding="utf-8")
    simulated.write_text(json.dumps({"drifted_share": 0.75}), encoding="utf-8")
    monkeypatch.setattr(settings, "DRIFT_SUMMARY_JSON", published)
    monkeypatch.setattr(settings, "DRIFT_SIMULATION_JSON", simulated)
    body = client.get("/metrics/prometheus").text
    assert 'agroscope_drift_share{scenario="published"} 0.25' in body
    assert 'agroscope_drift_share{scenario="simulated"} 0.75' in body

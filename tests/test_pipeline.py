"""Unit Tests for AgroScope settings and data ingestion, with no network access."""

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.config import load_settings, settings
from src.ingestion import fetch_satellite, fetch_weather
from src.monitoring.drift import is_drifted, parse_drift_metrics
from src.preprocessing import cloud_features
from src.preprocessing.preprocess_bands import compute_spectral_indices, remove_boa_offset
from src.target.generate_labels import assign_nearest_weather_points, compute_climatic_water_deficit
from src.training import promote, tracking
from src.training.promote import decide_promotion
from src.training.train import FEATURE_COLUMNS


def test_settings_load_every_section():
    assert {"aoi", "satellite", "weather", "labels", "training", "inference"} <= set(load_settings())


def test_training_features_come_from_settings():
    assert FEATURE_COLUMNS == settings["training"]["features"]
    assert len(FEATURE_COLUMNS) == 25


def test_weather_grid_covers_the_area_with_unique_points():
    points = fetch_weather.build_weather_grid(box(74.0, 18.0, 74.3, 18.2), 0.1)
    assert len(points) >= 12
    assert len({p["weather_point_id"] for p in points}) == len(points)
    assert all(73.9 <= p["longitude"] <= 74.4 for p in points)


def test_weather_request_asks_for_wind_in_metres_per_second(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout):
        captured.update(params)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"daily": {"time": ["2024-01-01"], "wind_speed_10m_max": [4.0]}},
        )

    monkeypatch.setattr(fetch_weather.requests, "get", fake_get)
    df = fetch_weather.request_daily_weather("https://example.test", 18.0, 74.0, {"forecast_days": 7})
    assert captured["wind_speed_unit"] == "ms"
    assert captured["forecast_days"] == 7
    assert df["wind_speed_mps"].tolist() == [4.0]


def test_rolling_features_are_calculated_per_point():
    rows = []
    for point_id, rain in (("WX_000", 1.0), ("WX_001", 10.0)):
        for day in pd.date_range("2024-01-01", periods=3):
            rows.append({
                "weather_point_id": point_id,
                "date": day,
                "rainfall_mm": rain,
                "temp_mean_c": 25.0,
                "temp_max_c": 30.0,
                "et0_mm": 4.0,
            })
    df = fetch_weather.add_grid_features(pd.DataFrame(rows))
    last_values = df.groupby("weather_point_id")["rainfall_8day"].last()
    assert last_values["WX_000"] == 3.0
    assert last_values["WX_001"] == 30.0


def test_cells_use_their_nearest_weather_point():
    cells = pd.DataFrame({"cell_id": ["A", "B"], "centroid_lon": [74.01, 74.29], "centroid_lat": [18.0, 18.0]})
    points = pd.DataFrame({
        "weather_point_id": ["WX_000", "WX_001"],
        "longitude": [74.0, 74.3],
        "latitude": [18.0, 18.0],
    })
    result = assign_nearest_weather_points(cells, points)
    assert dict(zip(result["cell_id"], result["weather_point_id"])) == {"A": "WX_000", "B": "WX_001"}


def test_label_thresholds_come_from_settings():
    labels = settings["labels"]
    kc = labels["default_kc"]
    low_et0 = (labels["deficit_low_mm"] - 1) / kc
    high_et0 = (labels["deficit_high_mm"] + 1) / kc
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "et0_8day_sum": [low_et0, high_et0], "rainfall_8day": [0.0, 0.0]})
    assert compute_climatic_water_deficit(df)["target_stress_label"].tolist() == ["LOW", "HIGH"]


def test_manifest_keeps_old_scenes_when_adding_new_ones():
    existing = pd.DataFrame({"scene_id": ["S1"], "acquisition_date": ["2024-01-01"], "status": ["complete"]})
    merged = fetch_satellite.merge_manifest(existing, [{"scene_id": "S2", "acquisition_date": "2024-01-06", "status": "found"}])
    assert merged["scene_id"].tolist() == ["S1", "S2"]


def test_since_last_starts_from_newest_known_scene():
    manifest = pd.DataFrame({"scene_id": ["S1", "S2"], "acquisition_date": ["2024-01-01", "2024-03-29"]})
    assert fetch_satellite.resolve_since_last_start(manifest, "2023-10-01") == "2024-03-29"
    assert fetch_satellite.resolve_since_last_start(pd.DataFrame(columns=["acquisition_date"]), "2023-10-01") == "2023-10-01"


def test_scene_is_complete_only_with_every_band(tmp_path):
    for band in settings["satellite"]["bands"]:
        (tmp_path / f"{band}.tif").write_bytes(b"0" * 2000)
    assert fetch_satellite.is_scene_complete(tmp_path)
    (tmp_path / "B02.tif").unlink()
    assert not fetch_satellite.is_scene_complete(tmp_path)


def test_offset_removal_gives_true_ndvi():
    bands_raw = {name: np.full((2, 2), value, dtype=np.uint16) for name, value in {"B02": 1300, "B04": 1500, "B08": 4000, "B11": 3000}.items()}
    scl = np.full((2, 2), 4, dtype=np.uint8)
    corrected = {name: remove_boa_offset(band) for name, band in bands_raw.items()}
    ndvi, _, _, _ = compute_spectral_indices(corrected["B02"], corrected["B04"], corrected["B08"], corrected["B11"], scl)
    assert ndvi[0, 0] == pytest.approx((3000 - 500) / (3000 + 500), abs=1e-3)


def test_offset_removal_never_goes_negative():
    assert remove_boa_offset(np.array([0, 500, 1500], dtype=np.uint16)).tolist() == [0, 0, 500]


def _fake_item(item_id, day, tile, baseline):
    return SimpleNamespace(
        id=item_id,
        datetime=datetime(2024, 1, day),
        properties={"s2:mgrs_tile": tile, "s2:processing_baseline": baseline},
    )


def test_offset_depends_on_processing_baseline():
    assert cloud_features.offset_for_item(_fake_item("new", 1, "43QDA", "05.10")) == settings["satellite"]["boa_add_offset"]
    assert cloud_features.offset_for_item(_fake_item("old", 1, "43QDA", "02.14")) == 0


def test_only_latest_reprocessed_copy_of_a_scene_is_kept():
    items = [
        _fake_item("old_copy", 1, "43QDA", "05.09"),
        _fake_item("new_copy", 1, "43QDA", "05.10"),
        _fake_item("other_tile", 1, "43QDV", "05.09"),
    ]
    assert {item.id for item in cloud_features.keep_latest_reprocessing(items)} == {"new_copy", "other_tile"}


def test_feature_search_starts_from_newest_known_scene(tmp_path):
    raw_manifest = tmp_path / "scenes_manifest.csv"
    pd.DataFrame({"scene_id": ["S1"], "acquisition_date": ["2024-03-29"]}).to_csv(raw_manifest, index=False)
    features = pd.DataFrame({"scene_id": ["S2"], "acquisition_date": ["2024-04-03"]})
    assert cloud_features.resolve_start_date(features, raw_manifest) == "2024-04-03"
    assert cloud_features.resolve_start_date(pd.DataFrame(), raw_manifest) == "2024-03-29"


def test_feature_manifest_is_saved_and_replaces_retried_scenes(tmp_path):
    manifest = cloud_features.load_feature_manifest(tmp_path)
    manifest = cloud_features.save_feature_manifest(tmp_path, manifest, {"scene_id": "S1", "acquisition_date": "2024-04-03", "status": "failed"})
    manifest = cloud_features.save_feature_manifest(tmp_path, manifest, {"scene_id": "S1", "acquisition_date": "2024-04-03", "status": "complete"})
    saved = cloud_features.load_feature_manifest(tmp_path)
    assert saved["status"].tolist() == ["complete"]


def test_weather_request_waits_longer_when_rate_limited(monkeypatch):
    responses = [
        SimpleNamespace(status_code=429),
        SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"daily": {"time": ["2024-01-01"]}},
        ),
    ]
    sleeps = []

    def fail_then_succeed(url, params, timeout):
        response = responses.pop(0)
        if response.__dict__.get("status_code") == 429:
            raise fetch_weather.requests.exceptions.HTTPError(response=response)
        return response

    monkeypatch.setattr(fetch_weather.requests, "get", fail_then_succeed)
    monkeypatch.setattr(fetch_weather.time, "sleep", sleeps.append)
    fetch_weather.request_daily_weather("https://example.test", 18.0, 74.0, {})
    assert sleeps == [settings["weather"]["rate_limit_wait_s"]]


def test_promotion_requires_candidate_to_match_champion():
    assert decide_promotion(0.8, None, 0.0)[0] is True
    assert decide_promotion(0.8, 0.8, 0.0)[0] is True
    assert decide_promotion(0.79, 0.8, 0.0)[0] is False
    assert decide_promotion(0.8, 0.8, 0.01)[0] is False


def test_drift_direction_depends_on_test_type():
    assert is_drifted("Wasserstein distance (normed)", 0.2, 0.1) is True
    assert is_drifted("Wasserstein distance (normed)", 0.05, 0.1) is False
    assert is_drifted("K-S p_value", 0.01, 0.05) is True
    assert is_drifted("K-S p_value", 0.5, 0.05) is False


def test_drift_metrics_are_parsed_from_evidently_output():
    metrics = [
        {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 1.0, "share": 0.5}},
        {"config": {"type": "evidently:metric_v2:ValueDrift", "column": "ndvi_mean", "method": "Wasserstein distance (normed)", "threshold": 0.1}, "value": 0.4},
        {"config": {"type": "evidently:metric_v2:ValueDrift", "column": "ndwi_mean", "method": "Wasserstein distance (normed)", "threshold": 0.1}, "value": 0.01},
    ]
    parsed = parse_drift_metrics(metrics)
    assert parsed["drifted_share"] == 0.5
    assert [f["drift"] for f in parsed["features"]] == [True, False]


def _log_candidate(tmp_path, name, score):
    candidate = tmp_path / name
    candidate.mkdir()
    for file_name in promote.EXPORTED_FILES:
        (candidate / file_name).write_text(json.dumps({"model": name}), encoding="utf-8")
    run_id = tracking.log_training_run({"name": name}, {"temporal_macro_f1": score}, {}, [candidate / "xgboost_model.pkl", candidate / "xgboost_model.json"])
    (candidate / "feature_metadata.json").write_text(json.dumps({"mlflow_run_id": run_id, "model": name}), encoding="utf-8")
    tracking.attach_run_artifact(run_id, candidate / "feature_metadata.json")
    return candidate


def test_gate_promotes_better_model_and_keeps_champion_over_worse_one(tmp_path, monkeypatch):
    artifact_root = (tmp_path / "artifacts").as_uri()
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(tracking, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(promote, "ARTIFACT_ROOT", artifact_root)
    production = tmp_path / "production"
    decision_path = tmp_path / "decision.json"

    def run_gate(name, score):
        return promote.run_promotion_gate(_log_candidate(tmp_path, name, score), production, decision_path)

    first = run_gate("first", 0.80)
    assert first["promoted"] is True
    assert first["active_version"] == 1

    worse = run_gate("worse", 0.70)
    assert worse["promoted"] is False
    assert worse["active_version"] == 1
    assert json.loads((production / "xgboost_model.pkl").read_text(encoding="utf-8"))["model"] == "first"

    better = run_gate("better", 0.90)
    assert better["promoted"] is True
    assert better["active_version"] == 3
    assert json.loads((production / "xgboost_model.pkl").read_text(encoding="utf-8"))["model"] == "better"

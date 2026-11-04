"""Preprocessing and prediction tests using a tiny model, so they run in CI without the real data."""

import json

import numpy as np
import pytest
import rasterio

from scripts.make_placeholder_artifacts import build_tiny_model, make_placeholders, write_tiny_cycle_inputs
from src.config import settings
from src.inference.explainability import ModelExplainer
from src.inference.predict import AgroScopePredictor
from src.preprocessing.preprocess_bands import compute_spectral_indices, extract_grid_zonal_stats_fast
from src.target.generate_labels import compute_climatic_water_deficit


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("tiny_model")
    build_tiny_model(model_dir)
    return model_dir


@pytest.fixture(scope="module")
def predictor(tiny_model_dir):
    return AgroScopePredictor(tiny_model_dir / "model.pkl", tiny_model_dir / "feature_metadata.json")


def test_prediction_returns_a_class_with_advice_and_shap_drivers(predictor):
    result = predictor.predict_instance({"ndvi_mean": 0.4, "ndwi_mean": -0.15, "temp_max_c": 35.0, "et0_8day_sum": 40.0, "rainfall_8day": 0.0})
    prediction, advisory = result["prediction"], result["advisory"]
    assert prediction["predicted_class_label"] in ("LOW", "MODERATE", "HIGH")
    assert sum(prediction["class_probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert len(prediction["key_drivers"]) == 4
    assert advisory["priority"] in (1, 2, 3)
    assert set(advisory["title"]) == {"en", "mr", "hi"}


def test_prediction_follows_the_input_the_tiny_model_learned(predictor):
    dry = predictor.predict_instance({"ndwi_mean": -0.2})["prediction"]["predicted_class_label"]
    wet = predictor.predict_instance({"ndwi_mean": 0.3})["prediction"]["predicted_class_label"]
    assert dry == "HIGH"
    assert wet == "LOW"


def test_shap_explainer_loads_for_the_tree_model(tiny_model_dir):
    explainer = ModelExplainer(tiny_model_dir / "model.pkl", tiny_model_dir / "feature_metadata.json")
    assert explainer._explainer is not None
    drivers = explainer.explain_instance({"ndwi_mean": -0.2})["key_drivers"]
    assert drivers[0]["feature"] == "ndwi_mean"


def test_batch_prediction_writes_matching_maps_and_statistics(predictor, tmp_path):
    features_csv, grid_geojson = write_tiny_cycle_inputs(tmp_path / "inputs", cells=30)
    predictor.run_cycle_inference(features_csv=features_csv, grid_geojson=grid_geojson, output_dir=tmp_path)

    stress = json.loads((tmp_path / "stress_map.geojson").read_text(encoding="utf-8"))["features"]
    irrigation = json.loads((tmp_path / "irrigation_map.geojson").read_text(encoding="utf-8"))["features"]
    stats = json.loads((tmp_path / "cycle_statistics.json").read_text(encoding="utf-8"))

    assert len(stress) == len(irrigation) == 30
    assert {f["properties"]["cell_id"] for f in stress} == {f["properties"]["cell_id"] for f in irrigation}
    assert stats["total_cultivated_hectares"] == 30 * settings["aoi"]["cell_area_ha"]
    breakdown = stats["stress_breakdown"]
    assert breakdown["low_stress_pct"] + breakdown["moderate_stress_pct"] + breakdown["high_stress_pct"] == pytest.approx(100.0, abs=0.2)


def test_batch_prediction_rejects_an_unknown_date(predictor, tmp_path):
    features_csv, grid_geojson = write_tiny_cycle_inputs(tmp_path / "inputs", cells=5)
    with pytest.raises(ValueError, match="not in dataset"):
        predictor.run_cycle_inference(date_str="1999-01-01", features_csv=features_csv, grid_geojson=grid_geojson, output_dir=tmp_path)


def test_placeholders_refuse_to_replace_a_real_model(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.pkl").write_bytes(b"real model")
    with pytest.raises(SystemExit, match="Refusing"):
        make_placeholders(tmp_path)
    assert (tmp_path / "models" / "model.pkl").read_bytes() == b"real model"


def _band(value):
    return np.full((20, 20), value, dtype=np.float32)


def _zonal_stats(valid_mask, cell_bounds=(0.0, 0.0, 200.0, 200.0)):
    ndvi, ndwi, evi = _band(0.5), _band(0.1), _band(0.3)
    return extract_grid_zonal_stats_fast(
        [{"cell_id": "CELL", "bounds": cell_bounds, "centroid_lon": 74.0, "centroid_lat": 18.0}],
        ndvi, ndwi, evi, _band(100), _band(200), _band(300), _band(400), _band(500), _band(600),
        valid_mask,
        rasterio.transform.from_origin(0.0, 200.0, 10.0, 10.0),
        (20, 20),
        "2024-01-01",
        "SCENE",
    )


def test_cell_values_ignore_cloudy_pixels_and_report_clear_share():
    mask = np.ones((20, 20), dtype=bool)
    mask[:10, :] = False
    record = _zonal_stats(mask)[0]
    assert record["valid_pixel_pct"] == 50.0
    assert record["ndvi_mean"] == 0.5
    assert record["b08_mean"] == 400.0


def test_cell_with_too_few_clear_pixels_is_dropped():
    mask = np.zeros((20, 20), dtype=bool)
    mask[0, :4] = True
    assert _zonal_stats(mask) == []


def test_cell_outside_the_scene_is_dropped():
    assert _zonal_stats(np.ones((20, 20), dtype=bool), cell_bounds=(1000.0, 1000.0, 1200.0, 1200.0)) == []


def test_cloud_pixels_become_missing_in_spectral_indices():
    scl = np.array([[4, 9]])  # 4 is vegetation, 9 is high cloud
    ndvi, _, _, valid = compute_spectral_indices(np.full((1, 2), 500.0), np.full((1, 2), 500.0), np.full((1, 2), 3000.0), np.full((1, 2), 1500.0), scl)
    assert valid.tolist() == [[True, False]]
    assert ndvi[0, 0] == pytest.approx(0.7143, abs=0.001)
    assert np.isnan(ndvi[0, 1])

"""Create small placeholder results, so CI can test and build without the real data.

The placeholders go through the real prediction code, so they have the real file formats, but
their contents are synthetic and mean nothing. The script refuses to run where a real model exists.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.inference.predict import AgroScopePredictor

FEATURES = settings["training"]["features"]
CLASS_LABELS = ["LOW", "MODERATE", "HIGH"]
PLACEHOLDER_NOTE = "Synthetic placeholder for CI. Not a real result."
CYCLE_DATE = "2024-01-01"


def make_feature_rows(rows: int, seed: int = 0) -> pd.DataFrame:
    """Random inputs in realistic ranges, with a stress label that follows canopy water."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.uniform(0.1, 0.5, size=(rows, len(FEATURES))), columns=FEATURES)
    frame["ndwi_mean"] = rng.uniform(-0.2, 0.3, size=rows)
    frame["target_stress_class"] = np.digitize(-frame["ndwi_mean"], [-0.15, 0.0])
    return frame


def build_tiny_model(model_dir: Path, rows: int = 300) -> None:
    """Train a very small real model, and save it with its metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = make_feature_rows(rows)
    model = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=10, max_depth=3, n_jobs=1, random_state=0, verbose=-1)
    model.fit(frame[FEATURES], frame["target_stress_class"])

    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    metadata = {
        "features": FEATURES,
        "classes": [0, 1, 2],
        "class_labels": CLASS_LABELS,
        "model_type": "lightgbm",
        "selected_model": "lightgbm",
        "placeholder": PLACEHOLDER_NOTE,
    }
    with open(model_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def write_tiny_cycle_inputs(work_dir: Path, cells: int = 120) -> tuple[Path, Path]:
    """Write one date of features and a matching grid, in the formats prediction reads."""
    work_dir.mkdir(parents=True, exist_ok=True)
    frame = make_feature_rows(cells, seed=1).drop(columns="target_stress_class")
    frame["cell_id"] = [f"PLACEHOLDER_CELL_{i:04d}" for i in range(cells)]
    frame["date"] = CYCLE_DATE
    frame["is_vegetated"] = True
    features_csv = work_dir / "features.csv"
    frame.to_csv(features_csv, index=False)

    grid = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"cell_id": cell_id},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74 + i * 0.01, 18.0], [74.01 + i * 0.01, 18.0], [74.01 + i * 0.01, 18.01], [74 + i * 0.01, 18.01], [74 + i * 0.01, 18.0]]],
                },
            }
            for i, cell_id in enumerate(frame["cell_id"])
        ],
    }
    grid_geojson = work_dir / "grid.geojson"
    with open(grid_geojson, "w", encoding="utf-8") as f:
        json.dump(grid, f)
    return features_csv, grid_geojson


def _write_if_absent(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def make_placeholders(root: Path) -> None:
    """Create everything the API and its Docker image expect to find."""
    model_dir = root / "models"
    if (model_dir / "model.pkl").exists():
        raise SystemExit(f"{model_dir / 'model.pkl'} already exists. Refusing to replace a real model with a placeholder.")

    build_tiny_model(model_dir)

    processed = root / "data" / "processed"
    features_csv, grid_geojson = write_tiny_cycle_inputs(root / "data" / "placeholder_inputs")
    predictor = AgroScopePredictor(model_dir / "model.pkl", model_dir / "feature_metadata.json")
    predictor.run_cycle_inference(features_csv=features_csv, grid_geojson=grid_geojson, output_dir=processed)

    note = json.dumps({"placeholder": PLACEHOLDER_NOTE}, indent=2)
    _write_if_absent(processed / "drift_summary.json", note)
    _write_if_absent(processed / "drift_report.html", f"<!doctype html><title>Placeholder</title><p>{PLACEHOLDER_NOTE}</p>")
    _write_if_absent(root / "reports" / "promotion_decision.json", note)
    _write_if_absent(root / "frontend" / "dist" / "index.html", f"<!doctype html><title>AgroScope</title><p>{PLACEHOLDER_NOTE}</p>")


if __name__ == "__main__":
    # Prediction prints emoji, which the default Windows encoding can't write.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project folder to write into")
    make_placeholders(parser.parse_args().root)

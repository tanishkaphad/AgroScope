"""AgroScope Data Drift Monitoring with Evidently.

Compares the published cycle's model inputs with data from the training period
and saves a summary for the API, plus a full HTML report.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.monitoring")

MONITORING = settings["monitoring"]
TRAINING = settings["training"]
SEASONAL_NOTE = (
    "The reference data covers one past season only, so drift here can reflect normal seasonal change "
    "as well as a real problem."
)


def is_drifted(method: str, value: float, threshold: float) -> bool:
    """Apply a drift test: distances drift when high, p-values when low."""
    is_distance = any(word in method.lower() for word in ("distance", "divergence", "psi"))
    return value > threshold if is_distance else value < threshold


def parse_drift_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull the per-feature results and overall drift share from an Evidently report."""
    features: List[Dict[str, Any]] = []
    share = 0.0
    for metric in metrics:
        config = metric["config"]
        if config["type"].endswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif config["type"].endswith("ValueDrift"):
            value = float(metric["value"])
            features.append({
                "feature": config["column"],
                "test": config["method"],
                "score": round(value, 4),
                "threshold": config["threshold"],
                "drift": is_drifted(config["method"], value, config["threshold"]),
            })
    return {"drifted_share": round(share, 4), "features": sorted(features, key=lambda f: f["feature"])}


def run_drift_check(
    features_csv: Optional[Path] = None,
    stats_json: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare the published date's inputs with the training period, and save the results."""
    features_csv = features_csv or (PROJECT_ROOT / "data" / "features" / "features.csv")
    stats_json = stats_json or (PROJECT_ROOT / "data" / "processed" / "cycle_statistics.json")
    output_dir = output_dir or (PROJECT_ROOT / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(stats_json, "r", encoding="utf-8") as f:
        current_date = json.load(f)["cycle_date"]

    columns = TRAINING["features"]
    df = pd.read_csv(features_csv, usecols=["date", "is_vegetated", *columns])
    df = df[df["is_vegetated"]]

    split_date = TRAINING["temporal_split_date"]
    reference = df[df["date"] < split_date][columns]
    current = df[df["date"] == current_date][columns]
    if reference.empty or current.empty:
        raise ValueError(f"Drift check needs data before {split_date} and on {current_date}.")

    sample_rows = MONITORING["sample_rows"]
    reference_sample = reference.sample(min(sample_rows, len(reference)), random_state=TRAINING["random_state"])
    current_sample = current.sample(min(sample_rows, len(current)), random_state=TRAINING["random_state"])

    logger.info("Comparing %d rows from %s with %d training-period rows...", len(current_sample), current_date, len(reference_sample))
    result = Report([DataDriftPreset()]).run(current_data=current_sample, reference_data=reference_sample)
    result.save_html(str(output_dir / "drift_report.html"))

    parsed = parse_drift_metrics(result.dict()["metrics"])
    drift_detected = parsed["drifted_share"] >= MONITORING["drift_share_alert"]
    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "reference_period": f"before {split_date}",
        "current_date": current_date,
        "reference_rows": len(reference_sample),
        "current_rows": len(current_sample),
        "drift_share_alert": MONITORING["drift_share_alert"],
        "drift_detected": drift_detected,
        "retrain_recommended": drift_detected,
        "note": SEASONAL_NOTE,
        **parsed,
    }
    with open(output_dir / "drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Drift %s: %.1f%% of features drifted.", "detected" if drift_detected else "not detected", parsed["drifted_share"] * 100)
    return summary


if __name__ == "__main__":
    run_drift_check()

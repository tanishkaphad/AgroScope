"""Show that drift monitoring responds to a known change, using simulated data.

The published date already shows 100% drift, because it compares late season with early season,
so it can't show a response. This compares two random samples from the same training period,
first unchanged and then with shifted temperature, rainfall and NDWI. Results are saved separately
and labelled as simulated, so they can't be mistaken for real results.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.monitoring.drift import parse_drift_metrics

TRAINING = settings["training"]
MONITORING = settings["monitoring"]

# A hotter, drier period with less canopy water
TEMPERATURE_SHIFT_C = 4.0
RAINFALL_SHIFT_MM = 20.0
NDWI_SHIFT = -0.1
TEMPERATURE_COLUMNS = ["temp_mean_c", "temp_max_c", "temp_8day_mean", "temp_30day_mean", "temp_8day_max"]
RAINFALL_COLUMNS = ["rainfall_8day", "rainfall_16day", "rainfall_30day"]


def apply_shift(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the simulated change applied."""
    shifted = frame.copy()
    shifted[TEMPERATURE_COLUMNS] += TEMPERATURE_SHIFT_C
    shifted[RAINFALL_COLUMNS] += RAINFALL_SHIFT_MM
    shifted["ndwi_mean"] += NDWI_SHIFT
    return shifted


def compare(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[Dict[str, Any], Any]:
    result = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    return parse_drift_metrics(result.dict()["metrics"]), result


def run_simulation() -> Dict[str, Any]:
    columns = TRAINING["features"]
    frame = pd.read_csv(PROJECT_ROOT / "data" / "features" / "features.csv", usecols=["date", "is_vegetated", *columns])
    training_period = frame[frame["is_vegetated"] & (frame["date"] < TRAINING["temporal_split_date"])][columns]

    rows = MONITORING["sample_rows"]
    sample = training_period.sample(min(rows * 2, len(training_period)), random_state=TRAINING["random_state"])
    reference, unchanged = sample.iloc[: len(sample) // 2], sample.iloc[len(sample) // 2 :]

    control, _ = compare(reference, unchanged)
    simulated, report = compare(reference, apply_shift(unchanged))

    reports_dir = PROJECT_ROOT / "reports"
    report.save_html(str(reports_dir / "drift_simulation_report.html"))

    shifted_columns = {*TEMPERATURE_COLUMNS, *RAINFALL_COLUMNS, "ndwi_mean"}
    flagged = {f["feature"] for f in simulated["features"] if f["drift"]}
    summary = {
        "simulated": True,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Simulated data, not a real result.",
        "shifts": {"temperature_c": TEMPERATURE_SHIFT_C, "rainfall_mm": RAINFALL_SHIFT_MM, "ndwi": NDWI_SHIFT},
        "rows_per_sample": len(unchanged),
        "control_drifted_share": control["drifted_share"],
        "drifted_share": simulated["drifted_share"],
        "drift_share_alert": MONITORING["drift_share_alert"],
        "alert_triggered": simulated["drifted_share"] >= MONITORING["drift_share_alert"],
        "shifted_inputs": sorted(shifted_columns),
        "shifted_inputs_flagged": sorted(shifted_columns & flagged),
        "shifted_inputs_missed": sorted(shifted_columns - flagged),
        "unshifted_inputs_flagged": sorted(flagged - shifted_columns),
        "features": simulated["features"],
    }
    with open(reports_dir / "drift_simulation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    outcome = run_simulation()
    print(json.dumps({k: v for k, v in outcome.items() if k != "features"}, indent=2))

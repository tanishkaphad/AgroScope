"""AgroScope Data Validation.

Checks required columns, value ranges, missing values and date coverage, and stops the run
on any failure. `--stage raw` runs before preprocessing, and `--stage features` before training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

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
logger = logging.getLogger("agroscope.validation")

VALIDATION = settings["validation"]
TRAINING = settings["training"]
WEATHER = settings["weather"]


def _check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> Dict[str, Any]:
    missing = sorted(set(required) - set(df.columns))
    return _check("required_columns", not missing, f"Missing: {missing}" if missing else "All present")


def check_value_ranges(df: pd.DataFrame, ranges: Dict[str, List[float]]) -> Dict[str, Any]:
    """Flag columns with values outside their allowed range. Absent columns are the columns check's job."""
    bad = {}
    for column, (low, high) in ranges.items():
        if column in df.columns:
            outside = int(((df[column] < low) | (df[column] > high)).sum())
            if outside:
                bad[column] = outside
    return _check("value_ranges", not bad, f"Rows outside range: {bad}" if bad else f"{len(ranges)} columns within range")


def check_missing_values(df: pd.DataFrame, columns: Iterable[str], max_share: float) -> Dict[str, Any]:
    present = [c for c in columns if c in df.columns]
    shares = df[present].isna().mean()
    bad = {c: round(float(s), 4) for c, s in shares.items() if s > max_share}
    return _check("missing_values", not bad, f"Above {max_share:.1%}: {bad}" if bad else f"All within {max_share:.1%}")


def check_date_coverage(dates: pd.Series, start: str, end: str) -> Dict[str, Any]:
    """Require the data to span at least the expected dates."""
    first, last = str(dates.min()), str(dates.max())
    return _check("date_coverage", first <= start and last >= end, f"Data covers {first} to {last}, and {start} to {end} is required")


def validate_weather(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check the weather table."""
    required = ["date", "weather_point_id", *VALIDATION["weather_ranges"]]
    checks = [check_required_columns(df, required)]
    if not checks[0]["passed"]:
        return checks

    checks.append(check_value_ranges(df, VALIDATION["weather_ranges"]))
    checks.append(check_missing_values(df, required, VALIDATION["max_missing_share"]))
    checks.append(check_date_coverage(df["date"], WEATHER["start_date"], WEATHER["end_date"]))

    days_per_point = df.groupby("weather_point_id")["date"].nunique()
    checks.append(_check("no_gaps_between_points", days_per_point.nunique() == 1, f"Days per point range from {days_per_point.min()} to {days_per_point.max()}"))
    return checks


def validate_scenes(manifest: pd.DataFrame, weather_dates: pd.Series) -> List[Dict[str, Any]]:
    """Check the satellite scene list, and that weather exists for every scene date."""
    checks = [check_required_columns(manifest, VALIDATION["manifest_columns"])]
    if not checks[0]["passed"]:
        return checks

    dates = manifest["acquisition_date"]
    scene_dates = dates.nunique()
    checks.append(_check("scene_count", scene_dates >= VALIDATION["min_dates"], f"{scene_dates} scene dates, and at least {VALIDATION['min_dates']} are required"))
    checks.append(check_missing_values(manifest, VALIDATION["manifest_columns"], 0.0))
    checks.append(_check("weather_covers_scenes", weather_dates.min() <= dates.min() and weather_dates.max() >= dates.max(), f"Scenes run from {dates.min()} to {dates.max()}"))
    return checks


def validate_features(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check the training table."""
    target, split_date = TRAINING["target_column"], TRAINING["temporal_split_date"]
    required = ["cell_id", "date", "is_vegetated", target, *TRAINING["features"]]
    checks = [check_required_columns(df, required)]
    if not checks[0]["passed"]:
        return checks

    checks.append(check_value_ranges(df, VALIDATION["feature_ranges"]))
    checks.append(check_missing_values(df, required, VALIDATION["max_missing_share"]))

    bad_labels = sorted(set(df[target].dropna().unique()) - {0, 1, 2})
    checks.append(_check("label_values", not bad_labels, f"Unexpected labels: {bad_labels}" if bad_labels else "Labels are 0, 1 and 2"))

    duplicates = int(df.duplicated(["cell_id", "date"]).sum())
    checks.append(_check("one_row_per_cell_and_date", duplicates == 0, f"{duplicates} duplicates"))

    dates = df["date"].nunique()
    before, after = df.loc[df["date"] < split_date, "date"].nunique(), df.loc[df["date"] >= split_date, "date"].nunique()
    checks.append(_check("date_coverage", dates >= VALIDATION["min_dates"] and before > 0 and after > 0, f"{dates} dates: {before} before {split_date}, {after} from it onwards"))

    # Macro F1 on later dates needs every class present, or the gate's score means little.
    later_classes = df.loc[(df["date"] >= split_date) & df["is_vegetated"], target].nunique()
    checks.append(_check("all_classes_in_later_dates", later_classes == 3, f"{later_classes} of 3 classes"))
    return checks


def run_validation(stage: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run one stage's checks and save the results. The file has no timestamp, so identical data reruns nothing downstream."""
    if stage == "raw":
        weather = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "weather" / "daily_weather.csv")
        manifest = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "satellite" / "scenes_manifest.csv")
        checks = validate_weather(weather) + validate_scenes(manifest, weather["date"])
        rows = {"weather": len(weather), "scenes": len(manifest)}
    else:
        features = pd.read_csv(PROJECT_ROOT / "data" / "features" / "features.csv")
        checks = validate_features(features)
        rows = {"features": len(features)}

    report = {"stage": stage, "passed": all(c["passed"] for c in checks), "rows": rows, "checks": checks}
    output_path = output_path or (PROJECT_ROOT / "reports" / f"validation_{stage}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for check in checks:
        logger.log(logging.INFO if check["passed"] else logging.ERROR, "%s %s: %s", "PASS" if check["passed"] else "FAIL", check["check"], check["detail"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate pipeline data, and stop the run on failure.")
    parser.add_argument("--stage", choices=["raw", "features"], required=True)
    args = parser.parse_args()

    if not run_validation(args.stage)["passed"]:
        sys.exit(1)

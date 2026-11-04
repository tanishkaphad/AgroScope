"""Score the selected model type on later dates held back from training, by date and by canal reach.

Each date stands in for labels arriving after the prediction. Reaches split the canal by distance
from its headworks at the western end, because the data has no demographic variables to compare.
Labels come from a formula, so these show agreement with it rather than field accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.training.train import FEATURE_COLUMNS, TARGET_COL, train_model

SPLIT_DATE = settings["training"]["temporal_split_date"]
REACHES = ("head", "middle", "tail")
HIGH_CLASS = 2


def score_group(actual: pd.Series, predicted: np.ndarray) -> Dict[str, Any]:
    """Score one group. Macro F1 covers only classes present, so an absent class can't count as a failure."""
    present = sorted(set(actual.unique()))
    has_high = HIGH_CLASS in present
    return {
        "rows": int(len(actual)),
        "macro_f1": round(float(f1_score(actual, predicted, labels=present, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "high_stress_recall": round(float(recall_score(actual == HIGH_CLASS, predicted == HIGH_CLASS, zero_division=0)), 4) if has_high else None,
        "classes_present": len(present),
    }


def assign_reaches(longitudes: pd.Series) -> pd.Series:
    """Split into three bands of equal width from west (head) to east (tail)."""
    return pd.cut(longitudes, bins=len(REACHES), labels=REACHES).astype(str)


def run_holdout_scores(output_path: Path | None = None) -> Dict[str, Any]:
    output_path = output_path or (PROJECT_ROOT / "reports" / "holdout_scores.json")
    with open(PROJECT_ROOT / "models" / "feature_metadata.json", "r", encoding="utf-8") as f:
        model_type = json.load(f)["model_type"]

    columns = ["date", "centroid_lon", "is_vegetated", TARGET_COL, *FEATURE_COLUMNS]
    frame = pd.read_csv(PROJECT_ROOT / "data" / "features" / "features.csv", usecols=columns)
    frame = frame[frame["is_vegetated"]].copy()
    frame["reach"] = assign_reaches(frame["centroid_lon"])

    past, later = frame[frame["date"] < SPLIT_DATE], frame[frame["date"] >= SPLIT_DATE].copy()
    model = train_model(model_type, past[FEATURE_COLUMNS], past[TARGET_COL].astype(int))
    later["predicted"] = model.predict(later[FEATURE_COLUMNS])

    def scores_by(column: str) -> Dict[str, Any]:
        return {str(key): score_group(group[TARGET_COL], group["predicted"].to_numpy()) for key, group in later.groupby(column)}

    by_reach, by_date = scores_by("reach"), scores_by("date")
    reach_f1 = [by_reach[r]["macro_f1"] for r in REACHES]
    results = {
        "model_type": model_type,
        "note": "Trained only on dates before the split. Labels are formula-based, not field observations.",
        "split_date": SPLIT_DATE,
        "train_rows": int(len(past)),
        "overall": score_group(later[TARGET_COL], later["predicted"].to_numpy()),
        "by_date": by_date,
        "by_reach": {r: by_reach[r] for r in REACHES},
        "reach_longitude_bands": {
            r: [round(float(later.loc[later["reach"] == r, "centroid_lon"].min()), 3), round(float(later.loc[later["reach"] == r, "centroid_lon"].max()), 3)]
            for r in REACHES
        },
        "largest_gap_between_reaches": round(max(reach_f1) - min(reach_f1), 4),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    outcome = run_holdout_scores()
    print(json.dumps({key: outcome[key] for key in ("model_type", "overall", "by_reach", "largest_gap_between_reaches")}, indent=2))

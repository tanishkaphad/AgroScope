"""Save real SHAP explanations from the champion model, for the model card."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.explainability import ModelExplainer


def run_shap_example() -> dict:
    explainer = ModelExplainer()
    with open(PROJECT_ROOT / "data" / "processed" / "cycle_statistics.json", "r", encoding="utf-8") as f:
        cycle_date = json.load(f)["cycle_date"]

    frame = pd.read_csv(PROJECT_ROOT / "data" / "features" / "features.csv", usecols=["cell_id", "date", "is_vegetated", *explainer.feature_names])
    cycle = frame[(frame["date"] == cycle_date) & frame["is_vegetated"]]
    predicted = explainer.model.predict(cycle[explainer.feature_names])

    examples = []
    for class_code in sorted(set(int(p) for p in predicted)):
        row = cycle[predicted == class_code].iloc[0]
        explanation = explainer.explain_instance(row[explainer.feature_names].astype(float))
        examples.append({"cell_id": row["cell_id"], "date": cycle_date, **explanation})

    output = {
        "model_type": type(explainer.model).__name__,
        "note": "`direction` means pushing toward or away from the predicted class, whichever class that is (B-37).",
        "examples": examples,
    }
    with open(PROJECT_ROOT / "reports" / "shap_example.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    explainer.generate_global_summary_plot()
    return output


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = run_shap_example()
    for example in result["examples"]:
        print(example["cell_id"], example["predicted_class_label"], [d["feature"] for d in example["key_drivers"]])

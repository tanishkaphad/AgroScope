"""Build reports/performance_report.md from saved runs, so no number is typed by hand.

Run this after the pipeline, scripts/holdout_scores.py, scripts/load_test.py and scripts/simulate_drift.py.
It stops if any of their results are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the step that creates it first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def table(headers: list, rows: list) -> list:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":---" for _ in headers) + " |",
        *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
    ]


def build_report() -> str:
    evaluation = load(REPORTS / "evaluation_result.json")
    promotion = load(REPORTS / "promotion_decision.json")
    holdout = load(REPORTS / "holdout_scores.json")
    load_test = load(REPORTS / "load_test_results.json")
    simulation = load(REPORTS / "drift_simulation.json")
    drift = load(PROCESSED / "drift_summary.json")
    cycle = load(PROCESSED / "cycle_statistics.json")

    overall = load_test["overall"]
    predict = load_test["by_request"]["predict"]
    priority = cycle["irrigation_priority_breakdown"]
    selected = evaluation["scores_by_model"][evaluation["selected_model"]]

    lines = [
        "# AgroScope Performance Report",
        "",
        "Built by `scripts/build_performance_report.py` from saved results. No number here is typed by hand.",
        "",
        "## Summary",
        "",
        *table(
            ["Metric", "Value", "Source"],
            [
                ["Model: macro F1 on later dates", selected["later_dates_macro_f1"], "`reports/evaluation_result.json`"],
                ["Business: hectares flagged Priority 1 this cycle", f"{priority['priority_1_urgent_ha']:,} ha ({priority['priority_1_pct']}%)", "`data/processed/cycle_statistics.json`"],
                ["p95 latency, all requests", f"{overall['latency_ms']['p95']} ms", "`reports/load_test_results.json`"],
                ["Throughput", f"{overall['throughput_rps']} requests per second", "`reports/load_test_results.json`"],
                ["Error rate", f"{overall['error_rate']:.2%}", "`reports/load_test_results.json`"],
                ["Drift: share of inputs drifted on the published date", f"{drift['drifted_share']:.0%}", "`data/processed/drift_summary.json`"],
            ],
        ),
        "",
        "## Model performance",
        "",
        f"The promotion gate uses `{promotion['metric']}`, with a minimum of {promotion['min_score']}. Latest decision: {'promoted' if promotion['promoted'] else 'rejected'}, {promotion['candidate_model']} version {promotion['candidate_version']}. {promotion['reason']}",
        "",
        *table(
            ["Model", "Unseen cells", "Later dates", "Later dates without label inputs"],
            [
                [name, s["unseen_cells_macro_f1"], s["later_dates_macro_f1"], s["later_dates_without_label_inputs_macro_f1"]]
                for name, s in evaluation["scores_by_model"].items()
            ],
        ),
        "",
        "**These scores don't show real-world accuracy.** Labels are calculated by a formula from three of the model's own inputs, so they show how well a model reproduces that formula. The last column removes those inputs, and is the more honest guide.",
        "",
        "### As labels become available",
        "",
        f"The {holdout['model_type']} model was trained only on {holdout['train_rows']:,} rows before {holdout['split_date']}, then scored on each later date, as if each date's labels had just arrived.",
        "",
        *table(
            ["Date", "Rows", "Macro F1", "Accuracy", "HIGH recall", "Classes present"],
            [[date, f"{s['rows']:,}", s["macro_f1"], s["accuracy"], s["high_stress_recall"], s["classes_present"]] for date, s in holdout["by_date"].items()],
        ),
        "",
        "Macro F1 covers only the classes present on that date.",
        "",
        "### By canal reach",
        "",
        "The data has no demographic variables, so results are compared along the canal, in three equal-width bands from the headworks in the west.",
        "",
        *table(
            ["Reach", "Longitude", "Rows", "Macro F1", "Accuracy", "HIGH recall"],
            [
                [name, f"{holdout['reach_longitude_bands'][name][0]} to {holdout['reach_longitude_bands'][name][1]}", f"{s['rows']:,}", s["macro_f1"], s["accuracy"], s["high_stress_recall"]]
                for name, s in holdout["by_reach"].items()
            ],
        ),
        "",
        f"The largest macro F1 gap between reaches is {holdout['largest_gap_between_reaches']}.",
        "",
        "## Business metric",
        "",
        f"**Hectares flagged for priority irrigation per cycle.** For the cycle of {cycle['cycle_date']}:",
        "",
        *table(
            ["Priority", "Hectares", "Share"],
            [
                ["1, urgent", f"{priority['priority_1_urgent_ha']:,}", f"{priority['priority_1_pct']}%"],
                ["2, scheduled", f"{priority['priority_2_scheduled_ha']:,}", f"{priority['priority_2_pct']}%"],
                ["3, normal", f"{priority['priority_3_normal_ha']:,}", f"{priority['priority_3_pct']}%"],
            ],
        ),
        "",
        "**This isn't validated against field outcomes.** No farmer or canal team has acted on these flags, so it counts what the system recommends, not water saved or crops protected. With most of the area in Priority 1, it can't yet help choose where water goes first. The total area also looks too large, because it exceeds the 66,250 ha culturable area in the study area file.",
        "",
        "## Serving performance",
        "",
        f"{load_test['users']} simulated users for {load_test['duration_s']} seconds against `{load_test['base_url']}`, model `{load_test['model_version']}`, measured {load_test['measured_at']}. {load_test['note']}",
        "",
        *table(
            ["Request", "Requests", "Errors", "Throughput (per second)", "p50 ms", "p95 ms", "p99 ms"],
            [
                [name, s["requests"], s["errors"], s["throughput_rps"], s["latency_ms"]["p50"], s["latency_ms"]["p95"], s["latency_ms"]["p99"]]
                for name, s in [("All", overall), *load_test["by_request"].items()]
            ],
        ),
        "",
        f"`/predict` includes a SHAP explanation for each request. Its p95 was {predict['latency_ms']['p95']} ms.",
        "",
        "## Drift",
        "",
        f"**Published date:** {drift['drifted_share']:.0%} of inputs drifted on {drift['current_date']}, against the period {drift['reference_period']}. {drift['note']}",
        "",
        f"**Simulated drift:** to show the monitor responds to a known change, two samples of {simulation['rows_per_sample']:,} rows from the training period were compared. This is simulated data, not a real result.",
        "",
        *table(
            ["Comparison", "Inputs drifted"],
            [
                ["Unchanged control", f"{simulation['control_drifted_share']:.0%}"],
                [f"Temperature +{simulation['shifts']['temperature_c']} °C, rainfall +{simulation['shifts']['rainfall_mm']} mm, NDWI {simulation['shifts']['ndwi']}", f"{simulation['drifted_share']:.0%}"],
            ],
        ),
        "",
        f"It flagged {len(simulation['shifted_inputs_flagged'])} of the {len(simulation['shifted_inputs'])} shifted inputs and {len(simulation['unshifted_inputs_flagged'])} unshifted ones.",
        "",
        f"The overall alert, at {simulation['drift_share_alert']:.0%} of inputs, {'fired' if simulation['alert_triggered'] else 'did not fire'} on this change."
        + ("" if simulation["alert_triggered"] else " Shifts in a minority of inputs can matter, so per-input results should be read alongside it."),
        "",
        "## Limits of these measurements",
        "",
        "- Everything ran on one computer, with the load test sharing its resources with the API, so these aren't production figures.",
        "- The data covers one season, from October 2023 to March 2024.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    output = REPORTS / "performance_report.md"
    output.write_text(build_report(), encoding="utf-8")
    print(f"Saved {output}")

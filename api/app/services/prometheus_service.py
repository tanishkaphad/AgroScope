"""AgroScope Prometheus metrics, scraped from /metrics/prometheus."""

from __future__ import annotations

import json
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram

from api.app.config import settings

REQUESTS = Counter(
    "agroscope_http_requests_total",
    "HTTP requests handled, by route and status code.",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "agroscope_http_request_duration_seconds",
    "Time taken to handle an HTTP request.",
    ["route"],
)
INFERENCE_LATENCY = Histogram(
    "agroscope_inference_duration_seconds",
    "Time taken by one model prediction, including its SHAP explanation.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "agroscope_predictions_total",
    "On-demand predictions served, by predicted stress class.",
    ["stress_class"],
)
PUBLISHED_HECTARES = Gauge(
    "agroscope_published_hectares",
    "Hectares in each stress class on the published map.",
    ["stress_class"],
)
DRIFT_SHARE = Gauge(
    "agroscope_drift_share",
    "Share of model inputs flagged as drifted by Evidently.",
    ["scenario"],
)
MODEL_LOADED = Gauge("agroscope_model_loaded", "1 when a model is loaded, otherwise 0.")


def _read_json(path: Path) -> dict:
    """Read a pipeline result, treating a missing file as no data."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def refresh_pipeline_gauges(model_loaded: bool) -> None:
    """Update the gauges from the pipeline's files, so new results show without a restart."""
    MODEL_LOADED.set(1 if model_loaded else 0)

    breakdown = _read_json(settings.CYCLE_STATS_JSON).get("stress_breakdown", {})
    for stress_class in ("low", "moderate", "high"):
        hectares = breakdown.get(f"{stress_class}_stress_hectares")
        if hectares is not None:
            PUBLISHED_HECTARES.labels(stress_class=stress_class.upper()).set(hectares)

    # A scenario with no file yet is left out, so Grafana shows "No data" and not a false zero.
    for scenario, path in (("published", settings.DRIFT_SUMMARY_JSON), ("simulated", settings.DRIFT_SIMULATION_JSON)):
        share = _read_json(path).get("drifted_share")
        if share is not None:
            DRIFT_SHARE.labels(scenario=scenario).set(share)

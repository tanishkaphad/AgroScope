"""Load-test the running API, and save p95 latency, throughput and error rate.

Simulated users repeat the requests the dashboard makes. Results show on the Grafana dashboard too.
Usage: python scripts/load_test.py --users 10 --duration 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICT_BODY = {"ndvi_mean": 0.3, "ndwi_mean": 0.0, "temp_max_c": 35.0, "rainfall_8day": 0.0, "et0_8day_sum": 40.0, "lang": "en"}
# (label, share of requests), in the rough proportions of a dashboard session
REQUEST_MIX = [("statistics", 0.3), ("cell_details", 0.3), ("predict", 0.3), ("map_sample", 0.1)]


def send_request(client: httpx.Client, label: str, cell_ids: List[str]) -> httpx.Response:
    if label == "statistics":
        return client.get("/statistics")
    if label == "cell_details":
        return client.get(f"/area/{random.choice(cell_ids)}")
    if label == "predict":
        return client.post("/predict", json=PREDICT_BODY)
    return client.get("/stress-map", params={"sample": "true", "limit": 1500})


def run_user(base_url: str, stop_at: float, cell_ids: List[str], seed: int, results: List[Tuple[str, float, bool]], lock: threading.Lock) -> None:
    """Send requests one after another until time runs out, as one user does."""
    rng = random.Random(seed)
    labels, weights = zip(*REQUEST_MIX)
    local: List[Tuple[str, float, bool]] = []
    with httpx.Client(base_url=base_url, timeout=60) as client:
        while time.perf_counter() < stop_at:
            label = rng.choices(labels, weights)[0]
            started = time.perf_counter()
            try:
                ok = send_request(client, label, cell_ids).status_code < 400
            except httpx.HTTPError:
                ok = False
            local.append((label, time.perf_counter() - started, ok))
    with lock:
        results.extend(local)


def summarise(samples: List[Tuple[str, float, bool]], elapsed: float) -> Dict[str, Any]:
    latencies = np.array([s[1] for s in samples]) * 1000
    errors = sum(not s[2] for s in samples)
    return {
        "requests": len(samples),
        "errors": errors,
        "error_rate": round(errors / len(samples), 4),
        "throughput_rps": round(len(samples) / elapsed, 2),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 1),
            "p95": round(float(np.percentile(latencies, 95)), 1),
            "p99": round(float(np.percentile(latencies, 99)), 1),
            "max": round(float(latencies.max()), 1),
        },
    }


def run_load_test(base_url: str, users: int, duration: int) -> Dict[str, Any]:
    with httpx.Client(base_url=base_url, timeout=60) as client:
        features = client.get("/stress-map", params={"sample": "true", "limit": 200}).raise_for_status().json()["features"]
        model = client.get("/health").raise_for_status().json()["model_version"]
    cell_ids = [f["properties"]["cell_id"] for f in features]

    results: List[Tuple[str, float, bool]] = []
    lock = threading.Lock()
    started = time.perf_counter()
    threads = [
        threading.Thread(target=run_user, args=(base_url, started + duration, cell_ids, seed, results, lock))
        for seed in range(users)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "model_version": model,
        "users": users,
        "duration_s": round(elapsed, 1),
        "note": "Run on one local computer, with the client and API sharing its resources.",
        "overall": summarise(results, elapsed),
        "by_request": {
            label: summarise([s for s in results if s[0] == label], elapsed)
            for label, _ in REQUEST_MIX
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--duration", type=int, default=60, help="Seconds to run")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "load_test_results.json")
    args = parser.parse_args()

    outcome = run_load_test(args.base_url, args.users, args.duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outcome, f, indent=2)
    print(json.dumps({"overall": outcome["overall"], "predict": outcome["by_request"]["predict"]}, indent=2))
    sys.exit(0)

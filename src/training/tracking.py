"""AgroScope MLflow Experiment Tracking Helpers.

Runs and model versions are stored locally, and `mlflow ui` shows them in a dashboard.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import mlflow
from mlflow.tracking import MlflowClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

TRACKING = settings["tracking"]
TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / TRACKING['database']).as_posix()}"
ARTIFACT_ROOT = (PROJECT_ROOT / TRACKING["artifact_dir"]).as_uri()


def get_client() -> MlflowClient:
    """Point MLflow at the local store and make sure the experiment exists."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    if client.get_experiment_by_name(TRACKING["experiment"]) is None:
        client.create_experiment(TRACKING["experiment"], artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(TRACKING["experiment"])
    return client


def file_md5(path: Path) -> str:
    """Identify the exact data version used for a run."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_training_run(
    params: Dict[str, object],
    metrics: Dict[str, float],
    tags: Dict[str, str],
    artifact_paths: List[Path],
) -> str:
    """Record a training run with its settings, scores, data version and model files."""
    get_client()
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="model_files")
        return run.info.run_id


def attach_run_artifact(run_id: str, path: Path) -> None:
    """Add a file to a finished run, such as metadata that names its own run."""
    get_client()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(path), artifact_path="model_files")

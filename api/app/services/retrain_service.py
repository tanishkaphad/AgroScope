"""AgroScope Retraining Job Service.

Runs the DVC training, promotion, prediction and drift stages in the background,
and reloads the API's model and results when they finish.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from api.app.config import settings
from api.app.services.geo_service import geo_service
from api.app.services.model_service import model_service

logger = logging.getLogger("agroscope.retrain_service")

RETRAIN_COMMAND = [sys.executable, "-m", "dvc", "repro", "--force", "--single-item", "train_evaluate"]
FOLLOW_UP_COMMAND = [sys.executable, "-m", "dvc", "repro", "monitor_drift"]


class RetrainService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: Dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None, "detail": None}

    def is_available(self) -> bool:
        """Retraining needs the DVC tools, which the deployed image leaves out."""
        return importlib.util.find_spec("dvc") is not None

    def start(self) -> bool:
        """Start a retraining job, unless one is already running."""
        with self._lock:
            if self.status["state"] == "running":
                return False
            self.status = {
                "state": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "detail": None,
            }
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        log_path = settings.DATA_DIR / "processed" / "retrain.log"
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                for command in (RETRAIN_COMMAND, FOLLOW_UP_COMMAND):
                    subprocess.run(command, cwd=settings.BASE_DIR, stdout=log, stderr=subprocess.STDOUT, check=True)
            model_service._load_predictor()
            geo_service.reload()
            state, detail = "completed", "Model and results reloaded."
        except Exception as exc:
            logger.error("Retraining failed: %s", exc)
            state, detail = "failed", f"Retraining failed. See {log_path.name}."

        self.status.update(state=state, detail=detail, finished_at=datetime.now(timezone.utc).isoformat())


retrain_service = RetrainService()

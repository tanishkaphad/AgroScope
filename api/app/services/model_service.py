"""AgroScope Model Service.

Manages model loading, single-instance prediction, SHAP explanation,
and parcel detail caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional
from pathlib import Path

from api.app.config import settings
from api.app.services.prometheus_service import INFERENCE_LATENCY, PREDICTIONS
from src.inference.predict import AgroScopePredictor

logger = logging.getLogger("agroscope.model_service")


class ModelService:
    _instance: Optional[ModelService] = None

    def __init__(self) -> None:
        self.predictor: Optional[AgroScopePredictor] = None
        self.version: Optional[str] = None
        self.total_inferences: int = 0
        self.start_time: float = time.time()
        self._load_predictor()

    @classmethod
    def get_instance(cls) -> ModelService:
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def _load_predictor(self) -> None:
        try:
            logger.info("Initializing AgroScopePredictor...")
            self.predictor = AgroScopePredictor(
                model_path=settings.MODEL_PATH,
                metadata_path=settings.METADATA_PATH,
            )
            digest = hashlib.md5(settings.MODEL_PATH.read_bytes()).hexdigest()
            self.version = f"{self.predictor.metadata.get('model_type', 'model')}-{digest[:8]}"
            logger.info("AgroScopePredictor initialized successfully.")
        except Exception as exc:
            logger.error("Failed to load AgroScopePredictor: %s", exc)
            self.predictor = None

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform on-demand inference and generate trilingual advisory."""
        if not self.predictor:
            raise RuntimeError("Model predictor is not initialized.")

        self.total_inferences += 1
        with INFERENCE_LATENCY.time():
            result = self.predictor.predict_instance(payload)
        PREDICTIONS.labels(stress_class=result["prediction"]["predicted_class_label"]).inc()
        return result

    def is_loaded(self) -> bool:
        return self.predictor is not None and self.predictor.model is not None

    def get_version(self) -> Optional[str]:
        """Identify the loaded model by a hash of its file."""
        return self.version if self.is_loaded() else None

    def get_scores(self) -> Dict[str, Optional[float]]:
        """Read the scores recorded for the model at training time."""
        if not settings.METADATA_PATH.exists():
            return {}
        with open(settings.METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model_type = metadata.get("model_type")
        return {
            "macro_f1_score": metadata.get("metrics", {}).get(model_type, {}).get("macro_f1"),
            "temporal_macro_f1": metadata.get("temporal_metrics", {}).get(model_type, {}).get("macro_f1"),
            "temporal_macro_f1_without_label_inputs": metadata.get("temporal_metrics_without_label_inputs", {}).get(model_type, {}).get("macro_f1"),
        }


model_service = ModelService.get_instance()

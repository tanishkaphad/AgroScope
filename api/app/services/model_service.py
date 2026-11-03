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
from src.inference.predict import AgroScopePredictor

logger = logging.getLogger("agroscope.model_service")


class ModelService:
    _instance: Optional[ModelService] = None

    def __init__(self) -> None:
        self.predictor: Optional[AgroScopePredictor] = None
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
            logger.info("AgroScopePredictor initialized successfully.")
        except Exception as exc:
            logger.error("Failed to load AgroScopePredictor: %s", exc)
            self.predictor = None

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform on-demand inference and generate trilingual advisory."""
        if not self.predictor:
            raise RuntimeError("Model predictor is not initialized.")

        self.total_inferences += 1
        return self.predictor.predict_instance(payload)

    def is_loaded(self) -> bool:
        return self.predictor is not None and self.predictor.model is not None

    def get_version(self) -> Optional[str]:
        """Identify the loaded model by a hash of its file."""
        if not self.is_loaded():
            return None
        digest = hashlib.md5(settings.MODEL_PATH.read_bytes()).hexdigest()
        return f"xgboost-{digest[:8]}"

    def get_macro_f1(self) -> Optional[float]:
        """Read the macro F1 recorded for the model at training time."""
        if not settings.METADATA_PATH.exists():
            return None
        with open(settings.METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("metrics", {}).get("xgboost", {}).get("macro_f1")


model_service = ModelService.get_instance()

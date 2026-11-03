"""AgroScope Model Service.

Manages model loading, single-instance prediction, SHAP explanation,
and parcel detail caching.
"""

from __future__ import annotations

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


model_service = ModelService.get_instance()

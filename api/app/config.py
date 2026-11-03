"""AgroScope API Configuration and Environment Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


class Settings:
    PROJECT_NAME: str = "AgroScope Satellite Water Stress & Advisory API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = (
        "Production REST API for satellite crop water stress estimation, "
        "explainable AI (SHAP), and irrigation priority advisories for canal command areas."
    )
    
    # Project Root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    
    # Specific artifact paths
    COMMAND_AREA_GEOJSON: Path = DATA_DIR / "raw" / "study_area" / "command_area.geojson"
    STRESS_MAP_GEOJSON: Path = DATA_DIR / "processed" / "stress_map.geojson"
    IRRIGATION_MAP_GEOJSON: Path = DATA_DIR / "processed" / "irrigation_map.geojson"
    CYCLE_STATS_JSON: Path = DATA_DIR / "processed" / "cycle_statistics.json"
    MODEL_PATH: Path = MODELS_DIR / "xgboost_model.pkl"
    METADATA_PATH: Path = MODELS_DIR / "feature_metadata.json"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://*.pages.dev",
        "https://*.up.railway.app",
        "*"
    ]
    
    # Supabase credentials (optional)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")


settings = Settings()

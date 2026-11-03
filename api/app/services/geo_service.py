"""AgroScope Geospatial Data Service.

Provides in-memory caching and rapid query capabilities for:
- Study area command boundary GeoJSON
- 8-day satellite cycle water stress choropleth maps
- Irrigation priority zones
- Specific parcel lookup by cell_id
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.app.config import settings

logger = logging.getLogger("agroscope.geo_service")


class GeoService:
    _instance: Optional[GeoService] = None

    def __init__(self) -> None:
        self.study_area: Optional[Dict[str, Any]] = None
        self.stress_map: Optional[Dict[str, Any]] = None
        self.irrigation_map: Optional[Dict[str, Any]] = None
        self.cycle_stats: Optional[Dict[str, Any]] = None
        self.cell_lookup: Dict[str, Dict[str, Any]] = {}
        self.irrigation_lookup: Dict[str, Dict[str, Any]] = {}
        self.available_dates: List[str] = []
        self._load_data()

    @classmethod
    def get_instance(cls) -> GeoService:
        if cls._instance is None:
            cls._instance = GeoService()
        return cls._instance

    def _load_data(self) -> None:
        # 1. Load Study Area Boundary
        if settings.COMMAND_AREA_GEOJSON.exists():
            try:
                with open(settings.COMMAND_AREA_GEOJSON, "r", encoding="utf-8") as f:
                    self.study_area = json.load(f)
                logger.info("Loaded study area boundary GeoJSON.")
            except Exception as exc:
                logger.error("Failed to load study area GeoJSON: %s", exc)

        # 2. Load Cycle Statistics
        if settings.CYCLE_STATS_JSON.exists():
            try:
                with open(settings.CYCLE_STATS_JSON, "r", encoding="utf-8") as f:
                    self.cycle_stats = json.load(f)
                logger.info("Loaded cycle statistics.")
            except Exception as exc:
                logger.error("Failed to load cycle statistics: %s", exc)

        # 3. Load Stress Map
        if settings.STRESS_MAP_GEOJSON.exists():
            try:
                logger.info("Loading stress map GeoJSON from disk...")
                with open(settings.STRESS_MAP_GEOJSON, "r", encoding="utf-8") as f:
                    self.stress_map = json.load(f)

                # Index features by cell_id for instant O(1) lookup
                features = self.stress_map.get("features", [])
                for feat in features:
                    props = feat.get("properties", {})
                    cid = props.get("cell_id")
                    if cid:
                        self.cell_lookup[cid] = feat
                self.available_dates = sorted({
                    feat["properties"]["date"] for feat in features if feat.get("properties", {}).get("date")
                })
                logger.info("Indexed %d stress map cells in memory.", len(self.cell_lookup))
            except Exception as exc:
                logger.error("Failed to load stress map GeoJSON: %s", exc)

        # 4. Load Irrigation Map
        if settings.IRRIGATION_MAP_GEOJSON.exists():
            try:
                logger.info("Loading irrigation priority map GeoJSON from disk...")
                with open(settings.IRRIGATION_MAP_GEOJSON, "r", encoding="utf-8") as f:
                    self.irrigation_map = json.load(f)
                for feat in self.irrigation_map.get("features", []):
                    cid = feat.get("properties", {}).get("cell_id")
                    if cid:
                        self.irrigation_lookup[cid] = feat
                logger.info("Loaded irrigation map GeoJSON.")
            except Exception as exc:
                logger.error("Failed to load irrigation map GeoJSON: %s", exc)

    def get_study_area(self) -> Optional[Dict[str, Any]]:
        return self.study_area

    def get_available_dates(self) -> List[str]:
        return self.available_dates

    def get_stress_map(self, sample: bool = True, limit: int = 1500) -> Optional[Dict[str, Any]]:
        """Return stress map GeoJSON.
        
        If sample=True, takes an evenly spaced sub-sample of cells to guarantee
        fast, 60fps Leaflet rendering without dropping overall spatial distribution.
        """
        if not self.stress_map:
            return None

        features = self.stress_map.get("features", [])
        if not sample or limit <= 0 or len(features) <= limit:
            return self.stress_map

        # Systematic stride sampling preserves geographic spread
        step = max(1, len(features) // limit)
        sampled_features = features[::step][:limit]

        return {
            "type": "FeatureCollection",
            "metadata": {
                "total_cells": len(features),
                "sampled_cells": len(sampled_features),
                "is_sampled": True,
            },
            "features": sampled_features,
        }

    def get_irrigation_map(self, sample: bool = True, limit: int = 1500) -> Optional[Dict[str, Any]]:
        """Return irrigation map GeoJSON with priority zones."""
        if not self.irrigation_map:
            return None

        features = self.irrigation_map.get("features", [])
        if not sample or limit <= 0 or len(features) <= limit:
            return self.irrigation_map

        step = max(1, len(features) // limit)
        sampled_features = features[::step][:limit]

        return {
            "type": "FeatureCollection",
            "metadata": {
                "total_cells": len(features),
                "sampled_cells": len(sampled_features),
                "is_sampled": True,
            },
            "features": sampled_features,
        }

    def get_statistics(self) -> Optional[Dict[str, Any]]:
        return self.cycle_stats

    def get_cell_detail(self, cell_id: str) -> Optional[Dict[str, Any]]:
        return self.cell_lookup.get(cell_id)

    def get_irrigation_detail(self, cell_id: str) -> Optional[Dict[str, Any]]:
        return self.irrigation_lookup.get(cell_id)

    def has_results(self) -> bool:
        return bool(self.stress_map and self.irrigation_map and self.cycle_stats)


geo_service = GeoService.get_instance()

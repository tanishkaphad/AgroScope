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

import pandas as pd

from api.app.config import settings
from src.advisory.advisory_engine import IrrigationAdvisoryEngine
from src.inference.predict import CLASS_LABELS, irrigation_properties, stress_properties

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
        self.results_by_date: Dict[str, pd.DataFrame] = {}
        self.statistics_by_date: Dict[str, Dict[str, Any]] = {}
        self.grid_geometry: Dict[str, Dict[str, Any]] = {}
        self._load_data()
        self._load_other_dates()

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

    def _load_other_dates(self) -> None:
        """Load results for every date. Without these files, only the published date is available."""
        paths = (settings.CYCLE_RESULTS_CSV, settings.CYCLE_STATS_BY_DATE_JSON, settings.GRID_GEOJSON)
        if not all(path.exists() for path in paths):
            logger.warning("Results for other dates are missing, so only the published date is available.")
            return
        try:
            results = pd.read_csv(settings.CYCLE_RESULTS_CSV)
            self.results_by_date = {date: group for date, group in results.groupby("date")}
            with open(settings.CYCLE_STATS_BY_DATE_JSON, "r", encoding="utf-8") as f:
                self.statistics_by_date = json.load(f)
            with open(settings.GRID_GEOJSON, "r", encoding="utf-8") as f:
                self.grid_geometry = {feat["properties"]["cell_id"]: feat["geometry"] for feat in json.load(f)["features"]}
            self.available_dates = sorted(set(self.available_dates) | set(self.results_by_date))
            logger.info("Loaded results for %d dates.", len(self.results_by_date))
        except Exception as exc:
            logger.error("Failed to load results for other dates: %s", exc)

    def has_date(self, date: str) -> bool:
        return date in self.available_dates

    def _is_other_date(self, date: Optional[str]) -> bool:
        """The published date is served from its full map files, and any other from the results table."""
        return date is not None and date != (self.cycle_stats or {}).get("cycle_date") and date in self.results_by_date

    @staticmethod
    def _stress_feature_properties(row: Any) -> Dict[str, Any]:
        return stress_properties(row.cell_id, row.date, CLASS_LABELS[row.stress_code], row.confidence, row.ndvi_mean, row.ndwi_mean)

    @staticmethod
    def _irrigation_feature_properties(row: Any) -> Dict[str, Any]:
        advisory = IrrigationAdvisoryEngine.evaluate_cell(
            stress_class=row.stress_code,
            confidence=row.confidence,
            ndwi_mean=row.ndwi_mean,
            temp_max_c=row.temp_max_c,
            et0_8day_sum=row.et0_8day_sum,
            rainfall_8day=row.rainfall_8day,
        )
        return irrigation_properties(row.cell_id, row.date, advisory, row.confidence, row.ndvi_mean, row.ndwi_mean)

    def _map_for_date(self, date: str, build_properties: Any, sample: bool, limit: int) -> Dict[str, Any]:
        """Build one date's map on request. Sampling matches the published map's, so it covers the area evenly."""
        rows = self.results_by_date[date]
        total = len(rows)
        if sample and limit > 0 and total > limit:
            rows = rows.iloc[:: max(1, total // limit)].iloc[:limit]

        features = [
            {"type": "Feature", "geometry": self.grid_geometry[row.cell_id], "properties": build_properties(row)}
            for row in rows.itertuples(index=False)
            if row.cell_id in self.grid_geometry
        ]
        return {
            "type": "FeatureCollection",
            "metadata": {"total_cells": total, "sampled_cells": len(features), "is_sampled": len(features) < total},
            "features": features,
        }

    def _cell_for_date(self, cell_id: str, date: str, build_properties: Any) -> Optional[Dict[str, Any]]:
        rows = self.results_by_date[date]
        match = rows[rows["cell_id"] == cell_id]
        if match.empty or cell_id not in self.grid_geometry:
            return None
        row = next(match.itertuples(index=False))
        return {"type": "Feature", "geometry": self.grid_geometry[cell_id], "properties": build_properties(row)}

    def reload(self) -> None:
        """Replace cached results with the latest files."""
        self.__init__()

    def get_study_area(self) -> Optional[Dict[str, Any]]:
        return self.study_area

    def get_available_dates(self) -> List[str]:
        return self.available_dates

    def get_stress_map(self, sample: bool = True, limit: int = 1500, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return stress map GeoJSON.
        
        If sample=True, takes an evenly spaced sub-sample of cells to guarantee
        fast, 60fps Leaflet rendering without dropping overall spatial distribution.
        """
        if self._is_other_date(date):
            return self._map_for_date(date, self._stress_feature_properties, sample, limit)
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

    def get_irrigation_map(self, sample: bool = True, limit: int = 1500, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return irrigation map GeoJSON with priority zones."""
        if self._is_other_date(date):
            return self._map_for_date(date, self._irrigation_feature_properties, sample, limit)
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

    def get_statistics(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._is_other_date(date):
            return self.statistics_by_date.get(date)
        return self.cycle_stats

    def get_cell_detail(self, cell_id: str, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._is_other_date(date):
            return self._cell_for_date(cell_id, date, self._stress_feature_properties)
        return self.cell_lookup.get(cell_id)

    def get_irrigation_detail(self, cell_id: str, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._is_other_date(date):
            return self._cell_for_date(cell_id, date, self._irrigation_feature_properties)
        return self.irrigation_lookup.get(cell_id)

    def has_results(self) -> bool:
        return bool(self.stress_map and self.irrigation_map and self.cycle_stats)


geo_service = GeoService.get_instance()

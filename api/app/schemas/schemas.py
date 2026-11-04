"""Pydantic Request and Response Schemas for AgroScope REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="HEALTHY")
    database: str = Field(..., example="CONNECTED")
    model: str = Field(..., example="LOADED")
    model_version: Optional[str] = Field(None, example="lightgbm-1a2b3c4d")
    timestamp: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_inferences: int
    active_model: Optional[str] = None
    macro_f1_score: Optional[float] = None
    temporal_macro_f1: Optional[float] = None
    temporal_macro_f1_without_label_inputs: Optional[float] = None
    score_note: str
    memory_rss_mb: float
    timestamp: str


class KeyDriver(BaseModel):
    feature: str
    label: str
    value: float
    impact_pct: float
    direction: str
    arrow: str
    summary: str


class AdvisoryResponse(BaseModel):
    priority: int = Field(..., example=1)
    color: str = Field(..., example="#dc3545")
    urgency: str = Field(..., example="Immediate (Next 24-48 Hours)")
    title: str = Field(..., example="CRITICAL: Immediate Irrigation Required")
    recommendation: str
    actionable_steps: List[str]
    translations: Optional[Dict[str, Any]] = None


class AreaDetailResponse(BaseModel):
    cell_id: str
    date: str
    crop_type: Optional[str] = None
    dominant_soil: Optional[str] = None
    water_stress: Dict[str, Any]
    irrigation_advisory: AdvisoryResponse
    key_drivers: List[KeyDriver]
    canopy_indices: Dict[str, float]


class PredictRequest(BaseModel):
    ndvi_mean: float = Field(0.28, ge=-1.0, le=1.0, description="Vegetation Greenness Index")
    ndwi_mean: float = Field(0.04, ge=-1.0, le=1.0, description="Canopy Water Content Index")
    evi_mean: Optional[float] = Field(0.22, description="Enhanced Vegetation Index")
    temp_max_c: float = Field(34.0, ge=0.0, le=55.0, description="Peak Daily Temperature (°C)")
    temp_mean_c: Optional[float] = Field(28.0, description="Mean Daily Temperature (°C)")
    rainfall_8day: float = Field(0.0, ge=0.0, description="8-Day Cumulative Rainfall (mm)")
    rainfall_16day: Optional[float] = Field(0.0, description="16-Day Cumulative Rainfall (mm)")
    humidity_pct: Optional[float] = Field(42.0, ge=0.0, le=100.0, description="Atmospheric Humidity (%)")
    solar_radiation_mj: Optional[float] = Field(22.0, description="Solar Radiation (MJ/m²)")
    et0_8day_sum: Optional[float] = Field(38.0, description="8-Day Evapotranspiration Demand (mm)")
    crop_type: Optional[str] = Field("Sugarcane", description="Crop Type")
    lang: Optional[str] = Field("en", description="Advisory language ('en', 'mr', 'hi')")


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_class_code: int
    confidence: float
    class_probabilities: Dict[str, float]
    advisory: AdvisoryResponse
    key_drivers: List[KeyDriver]
    note: str


class StatisticsResponse(BaseModel):
    cycle_date: str
    total_culturable_cells: int
    active_cultivated_cells: int
    fallow_or_bare_cells: int
    total_cultivated_hectares: float
    stress_breakdown: Dict[str, Any]
    irrigation_priority_breakdown: Dict[str, Any]
    mean_canopy_metrics: Dict[str, float]

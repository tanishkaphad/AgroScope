"""Geospatial Layers and Spatial Unit Endpoints."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from api.app.services.geo_service import geo_service
from api.app.services.metrics_service import metrics_service

router = APIRouter(tags=["Geospatial Layers"])


@router.get("/study-area")
async def get_study_area_boundary():
    """Retrieve official polygon boundary of the canal command area."""
    metrics_service.increment_request()
    data = geo_service.get_study_area()
    if data is None:
        raise HTTPException(status_code=503, detail="Study area boundary is not available.")
    return data


@router.get("/available-dates", response_model=List[str])
async def get_available_cycle_dates():
    """List processed 8-day satellite observation dates."""
    metrics_service.increment_request()
    return geo_service.get_available_dates()


@router.get("/stress-map")
async def get_water_stress_map(
    date: Optional[str] = Query(None, description="Cycle date (e.g. '2023-12-10')"),
    sample: bool = Query(True, description="Enable systematic spatial sampling for fast browser rendering"),
    limit: int = Query(1500, ge=0, le=25000, description="Max cells to return (0 for all cells)"),
):
    """Retrieve GeoJSON FeatureCollection of water stress predictions."""
    metrics_service.increment_request()
    if date and date not in geo_service.get_available_dates():
        raise HTTPException(status_code=404, detail=f"No stress map available for {date}.")
    data = geo_service.get_stress_map(sample=sample, limit=limit)
    if data is None:
        raise HTTPException(status_code=503, detail="Stress map results are not available.")
    return JSONResponse(content=data)


@router.get("/irrigation-map")
async def get_irrigation_priority_map(
    sample: bool = Query(True, description="Enable systematic spatial sampling for fast browser rendering"),
    limit: int = Query(1500, ge=0, le=25000, description="Max cells to return (0 for all cells)"),
):
    """Retrieve GeoJSON FeatureCollection of irrigation priority zones (P1, P2, P3)."""
    metrics_service.increment_request()
    data = geo_service.get_irrigation_map(sample=sample, limit=limit)
    if data is None:
        raise HTTPException(status_code=503, detail="Irrigation map results are not available.")
    return JSONResponse(content=data)

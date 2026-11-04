"""Summary Statistics and High-Level KPI Endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from api.app.schemas.schemas import StatisticsResponse
from api.app.services.geo_service import geo_service
from api.app.services.metrics_service import metrics_service

router = APIRouter(tags=["Analytics & KPIs"])


@router.get("/statistics", response_model=StatisticsResponse)
async def get_cycle_statistics(date: Optional[str] = Query(None, description="Cycle date, or the published date if left out")):
    """Retrieve aggregate statistics: hectares under High/Moderate/Low stress and priority breakdown."""
    metrics_service.increment_request()
    if date and not geo_service.has_date(date):
        raise HTTPException(status_code=404, detail=f"No statistics available for {date}.")
    stats = geo_service.get_statistics(date)
    if stats is None:
        raise HTTPException(status_code=503, detail="Cycle statistics are not available.")
    return stats

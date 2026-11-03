"""Summary Statistics and High-Level KPI Endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from api.app.schemas.schemas import StatisticsResponse
from api.app.services.geo_service import geo_service
from api.app.services.metrics_service import metrics_service

router = APIRouter(tags=["Analytics & KPIs"])


@router.get("/statistics", response_model=StatisticsResponse)
async def get_cycle_statistics():
    """Retrieve aggregate statistics: hectares under High/Moderate/Low stress and priority breakdown."""
    metrics_service.increment_request()
    return geo_service.get_statistics()

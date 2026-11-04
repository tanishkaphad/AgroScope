"""Health, Liveness, and Metrics Monitoring Endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from api.app.schemas.schemas import HealthResponse, MetricsResponse
from api.app.services.geo_service import geo_service
from api.app.services.model_service import model_service
from api.app.services.metrics_service import metrics_service
from api.app.services.prometheus_service import refresh_pipeline_gauges

router = APIRouter(tags=["Monitoring & Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness and readiness probe for UptimeRobot and Railway."""
    metrics_service.increment_request()
    model_loaded = model_service.is_loaded()

    return HealthResponse(
        status="HEALTHY" if model_loaded and geo_service.has_results() else "DEGRADED",
        database="NOT_CONNECTED (LOCAL FILES)",
        model="LOADED" if model_loaded else "FAILED_TO_LOAD",
        model_version=model_service.get_version(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics():
    """Application performance metrics including latency, memory, and inference counts."""
    metrics_service.increment_request()
    sys_stats = metrics_service.get_metrics()

    return MetricsResponse(
        uptime_seconds=sys_stats["uptime_seconds"],
        total_inferences=sys_stats["total_inferences"],
        active_model=model_service.get_version(),
        **model_service.get_scores(),
        score_note="Measured against formula-based labels, not field observations.",
        memory_rss_mb=sys_stats["memory_rss_mb"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/metrics/prometheus", include_in_schema=False)
async def get_prometheus_metrics() -> Response:
    """Metrics in Prometheus format. The JSON /metrics endpoint is separate and unchanged."""
    refresh_pipeline_gauges(model_service.is_loaded())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

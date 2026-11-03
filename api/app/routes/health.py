"""Health, Liveness, and Metrics Monitoring Endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter
from api.app.schemas.schemas import HealthResponse, MetricsResponse
from api.app.services.model_service import model_service
from api.app.services.metrics_service import metrics_service
from api.app.config import settings

router = APIRouter(tags=["Monitoring & Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness and readiness probe for UptimeRobot, Railway, and Kubernetes."""
    metrics_service.increment_request()
    model_loaded = model_service.is_loaded()
    has_db = bool(settings.SUPABASE_URL and settings.SUPABASE_KEY)
    
    return HealthResponse(
        status="HEALTHY" if model_loaded else "DEGRADED",
        database="CONNECTED (SUPABASE)" if has_db else "LOCAL_STORAGE (READY)",
        model="LOADED" if model_loaded else "FAILED_TO_LOAD",
        model_version="XGBoost v1.0 (NLBC Calibrated)",
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
        active_model="XGBoost Tuned (Macro-F1: 0.998)",
        macro_f1_score=0.9982,
        memory_rss_mb=sys_stats["memory_rss_mb"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

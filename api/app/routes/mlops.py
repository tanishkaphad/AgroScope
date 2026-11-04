"""MLOps, Model Drift Monitoring, and Retraining Endpoints."""

from __future__ import annotations

import json
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from api.app.config import settings
from api.app.services.metrics_service import metrics_service
from api.app.services.retrain_service import retrain_service

router = APIRouter(tags=["MLOps & Continuous Monitoring"])


@router.get("/monitoring/drift")
async def get_drift_status() -> Dict[str, Any]:
    """Return the latest Evidently drift summary."""
    metrics_service.increment_request()
    if not settings.DRIFT_SUMMARY_JSON.exists():
        raise HTTPException(status_code=503, detail="No drift report is available yet.")
    with open(settings.DRIFT_SUMMARY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/monitoring/drift/report")
async def get_drift_report() -> FileResponse:
    """Return the full Evidently HTML report."""
    metrics_service.increment_request()
    if not settings.DRIFT_REPORT_HTML.exists():
        raise HTTPException(status_code=503, detail="No drift report is available yet.")
    return FileResponse(settings.DRIFT_REPORT_HTML, media_type="text/html")


@router.get("/monitoring/promotion")
async def get_promotion_decision() -> Dict[str, Any]:
    """Return the latest champion-challenger decision."""
    metrics_service.increment_request()
    if not settings.PROMOTION_DECISION_JSON.exists():
        raise HTTPException(status_code=503, detail="No promotion decision is available yet.")
    with open(settings.PROMOTION_DECISION_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/retrain", status_code=202)
async def trigger_retraining(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Start retraining through the promotion gate. Requires the admin token."""
    metrics_service.increment_request()
    if not settings.RETRAIN_TOKEN:
        raise HTTPException(status_code=503, detail="Retraining is disabled because RETRAIN_TOKEN isn't set.")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.RETRAIN_TOKEN):
        raise HTTPException(status_code=403, detail="A valid admin token is required.")
    if not retrain_service.is_available():
        raise HTTPException(status_code=501, detail="Retraining isn't available in this deployment.")
    if not retrain_service.start():
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    return retrain_service.status


@router.get("/retrain/status")
async def get_retraining_status() -> Dict[str, Any]:
    """Return the state of the latest retraining job."""
    metrics_service.increment_request()
    return retrain_service.status

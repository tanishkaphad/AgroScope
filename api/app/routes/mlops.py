"""MLOps, Model Drift Monitoring, and Retraining Endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from api.app.services.metrics_service import metrics_service

router = APIRouter(tags=["MLOps & Continuous Monitoring"])


@router.get("/monitoring/drift")
async def get_drift_status():
    """Drift monitoring is not built yet."""
    metrics_service.increment_request()
    raise HTTPException(status_code=501, detail="Drift monitoring is not implemented yet.")


@router.post("/retrain")
async def trigger_retraining():
    """Automated retraining is not built yet."""
    metrics_service.increment_request()
    raise HTTPException(status_code=501, detail="Automated retraining is not implemented yet.")

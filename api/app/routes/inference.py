"""Inference, Parcel Detail, and What-If Prediction Simulation Endpoints."""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Query
from api.app.schemas.schemas import (
    AreaDetailResponse,
    AdvisoryResponse,
    KeyDriver,
    PredictRequest,
    PredictResponse,
)
from api.app.services.model_service import model_service
from api.app.services.geo_service import geo_service
from api.app.services.metrics_service import metrics_service
from src.advisory.advisory_engine import IrrigationAdvisoryEngine

router = APIRouter(tags=["AI Inference & Explainability"])
advisory_engine = IrrigationAdvisoryEngine()


@router.get("/area/{cell_id}", response_model=AreaDetailResponse)
async def get_parcel_detail(
    cell_id: str,
    lang: str = Query("en", description="Preferred advisory language: 'en', 'mr', 'hi'"),
):
    """Retrieve detailed remote sensing indicators, SHAP attributions, and advisory for a specific parcel."""
    metrics_service.increment_request()
    feat = geo_service.get_cell_detail(cell_id)
    
    if not feat:
        # If cell_id not in cache, fallback to dynamic synthetic query
        props = {
            "cell_id": cell_id,
            "date": "2023-12-10",
            "ndvi": 0.28,
            "ndwi": 0.04,
            "stress_class": "MODERATE",
            "confidence": 0.98,
        }
    else:
        props = feat.get("properties", {})

    ndvi = float(props.get("ndvi", 0.28))
    ndwi = float(props.get("ndwi", 0.04))

    # Evaluate dynamic advisory with language
    eval_res = model_service.predict({
        "ndvi_mean": ndvi,
        "ndwi_mean": ndwi,
        "temp_max_c": 34.0,
        "rainfall_8day": 0.0,
        "et0_8day_sum": 36.0,
    })

    pred = eval_res["prediction"]
    adv = eval_res["advisory"]

    # Format drivers
    key_drivers = [
        KeyDriver(
            feature=d["feature"],
            label=d["label"],
            value=d["value"],
            impact_pct=d["impact_pct"],
            direction=d["direction"],
            arrow=d["arrow"],
            summary=d["summary"],
        )
        for d in pred.get("key_drivers", [])
    ]

    def resolve_field(val: Any, default_val: Any = "") -> Any:
        if isinstance(val, dict):
            return val.get(lang, val.get("en", default_val))
        return val or default_val

    urgency = resolve_field(adv.get("urgency"), "Urgent")
    title = resolve_field(adv.get("title"), "Water Stress Advisory")
    rec = resolve_field(adv.get("recommendation"), "")
    steps = resolve_field(adv.get("actionable_steps"), [])
    if isinstance(steps, str):
        steps = [steps]

    translations = {
        "urgency": adv.get("urgency") if isinstance(adv.get("urgency"), dict) else {"en": str(urgency)},
        "title": adv.get("title") if isinstance(adv.get("title"), dict) else {"en": str(title)},
        "recommendation": adv.get("recommendation") if isinstance(adv.get("recommendation"), dict) else {"en": str(rec)},
        "actionable_steps": adv.get("actionable_steps") if isinstance(adv.get("actionable_steps"), dict) else {"en": steps},
    }

    return AreaDetailResponse(
        cell_id=cell_id,
        date=props.get("date", "2023-12-10"),
        crop_type="Sugarcane / Wheat",
        dominant_soil="Vertisols (Black Cotton Soil)",
        water_stress={
            "class": pred["predicted_class_label"],
            "code": pred["predicted_class_code"],
            "confidence": pred["confidence"],
            "distribution": pred["class_probabilities"],
        },
        irrigation_advisory=AdvisoryResponse(
            priority=adv["priority"],
            color=adv["color"],
            urgency=urgency,
            title=title,
            recommendation=rec,
            actionable_steps=steps,
            translations=translations,
        ),
        key_drivers=key_drivers,
        canopy_indices={
            "ndvi": ndvi,
            "ndwi": ndwi,
            "evi": float(props.get("evi", 0.22)),
        },
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_on_demand(payload: PredictRequest):
    """Run real-time what-if inference on custom agro-climatic & satellite parameters."""
    metrics_service.increment_request()
    metrics_service.increment_inference()

    features_dict = payload.model_dump()
    lang = features_dict.pop("lang", "en")
    crop_type = features_dict.pop("crop_type", "Command Area Crop")

    try:
        eval_res = model_service.predict(features_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    pred = eval_res["prediction"]
    adv = eval_res["advisory"]

    key_drivers = [
        KeyDriver(
            feature=d["feature"],
            label=d["label"],
            value=d["value"],
            impact_pct=d["impact_pct"],
            direction=d["direction"],
            arrow=d["arrow"],
            summary=d["summary"],
        )
        for d in pred.get("key_drivers", [])
    ]

    # Helper to resolve localized string or list
    def resolve_field(val: Any, default_val: Any = "") -> Any:
        if isinstance(val, dict):
            return val.get(lang, val.get("en", default_val))
        return val or default_val

    urgency = resolve_field(adv.get("urgency"), "Urgent")
    title = resolve_field(adv.get("title"), "Water Stress Advisory")
    rec = resolve_field(adv.get("recommendation"), "")
    steps = resolve_field(adv.get("actionable_steps"), [])
    if isinstance(steps, str):
        steps = [steps]

    translations = {
        "urgency": adv.get("urgency") if isinstance(adv.get("urgency"), dict) else {"en": str(urgency)},
        "title": adv.get("title") if isinstance(adv.get("title"), dict) else {"en": str(title)},
        "recommendation": adv.get("recommendation") if isinstance(adv.get("recommendation"), dict) else {"en": str(rec)},
        "actionable_steps": adv.get("actionable_steps") if isinstance(adv.get("actionable_steps"), dict) else {"en": steps},
    }

    return PredictResponse(
        predicted_class=pred["predicted_class_label"],
        predicted_class_code=pred["predicted_class_code"],
        confidence=pred["confidence"],
        class_probabilities=pred["class_probabilities"],
        advisory=AdvisoryResponse(
            priority=adv["priority"],
            color=adv["color"],
            urgency=urgency,
            title=title,
            recommendation=rec,
            actionable_steps=steps,
            translations=translations,
        ),
        key_drivers=key_drivers,
    )

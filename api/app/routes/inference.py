"""Inference, Parcel Detail, and What-If Prediction Simulation Endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional
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

router = APIRouter(tags=["AI Inference & Explainability"])


@router.get("/area/{cell_id}", response_model=AreaDetailResponse)
async def get_parcel_detail(
    cell_id: str,
    lang: str = Query("en", description="Preferred advisory language: 'en', 'mr', 'hi'"),
    date: Optional[str] = Query(None, description="Cycle date, or the published date if left out"),
):
    """Retrieve the stored stress prediction and irrigation advisory for a specific parcel."""
    metrics_service.increment_request()

    if not geo_service.has_results():
        raise HTTPException(status_code=503, detail="Prediction results are not available.")

    if date and not geo_service.has_date(date):
        raise HTTPException(status_code=404, detail=f"No results available for {date}.")

    stress_feat = geo_service.get_cell_detail(cell_id, date)
    irrigation_feat = geo_service.get_irrigation_detail(cell_id, date)
    if not stress_feat or not irrigation_feat:
        raise HTTPException(status_code=404, detail=f"No prediction available for cell {cell_id}{f' on {date}' if date else ''}.")

    stress = stress_feat["properties"]
    irrigation = irrigation_feat["properties"]

    translations = {
        field: {code: irrigation[f"{field}_{code}"] for code in ("en", "mr", "hi")}
        for field in ("urgency", "title", "recommendation", "actionable_steps")
    }
    language = lang if lang in ("en", "mr", "hi") else "en"

    return AreaDetailResponse(
        cell_id=cell_id,
        date=stress["date"],
        water_stress={
            "class": stress["stress_class"],
            "confidence": stress["confidence"],
        },
        irrigation_advisory=AdvisoryResponse(
            priority=irrigation["priority"],
            color=irrigation["fill_color"],
            urgency=translations["urgency"][language],
            title=translations["title"][language],
            recommendation=translations["recommendation"][language],
            actionable_steps=translations["actionable_steps"][language],
            translations=translations,
        ),
        key_drivers=[],
        canopy_indices={
            "ndvi": stress["ndvi"],
            "ndwi": stress["ndwi"],
        },
    )


@router.post("/predict", response_model=PredictResponse)
def predict_on_demand(payload: PredictRequest):
    """Run real-time what-if inference on custom agro-climatic & satellite parameters.

    A plain function, so FastAPI runs it in a worker thread and slow inference can't block other requests.
    """
    metrics_service.increment_request()

    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not available.")

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
        note="Simulation only. Inputs not supplied are filled with fixed values, so don't use this result for water decisions.",
    )

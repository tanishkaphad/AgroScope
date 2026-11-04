"""AgroScope Inference & Geospatial Prediction Pipeline.

Implements Phases 13, 14, 15, and 16 of the AgroScope workflow:
1. Loads production XGBoost model from models/xgboost_model.pkl.
2. Performs batch inference over all 250m spatial units for any 8-day satellite cycle.
3. Attaches rule-based irrigation priority advisories (Priority 1, 2, 3).
4. Generates query-ready geospatial GeoJSONs:
   - data/processed/stress_map.geojson
   - data/processed/irrigation_map.geojson
5. Computes high-level KPI area statistics (hectares under Priority 1/2/3).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advisory.advisory_engine import IrrigationAdvisoryEngine
from src.config import settings
from src.inference.explainability import ModelExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.inference")

CELL_AREA_HECTARES = settings["aoi"]["cell_area_ha"]
MIN_LATEST_COVERAGE = settings["inference"]["min_latest_coverage"]  # "latest" skips dates where a partial scene covers less of the grid


class AgroScopePredictor:
    """Manages batch and on-demand crop water stress inference and advisory generation."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.model_path = model_path or (base_dir / "models" / "xgboost_model.pkl")
        self.metadata_path = metadata_path or (base_dir / "models" / "feature_metadata.json")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info("Loading inference model from: %s", self.model_path)
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_names: List[str] = self.metadata["features"]
        self.class_labels: List[str] = self.metadata.get("class_labels", ["LOW", "MODERATE", "HIGH"])
        self.advisory_engine = IrrigationAdvisoryEngine()
        self.explainer = ModelExplainer(model_path=self.model_path, metadata_path=self.metadata_path)

    def predict_instance(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """Predict stress, priority advisory, and key drivers for a single unit or custom payload."""
        explanation = self.explainer.explain_instance(features_dict)
        pred_class = explanation["predicted_class_code"]
        confidence = explanation["confidence"]

        ndwi = float(features_dict.get("ndwi_mean", 0.05))
        temp_max = float(features_dict.get("temp_max_c", 32.0))
        et0_sum = float(features_dict.get("et0_8day_sum", 35.0))
        rain_sum = float(features_dict.get("rainfall_8day", 0.0))

        advisory = self.advisory_engine.evaluate_cell(
            stress_class=pred_class,
            confidence=confidence,
            ndwi_mean=ndwi,
            temp_max_c=temp_max,
            et0_8day_sum=et0_sum,
            rainfall_8day=rain_sum,
        )

        return {
            "prediction": explanation,
            "advisory": advisory,
        }

    def run_cycle_inference(
        self,
        date_str: Optional[str] = None,
        lang: str = "en",
        features_csv: Optional[Path] = None,
        grid_geojson: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run batch inference for a specific 8-day cycle date and export GeoJSON maps."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        features_csv = features_csv or (base_dir / "data" / "features" / "features.csv")
        grid_geojson = grid_geojson or (base_dir / "data" / "processed" / "spatial_grid_250m.geojson")
        output_dir = output_dir or (base_dir / "data" / "processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading unified dataset from: %s", features_csv)
        df = pd.read_csv(features_csv)

        # Available observation dates
        available_dates = sorted(df["date"].unique().tolist())
        if not available_dates:
            raise ValueError("No observation dates found in features dataset.")

        if date_str is None or date_str == "latest":
            cells_per_date = df.groupby("date")["cell_id"].nunique()
            covered_dates = cells_per_date[cells_per_date >= cells_per_date.max() * MIN_LATEST_COVERAGE].index
            target_date = max(covered_dates)
            logger.info("Selecting latest cycle date with full coverage: %s", target_date)
        else:
            if date_str not in available_dates:
                raise ValueError(f"Date '{date_str}' not in dataset. Available dates: {available_dates}")
            target_date = date_str

        # Filter to cycle date
        cycle_df = df[df["date"] == target_date].copy()
        total_cells = len(cycle_df)
        logger.info("Cycle %s contains %d spatial cells.", target_date, total_cells)

        # Separate vegetated crops from fallow/bare soil
        is_veg_mask = cycle_df["is_vegetated"] == True if "is_vegetated" in cycle_df.columns else (cycle_df["ndvi_mean"] >= settings["labels"]["min_vegetated_ndvi"])
        veg_df = cycle_df[is_veg_mask].copy()
        logger.info("Active crop canopy cells for inference: %d (%.1f%%)", len(veg_df), (len(veg_df) / total_cells) * 100)

        # Batch Model Prediction
        X = veg_df[self.feature_names]
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        veg_df["pred_stress_class"] = preds
        veg_df["pred_stress_label"] = [self.class_labels[p] for p in preds]
        veg_df["confidence"] = np.max(probs, axis=1).round(3)

        # Batch Advisory Assignment
        priorities = []
        advisories_raw = []
        colors = []
        for idx, row in veg_df.iterrows():
            adv = self.advisory_engine.evaluate_cell(
                stress_class=row["pred_stress_class"],
                confidence=row["confidence"],
                ndwi_mean=row["ndwi_mean"],
                temp_max_c=row.get("temp_max_c", 32.0),
                et0_8day_sum=row.get("et0_8day_sum", 35.0),
                rainfall_8day=row.get("rainfall_8day", 0.0),
            )
            priorities.append(adv["priority"])
            colors.append(adv["color"])
            advisories_raw.append(adv)

        veg_df["irrigation_priority"] = priorities
        veg_df["priority_color"] = colors

        # Join predictions by cell_id into a lookup dict
        pred_lookup: Dict[str, Dict[str, Any]] = {}
        for idx, (_, row) in enumerate(veg_df.iterrows()):
            cid = row["cell_id"]
            adv = advisories_raw[idx]
            pred_lookup[cid] = {
                "stress_class": row["pred_stress_label"],
                "stress_code": int(row["pred_stress_class"]),
                "confidence": float(row["confidence"]),
                "priority": int(row["irrigation_priority"]),
                "color": row["priority_color"],
                "ndvi": round(float(row["ndvi_mean"]), 3),
                "ndwi": round(float(row["ndwi_mean"]), 3),
                "urgency": adv["urgency"],
                "title": adv["title"],
                "recommendation": adv["recommendation"],
                "actionable_steps": adv["actionable_steps"],
            }

        # Load grid geometries
        logger.info("Reading spatial geometries from: %s", grid_geojson)
        with open(grid_geojson, "r", encoding="utf-8") as f:
            grid_data = json.load(f)

        stress_features: List[Dict[str, Any]] = []
        irrigation_features: List[Dict[str, Any]] = []

        stress_colors = {
            "LOW": "#28a745",        # Green
            "MODERATE": "#ffc107",   # Yellow
            "HIGH": "#dc3545",       # Red
        }

        for feat in grid_data.get("features", []):
            cid = feat.get("properties", {}).get("cell_id")
            if cid in pred_lookup:
                info = pred_lookup[cid]

                # Stress Map Feature
                s_feat = {
                    "type": "Feature",
                    "geometry": feat["geometry"],
                    "properties": {
                        "cell_id": cid,
                        "date": target_date,
                        "stress_class": info["stress_class"],
                        "confidence": info["confidence"],
                        "fill_color": stress_colors[info["stress_class"]],
                        "ndvi": info["ndvi"],
                        "ndwi": info["ndwi"],
                    },
                }
                stress_features.append(s_feat)

                # Irrigation Priority Feature (Multilingual properties)
                i_feat = {
                    "type": "Feature",
                    "geometry": feat["geometry"],
                    "properties": {
                        "cell_id": cid,
                        "date": target_date,
                        "priority": info["priority"],
                        "priority_label": f"Priority {info['priority']}",
                        "fill_color": info["color"],
                        "confidence": info["confidence"],
                        "ndvi": info["ndvi"],
                        "ndwi": info["ndwi"],
                        "urgency_en": info["urgency"]["en"],
                        "urgency_mr": info["urgency"]["mr"],
                        "urgency_hi": info["urgency"]["hi"],
                        "title_en": info["title"]["en"],
                        "title_mr": info["title"]["mr"],
                        "title_hi": info["title"]["hi"],
                        "recommendation_en": info["recommendation"]["en"],
                        "recommendation_mr": info["recommendation"]["mr"],
                        "recommendation_hi": info["recommendation"]["hi"],
                        "actionable_steps_en": info["actionable_steps"]["en"],
                        "actionable_steps_mr": info["actionable_steps"]["mr"],
                        "actionable_steps_hi": info["actionable_steps"]["hi"],
                    },
                }
                irrigation_features.append(i_feat)

        stress_geojson_path = output_dir / "stress_map.geojson"
        with open(stress_geojson_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": stress_features}, f)

        irrigation_geojson_path = output_dir / "irrigation_map.geojson"
        with open(irrigation_geojson_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": irrigation_features}, f)

        logger.info("Saved: %s (%d features)", stress_geojson_path, len(stress_features))
        logger.info("Saved: %s (%d features)", irrigation_geojson_path, len(irrigation_features))

        # Calculate Area & KPI Statistics
        crop_cells_count = len(veg_df)
        total_cultivated_ha = round(crop_cells_count * CELL_AREA_HECTARES, 1)

        stress_counts = veg_df["pred_stress_label"].value_counts().to_dict()
        priority_counts = veg_df["irrigation_priority"].value_counts().to_dict()

        p1_cells = priority_counts.get(1, 0)
        p2_cells = priority_counts.get(2, 0)
        p3_cells = priority_counts.get(3, 0)

        stats = {
            "cycle_date": target_date,
            "total_culturable_cells": total_cells,
            "active_cultivated_cells": crop_cells_count,
            "fallow_or_bare_cells": total_cells - crop_cells_count,
            "total_cultivated_hectares": total_cultivated_ha,
            "stress_breakdown": {
                "high_stress_hectares": round(stress_counts.get("HIGH", 0) * CELL_AREA_HECTARES, 1),
                "high_stress_pct": round((stress_counts.get("HIGH", 0) / crop_cells_count) * 100, 1),
                "moderate_stress_hectares": round(stress_counts.get("MODERATE", 0) * CELL_AREA_HECTARES, 1),
                "moderate_stress_pct": round((stress_counts.get("MODERATE", 0) / crop_cells_count) * 100, 1),
                "low_stress_hectares": round(stress_counts.get("LOW", 0) * CELL_AREA_HECTARES, 1),
                "low_stress_pct": round((stress_counts.get("LOW", 0) / crop_cells_count) * 100, 1),
            },
            "irrigation_priority_breakdown": {
                "priority_1_urgent_ha": round(p1_cells * CELL_AREA_HECTARES, 1),
                "priority_1_pct": round((p1_cells / crop_cells_count) * 100, 1),
                "priority_2_scheduled_ha": round(p2_cells * CELL_AREA_HECTARES, 1),
                "priority_2_pct": round((p2_cells / crop_cells_count) * 100, 1),
                "priority_3_normal_ha": round(p3_cells * CELL_AREA_HECTARES, 1),
                "priority_3_pct": round((p3_cells / crop_cells_count) * 100, 1),
            },
            "mean_canopy_metrics": {
                "mean_ndvi": round(float(veg_df["ndvi_mean"].mean()), 3),
                "mean_ndwi": round(float(veg_df["ndwi_mean"].mean()), 3),
            },
        }

        stats_path = output_dir / "cycle_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved cycle KPI statistics to: %s", stats_path)

        # Print Executive Terminal Summary in Selected Language
        l = lang.lower() if lang.lower() in ("en", "mr", "hi") else "en"

        if l == "mr":
            print("\n" + "=" * 75)
            print(f"एग्रोस्कोप भौगोलिक पीक जल ताण विश्लेषण — निरीक्षण तारीख: {target_date}")
            print("=" * 75)
            print(f"एकूण सक्रिय बागायती क्षेत्र: {total_cultivated_ha:,} हेक्टर ({crop_cells_count:,} शेत विभाग)")
            print("-" * 75)
            print("पिकातील पाण्याचा ताण (Crop Water Stress Breakdown):")
            print(f"  🔴 तीव्र ताण (High Stress):     {stats['stress_breakdown']['high_stress_hectares']:>9,} हेक्टर  ({stats['stress_breakdown']['high_stress_pct']}%)")
            print(f"  🟡 मध्यम ताण (Moderate Stress): {stats['stress_breakdown']['moderate_stress_hectares']:>9,} हेक्टर  ({stats['stress_breakdown']['moderate_stress_pct']}%)")
            print(f"  🟢 कमी ताण (Low / Optimal):    {stats['stress_breakdown']['low_stress_hectares']:>9,} हेक्टर  ({stats['stress_breakdown']['low_stress_pct']}%)")
            print("-" * 75)
            print("सिंचन नियोजन व कृषी सल्ला (Irrigation Action Advisory):")
            print(f"  🚨 प्राथमिकता १ (तातडीने २४-४८ तासांत): {stats['irrigation_priority_breakdown']['priority_1_urgent_ha']:>9,} हेक्टर  ({stats['irrigation_priority_breakdown']['priority_1_pct']}%)")
            print(f"  📅 प्राथमिकता २ (पुढील ३-५ दिवसांत):     {stats['irrigation_priority_breakdown']['priority_2_scheduled_ha']:>9,} हेक्टर  ({stats['irrigation_priority_breakdown']['priority_2_pct']}%)")
            print(f"  ✅ प्राथमिकता ३ (पुरेसा ओलावा/बचत):       {stats['irrigation_priority_breakdown']['priority_3_normal_ha']:>9,} हेक्टर  ({stats['irrigation_priority_breakdown']['priority_3_pct']}%)")
            print("=" * 75 + "\n")
        elif l == "hi":
            print("\n" + "=" * 75)
            print(f"एग्रोस्कोप भू-स्थानिक फसल जल तनाव विश्लेषण — निरीक्षण तिथि: {target_date}")
            print("=" * 75)
            print(f"कुल सक्रिय सिंचित क्षेत्र: {total_cultivated_ha:,} हेक्टेयर ({crop_cells_count:,} खेत इकाइयां)")
            print("-" * 75)
            print("फसल जल तनाव विवरण (Crop Water Stress Breakdown):")
            print(f"  🔴 उच्च तनाव (High Stress):      {stats['stress_breakdown']['high_stress_hectares']:>9,} हेक्टेयर  ({stats['stress_breakdown']['high_stress_pct']}%)")
            print(f"  🟡 मध्यम तनाव (Moderate Stress): {stats['stress_breakdown']['moderate_stress_hectares']:>9,} हेक्टेयर  ({stats['stress_breakdown']['moderate_stress_pct']}%)")
            print(f"  🟢 कम तनाव (Low / Optimal):     {stats['stress_breakdown']['low_stress_hectares']:>9,} हेक्टेयर  ({stats['stress_breakdown']['low_stress_pct']}%)")
            print("-" * 75)
            print("सिंचाई कार्य योजना व सलाह (Irrigation Action Advisory):")
            print(f"  🚨 प्राथमिकता १ (तुरंत २४-४८ घंटों में): {stats['irrigation_priority_breakdown']['priority_1_urgent_ha']:>9,} हेक्टेयर  ({stats['irrigation_priority_breakdown']['priority_1_pct']}%)")
            print(f"  📅 प्राथमिकता २ (आगामी ३-५ दिनों में):   {stats['irrigation_priority_breakdown']['priority_2_scheduled_ha']:>9,} हेक्टेयर  ({stats['irrigation_priority_breakdown']['priority_2_pct']}%)")
            print(f"  ✅ प्राथमिकता ३ (संतोषजनक नमी):          {stats['irrigation_priority_breakdown']['priority_3_normal_ha']:>9,} हेक्टेयर  ({stats['irrigation_priority_breakdown']['priority_3_pct']}%)")
            print("=" * 75 + "\n")
        else:
            print("\n" + "=" * 75)
            print(f"AGROSCOPE GEOSPATIAL INFERENCE SUMMARY — CYCLE: {target_date}")
            print("=" * 75)
            print(f"Total Active Cultivated Area: {total_cultivated_ha:,} ha ({crop_cells_count:,} cells)")
            print("-" * 75)
            print("CROP WATER STRESS BREAKDOWN:")
            print(f"  🔴 High Stress:     {stats['stress_breakdown']['high_stress_hectares']:>9,} ha  ({stats['stress_breakdown']['high_stress_pct']}%)")
            print(f"  🟡 Moderate Stress: {stats['stress_breakdown']['moderate_stress_hectares']:>9,} ha  ({stats['stress_breakdown']['moderate_stress_pct']}%)")
            print(f"  🟢 Low Stress:      {stats['stress_breakdown']['low_stress_hectares']:>9,} ha  ({stats['stress_breakdown']['low_stress_pct']}%)")
            print("-" * 75)
            print("IRRIGATION ACTION ADVISORY:")
            print(f"  🚨 Priority 1 (Next 24-48h): {stats['irrigation_priority_breakdown']['priority_1_urgent_ha']:>9,} ha  ({stats['irrigation_priority_breakdown']['priority_1_pct']}%)")
            print(f"  📅 Priority 2 (Next 3-5d):   {stats['irrigation_priority_breakdown']['priority_2_scheduled_ha']:>9,} ha  ({stats['irrigation_priority_breakdown']['priority_2_pct']}%)")
            print(f"  ✅ Priority 3 (Optimal):     {stats['irrigation_priority_breakdown']['priority_3_normal_ha']:>9,} ha  ({stats['irrigation_priority_breakdown']['priority_3_pct']}%)")
            print("=" * 75 + "\n")

        return {
            "stats": stats,
            "stress_geojson": stress_geojson_path,
            "irrigation_geojson": irrigation_geojson_path,
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI options for inference pipeline."""
    parser = argparse.ArgumentParser(description="Run AgroScope Geospatial Inference & GeoJSON Export.")
    parser.add_argument("--date", type=str, default="latest", help="Cycle date (YYYY-MM-DD) or 'latest'")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "mr", "hi"], help="Output language: 'en' (English), 'mr' (Marathi), 'hi' (Hindi)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = AgroScopePredictor()
    predictor.run_cycle_inference(date_str=args.date, lang=args.lang)

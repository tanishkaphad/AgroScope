"""AgroScope Model Explainability Module (SHAP & Feature Attributions).

Implements Phase 12 of the AgroScope workflow:
1. Explains why a spatial cell or region was predicted as stressed.
2. Uses TreeExplainer to compute exact feature attributions for the multi-class tree models.
3. Generates global SHAP summary visualizations for reports/figures/shap_summary.png.
4. Returns local feature contribution breakdowns for on-demand FastAPI /area/{id} endpoint queries.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.explainability")

# Human-friendly descriptions for model features
FEATURE_EXPLANATIONS = {
    "ndwi_mean": "Canopy Leaf Water Content",
    "ndwi_std": "Canopy Water Spatial Variability",
    "ndvi_mean": "Vegetation Greenness & Canopy Vigor",
    "ndvi_std": "Vegetation Density Variance",
    "evi_mean": "Enhanced Canopy Biomass Index",
    "evi_std": "Biomass Structural Variance",
    "b02_mean": "Blue Spectral Band Reflectance",
    "b03_mean": "Green Spectral Band Reflectance",
    "b04_mean": "Red Spectral Band (Chlorophyll Absorption)",
    "b08_mean": "Near Infrared (Cellular Structure Reflectance)",
    "b11_mean": "Shortwave Infrared 1 (Leaf Moisture Absorption)",
    "b12_mean": "Shortwave Infrared 2 (Soil/Canopy Water Absorption)",
    "temp_mean_c": "Mean Daily Temperature",
    "temp_max_c": "Maximum Daily Temperature",
    "humidity_pct": "Relative Atmospheric Humidity",
    "wind_speed_mps": "Wind Speed (Advective Evaporative Pull)",
    "solar_radiation_mj": "Incoming Solar Radiation Energy",
    "rainfall_8day": "Recent 8-Day Cumulative Rainfall",
    "rainfall_16day": "16-Day Cumulative Rainfall History",
    "rainfall_30day": "30-Day Monthly Rainfall Accumulation",
    "temp_8day_mean": "8-Day Average Temperature",
    "temp_30day_mean": "30-Day Seasonal Mean Temperature",
    "temp_8day_max": "8-Day Peak Maximum Temperature",
    "et0_8day_sum": "8-Day Reference Evapotranspiration (ETo)",
    "et0_16day_sum": "16-Day Cumulative Evapotranspiration Demand",
}


class ModelExplainer:
    """Computes SHAP explanations and local feature drivers for water stress predictions."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.model_path = model_path or (base_dir / "models" / "model.pkl")
        self.metadata_path = metadata_path or (base_dir / "models" / "feature_metadata.json")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}")

        logger.info("Loading model for explainability from: %s", self.model_path)
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_names: List[str] = meta.get("features", [])
            self.class_labels: List[str] = meta.get("class_labels", ["LOW", "MODERATE", "HIGH"])
        else:
            self.feature_names = getattr(self.model, "feature_names_in_", []).tolist()
            self.class_labels = ["LOW", "MODERATE", "HIGH"]

        self._explainer = None
        self._init_shap_explainer()

    def _init_shap_explainer(self) -> None:
        """Initialize SHAP TreeExplainer if supported by model parameters."""
        try:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as exc:
            logger.warning(
                "SHAP TreeExplainer initialization skipped (%s). "
                "Local explainability will utilize model gain/feature-deviation fallback.",
                exc
            )
            self._explainer = None

    def explain_instance(
        self,
        features: Dict[str, float] | pd.Series | pd.DataFrame,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """Explain why a specific spatial unit received its predicted water stress class."""
        # Convert input to 1-row DataFrame matching training feature columns
        if isinstance(features, dict):
            row_df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            row_df = pd.DataFrame([features.to_dict()])
        else:
            row_df = features.copy()

        # Ensure all columns exist
        for col in self.feature_names:
            if col not in row_df.columns:
                row_df[col] = 0.0
        row_df = row_df[self.feature_names]

        # Model prediction & probabilities
        probs = self.model.predict_proba(row_df)[0]
        pred_class = int(np.argmax(probs))
        pred_label = self.class_labels[pred_class]
        confidence = float(probs[pred_class])

        drivers: List[Dict[str, Any]] = []

        if self._explainer is not None:
            # Native SHAP values
            shap_values = self._explainer.shap_values(row_df)
            # For multi-class, shap_values has shape (n_samples, n_features, n_classes) or list of classes
            if isinstance(shap_values, list):
                class_shap = shap_values[pred_class][0]
            elif shap_values.ndim == 3:
                class_shap = shap_values[0, :, pred_class]
            else:
                class_shap = shap_values[0]

            # Rank by absolute impact on the predicted class
            ranked_indices = np.argsort(np.abs(class_shap))[::-1][:top_k]
            total_abs_impact = np.sum(np.abs(class_shap)) or 1.0

            for idx in ranked_indices:
                feat_name = self.feature_names[idx]
                val = float(row_df.iloc[0, idx])
                raw_shap = float(class_shap[idx])
                pct_contrib = round((abs(raw_shap) / total_abs_impact) * 100, 1)

                direction = "increases_stress" if raw_shap > 0 else "mitigates_stress"
                arrow = "↑" if raw_shap > 0 else "↓"

                drivers.append({
                    "feature": feat_name,
                    "label": FEATURE_EXPLANATIONS.get(feat_name, feat_name),
                    "value": round(val, 3),
                    "impact_pct": pct_contrib,
                    "direction": direction,
                    "arrow": arrow,
                    "summary": f"{arrow} {FEATURE_EXPLANATIONS.get(feat_name, feat_name)} ({val:.2f}) — Contributes {pct_contrib}% to {pred_label}",
                })
        else:
            # Fallback heuristic using the model's feature importances & standardized deviation
            importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)))
            ranked_indices = np.argsort(importances)[::-1][:top_k]
            total_imp = np.sum(importances[ranked_indices]) or 1.0

            for idx in ranked_indices:
                feat_name = self.feature_names[idx]
                val = float(row_df.iloc[0, idx])
                imp = float(importances[idx])
                pct = round((imp / total_imp) * 100, 1)
                drivers.append({
                    "feature": feat_name,
                    "label": FEATURE_EXPLANATIONS.get(feat_name, feat_name),
                    "value": round(val, 3),
                    "impact_pct": pct,
                    "direction": "influences_prediction",
                    "arrow": "•",
                    "summary": f"• {FEATURE_EXPLANATIONS.get(feat_name, feat_name)} ({val:.2f}) — {pct}% relative importance",
                })

        return {
            "predicted_class_code": pred_class,
            "predicted_class_label": pred_label,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                label: round(float(prob), 4) for label, prob in zip(self.class_labels, probs)
            },
            "key_drivers": drivers,
        }

    def generate_global_summary_plot(
        self,
        features_csv: Optional[Path] = None,
        output_image: Optional[Path] = None,
        sample_size: int = 500,
    ) -> Optional[Path]:
        """Generate global SHAP summary plot across a representative sample of spatial units."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        features_csv = features_csv or (base_dir / "data" / "features" / "features.csv")
        output_image = output_image or (base_dir / "reports" / "figures" / "shap_summary.png")

        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("shap or matplotlib not installed. Skipping global summary plot generation.")
            return None

        if not features_csv.exists():
            logger.warning("Features CSV not found at: %s", features_csv)
            return None

        logger.info("Sampling %d rows from %s for SHAP summary...", sample_size, features_csv)
        df = pd.read_csv(features_csv)
        if "is_vegetated" in df.columns:
            df = df[df["is_vegetated"] == True]

        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)[self.feature_names]
        else:
            sample_df = df[self.feature_names]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(sample_df)

        output_image.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 7), dpi=300)

        # Plot for Class 2 (HIGH STRESS) as it is the most critical for irrigation intervention
        target_shap = shap_values[2] if isinstance(shap_values, list) else (
            shap_values[:, :, 2] if shap_values.ndim == 3 else shap_values
        )

        shap.summary_plot(
            target_shap,
            sample_df,
            feature_names=[FEATURE_EXPLANATIONS.get(f, f) for f in self.feature_names],
            show=False,
            max_display=12,
        )
        plt.title("SHAP Global Summary — Feature Drivers for HIGH Crop Water Stress", pad=15, weight="bold")
        plt.tight_layout()
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Saved SHAP global summary plot to: %s", output_image)
        return output_image

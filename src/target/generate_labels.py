"""AgroScope Ground Truth & Target Generation Module.

Implements Phase 7 of the AgroScope workflow:
1. Formulates the 3-class crop water stress classification target:
   - 0: LOW STRESS
   - 1: MODERATE STRESS
   - 2: HIGH STRESS
2. Uses the FAO-56 Penman-Monteith Climatic Water Balance Deficit proxy:
   - Deficit = (Kc * ETo_8day) - Effective_Rainfall_8day
3. Employs scientifically validated crop coefficients (Kc) for the Nira Left Bank Canal
   command area (Maharashtra Rabi season: Wheat, Sugarcane, Gram/Chana, Jowar).
4. Applies defensible physical thresholds to categorize stress levels.
5. Merges satellite grid observations with weather water deficit labels to generate
   the ground truth dataset for model training.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.target_generation")

# FAO-56 Crop Coefficients for Dominant Crops in Nira Left Bank Canal (Rabi Season)
CROP_KC_TABLE: Dict[str, Dict[str, float]] = {
    "Wheat": {
        "initial": 0.40,
        "mid_season": 1.15,
        "late_season": 0.35,
        "season_average": 0.90,
    },
    "Sugarcane": {
        "initial": 0.80,
        "mid_season": 1.25,
        "late_season": 0.75,
        "season_average": 1.10,
    },
    "Gram_Chana": {
        "initial": 0.40,
        "mid_season": 1.00,
        "late_season": 0.35,
        "season_average": 0.75,
    },
    "Jowar": {
        "initial": 0.35,
        "mid_season": 1.05,
        "late_season": 0.55,
        "season_average": 0.80,
    },
    "Command_Area_Weighted": {
        # Weighted by acreage in NLBC command area (Sugarcane ~40%, Wheat ~30%, Gram/Jowar ~30%)
        "season_average": 0.95,
    },
}

# FAO-56 Climatic Water Deficit (CWD) 8-day thresholds (mm)
# Low Stress: Deficit < 12 mm
# Moderate Stress: 12 mm <= Deficit <= 28 mm
# High Stress: Deficit > 28 mm
CWD_LOW_THRESHOLD = 12.0
CWD_HIGH_THRESHOLD = 28.0


def compute_climatic_water_deficit(
    weather_df: pd.DataFrame,
    kc: float = 0.95,
) -> pd.DataFrame:
    """Compute 8-day Climatic Water Deficit (CWD) and derive 3-class target labels."""
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # If 8-day rolling sums are not already present, compute them
    if "et0_8day_sum" not in df.columns:
        df["et0_8day_sum"] = df["et0_mm"].rolling(window=8, min_periods=1).sum()
    if "rainfall_8day" not in df.columns:
        rain_col = "rainfall_mm" if "rainfall_mm" in df.columns else "rain_mm"
        df["rainfall_8day"] = df[rain_col].rolling(window=8, min_periods=1).sum()

    # Crop Evapotranspiration Demand (ETc) = Kc * ETo
    df["crop_water_demand_8day_mm"] = (df["et0_8day_sum"] * kc).round(2)

    # USDA-SCS Effective Rainfall method (approximated for 8-day windows)
    # Light rainfall (< 5mm) mostly evaporates; effective fraction is ~80%
    df["effective_rainfall_8day_mm"] = np.clip(df["rainfall_8day"] * 0.8, 0, df["crop_water_demand_8day_mm"]).round(2)

    # Climatic Water Deficit (CWD) = Demand - Effective Rainfall
    df["water_deficit_8day_mm"] = np.maximum(
        0.0, df["crop_water_demand_8day_mm"] - df["effective_rainfall_8day_mm"]
    ).round(2)

    # Class 0: LOW STRESS (< 12 mm deficit)
    # Class 1: MODERATE STRESS (12 to 28 mm deficit)
    # Class 2: HIGH STRESS (> 28 mm deficit)
    conditions = [
        df["water_deficit_8day_mm"] < CWD_LOW_THRESHOLD,
        (df["water_deficit_8day_mm"] >= CWD_LOW_THRESHOLD) & (df["water_deficit_8day_mm"] <= CWD_HIGH_THRESHOLD),
        df["water_deficit_8day_mm"] > CWD_HIGH_THRESHOLD,
    ]
    choices = [0, 1, 2]
    df["target_stress_class"] = np.select(conditions, choices, default=1).astype(int)

    label_names = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
    df["target_stress_label"] = df["target_stress_class"].map(label_names)

    return df


def generate_target_dataset(
    weather_csv: Optional[Path] = None,
    satellite_features_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    kc: float = 0.95,
) -> Path:
    """Generate target labels and optionally merge with satellite grid features."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if weather_csv is None:
        weather_csv = base_dir / "data" / "raw" / "weather" / "daily_weather.csv"
    if satellite_features_csv is None:
        satellite_features_csv = base_dir / "data" / "processed" / "satellite_grid_features.csv"
    if output_dir is None:
        output_dir = base_dir / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not weather_csv.exists():
        raise FileNotFoundError(f"Weather CSV not found at: {weather_csv}")

    logger.info("Loading weather data from: %s", weather_csv)
    weather_df = pd.read_csv(weather_csv)

    # Compute CWD and target labels
    labeled_weather = compute_climatic_water_deficit(weather_df, kc=kc)

    target_labels_path = output_dir / "target_labels.csv"
    labeled_weather.to_csv(target_labels_path, index=False)
    logger.info("Saved daily target labels to: %s", target_labels_path)

    # Label distribution summary
    dist = labeled_weather["target_stress_label"].value_counts(normalize=True) * 100
    logger.info("Target class distribution in weather dataset:\n%s", dist.round(1).to_string())

    # If satellite features are present, join them by date
    if satellite_features_csv.exists():
        logger.info("Merging satellite grid features with target labels...")
        sat_df = pd.read_csv(satellite_features_csv)
        sat_df["date"] = pd.to_datetime(sat_df["date"]).dt.strftime("%Y-%m-%d")

        # Select relevant weather and target columns to join
        weather_cols = [
            "date",
            "temp_mean_c",
            "temp_max_c",
            "humidity_pct",
            "wind_speed_mps",
            "solar_radiation_mj",
            "rainfall_8day",
            "rainfall_16day",
            "rainfall_30day",
            "temp_8day_mean",
            "temp_30day_mean",
            "temp_8day_max",
            "et0_8day_sum",
            "et0_16day_sum",
            "water_deficit_8day_mm",
            "target_stress_class",
            "target_stress_label",
        ]
        weather_subset = labeled_weather[[c for c in weather_cols if c in labeled_weather.columns]]

        merged = pd.merge(sat_df, weather_subset, on="date", how="inner")

        # Scientific False-Alarm Filter:
        # If NDVI < 0.15, the cell is water, barren soil, or non-agricultural fallow.
        # Stressed crop prediction is only valid for cultivated/vegetated fields.
        merged["is_vegetated"] = merged["ndvi_mean"] >= 0.15

        features_dir = base_dir / "data" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        unified_dataset_path = features_dir / "features.csv"
        merged.to_csv(unified_dataset_path, index=False)
        logger.info(
            "Saved unified ML dataset with %d rows and %d columns to: %s",
            len(merged),
            len(merged.columns),
            unified_dataset_path,
        )
        return unified_dataset_path

    return target_labels_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate FAO-56 Climatic Water Deficit target labels.")
    parser.add_argument("--kc", type=float, default=0.95, help="Crop coefficient (default: 0.95)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_target_dataset(kc=args.kc)

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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.target_generation")

LABELS = settings["labels"]


def compute_climatic_water_deficit(
    weather_df: pd.DataFrame,
    kc: float = LABELS["default_kc"],
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
    df["effective_rainfall_8day_mm"] = np.clip(df["rainfall_8day"] * LABELS["effective_rainfall_fraction"], 0, df["crop_water_demand_8day_mm"]).round(2)

    # Climatic Water Deficit (CWD) = Demand - Effective Rainfall
    df["water_deficit_8day_mm"] = np.maximum(
        0.0, df["crop_water_demand_8day_mm"] - df["effective_rainfall_8day_mm"]
    ).round(2)

    # Class 0: LOW STRESS (< 12 mm deficit)
    # Class 1: MODERATE STRESS (12 to 28 mm deficit)
    # Class 2: HIGH STRESS (> 28 mm deficit)
    conditions = [
        df["water_deficit_8day_mm"] < LABELS["deficit_low_mm"],
        (df["water_deficit_8day_mm"] >= LABELS["deficit_low_mm"]) & (df["water_deficit_8day_mm"] <= LABELS["deficit_high_mm"]),
        df["water_deficit_8day_mm"] > LABELS["deficit_high_mm"],
    ]
    choices = [0, 1, 2]
    df["target_stress_class"] = np.select(conditions, choices, default=1).astype(int)

    label_names = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
    df["target_stress_label"] = df["target_stress_class"].map(label_names)

    return df


def assign_nearest_weather_points(cells: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Map each grid cell to its nearest weather point."""
    # Scale longitude so distances are comparable at this latitude
    lon_scale = np.cos(np.radians(points["latitude"].mean()))
    cell_xy = np.column_stack([cells["centroid_lon"] * lon_scale, cells["centroid_lat"]])
    point_xy = np.column_stack([points["longitude"] * lon_scale, points["latitude"]])

    distances = np.linalg.norm(cell_xy[:, None, :] - point_xy[None, :, :], axis=2)
    nearest = points["weather_point_id"].to_numpy()[distances.argmin(axis=1)]
    return pd.DataFrame({"cell_id": cells["cell_id"].to_numpy(), "weather_point_id": nearest})


def generate_target_dataset(
    weather_csv: Optional[Path] = None,
    satellite_features_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    kc: float = LABELS["default_kc"],
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
            "weather_point_id",
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
            "crop_water_demand_8day_mm",
            "effective_rainfall_8day_mm",
        ]
        weather_subset = labeled_weather[[c for c in weather_cols if c in labeled_weather.columns]]

        points = weather_df[["weather_point_id", "latitude", "longitude"]].drop_duplicates()
        cells = sat_df[["cell_id", "centroid_lon", "centroid_lat"]].drop_duplicates("cell_id")
        sat_df = sat_df.merge(assign_nearest_weather_points(cells, points), on="cell_id", how="left")

        merged = pd.merge(sat_df, weather_subset, on=["weather_point_id", "date"], how="inner")

        # Scientific False-Alarm Filter:
        # If NDVI < 0.15, the cell is water, barren soil, or non-agricultural fallow.
        # Stressed crop prediction is only valid for cultivated/vegetated fields.
        merged["is_vegetated"] = merged["ndvi_mean"] >= LABELS["min_vegetated_ndvi"]

        # FAO-56 Root-Zone Canopy Water Balance (Field-Level Spatial Ground Truth):
        # Climatic demand provides atmospheric evaporative pull; Sentinel-2 NDWI reflects field canopy hydration.
        # Irrigated crops (NDWI >= 0.18) buffer evaporative demand through root-zone moisture.
        # Depleted crops (NDWI <= 0.02) have zero irrigation buffer and experience full climatic deficit.
        buffer_settings = LABELS["canopy_buffer"]
        canopy_irrigation_buffer = (
            np.clip((merged["ndwi_mean"] - buffer_settings["ndwi_dry"]) / buffer_settings["ndwi_range"], 0.0, 1.0)
            * merged["crop_water_demand_8day_mm"]
            * buffer_settings["max_fraction"]
        )
        merged["water_deficit_8day_mm"] = np.maximum(
            0.0, merged["crop_water_demand_8day_mm"] - merged["effective_rainfall_8day_mm"] - canopy_irrigation_buffer
        ).round(2)

        # 3-Class Target Categorization
        conditions = [
            merged["water_deficit_8day_mm"] < LABELS["deficit_low_mm"],
            (merged["water_deficit_8day_mm"] >= LABELS["deficit_low_mm"]) & (merged["water_deficit_8day_mm"] <= LABELS["deficit_high_mm"]),
            merged["water_deficit_8day_mm"] > LABELS["deficit_high_mm"],
        ]
        choices = [0, 1, 2]
        merged["target_stress_class"] = np.select(conditions, choices, default=1).astype(int)
        label_names = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        merged["target_stress_label"] = merged["target_stress_class"].map(label_names)

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
        # Class distribution in active crop area
        crop_dist = merged[merged["is_vegetated"]]["target_stress_label"].value_counts(normalize=True) * 100
        logger.info("Field-level crop stress distribution (is_vegetated=True):\n%s", crop_dist.round(1).to_string())
        return unified_dataset_path

    return target_labels_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate FAO-56 Climatic Water Deficit target labels.")
    parser.add_argument("--kc", type=float, default=LABELS["default_kc"], help="Crop coefficient")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_target_dataset(kc=args.kc)

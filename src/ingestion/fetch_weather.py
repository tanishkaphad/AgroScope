"""AgroScope Weather Data Ingestion Module.

Fetches historical and near-real-time weather observations for the Canal
Command Area using the Open-Meteo Historical Weather API (100% Free, no API
key required).

Required Variables:
    - Daily Precipitation / Rainfall (mm)
    - Temperature (Mean, Max, Min in °C)
    - Relative Humidity (%)
    - Wind Speed (m/s)
    - Solar Radiation (MJ/m²)
    - Reference Evapotranspiration - FAO-56 Penman-Monteith (ETo in mm/day)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.weather_ingestion")

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def load_command_area_bounds(geojson_path: Path) -> Tuple[float, float, Dict[str, Any]]:
    """Load study area GeoJSON and compute centroid (lat, lon) and metadata."""
    if not geojson_path.exists():
        raise FileNotFoundError(f"Command area GeoJSON not found at: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find the polygon boundary feature
    polygon_feature = None
    for feat in data.get("features", []):
        if feat.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
            polygon_feature = feat
            break

    if not polygon_feature:
        raise ValueError(f"No Polygon geometry found in {geojson_path}")

    coords = polygon_feature["geometry"]["coordinates"]
    # Handle single Polygon exterior ring
    ring = coords[0] if polygon_feature["geometry"]["type"] == "Polygon" else coords[0][0]

    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]

    centroid_lon = sum(lons) / len(lons)
    centroid_lat = sum(lats) / len(lats)

    properties = polygon_feature.get("properties", {})
    logger.info(
        "Study Area Loaded: %s | Centroid: Lat %.4f, Lon %.4f",
        properties.get("name", "Unknown AOI"),
        centroid_lat,
        centroid_lon,
    )
    return centroid_lat, centroid_lon, properties


def fetch_weather_timeseries(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> pd.DataFrame:
    """Fetch daily weather parameters from Open-Meteo Archive API.

    Args:
        lat: Latitude of AOI centroid.
        lon: Longitude of AOI centroid.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        max_retries: Number of request retry attempts.
        backoff_factor: Exponential backoff factor.

    Returns:
        DataFrame containing date-indexed weather metrics.
    """
    daily_variables = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]

    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_variables),
        "timezone": "Asia/Kolkata",
    }

    logger.info("Requesting weather data from Open-Meteo (%s to %s)...", start_date, end_date)

    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as err:
            logger.warning("Attempt %d failed: %s", attempt, err)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to fetch weather data after {max_retries} attempts.") from err
            time.sleep(backoff_factor * attempt)

    data = response.json()
    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("Invalid response payload received from Open-Meteo API.")

    df = pd.DataFrame(daily)
    df.rename(
        columns={
            "time": "date",
            "temperature_2m_mean": "temp_mean_c",
            "temperature_2m_max": "temp_max_c",
            "temperature_2m_min": "temp_min_c",
            "precipitation_sum": "rainfall_mm",
            "rain_sum": "rain_mm",
            "relative_humidity_2m_mean": "humidity_pct",
            "wind_speed_10m_max": "wind_speed_mps",
            "shortwave_radiation_sum": "solar_radiation_mj",
            "et0_fao_evapotranspiration": "et0_mm",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Successfully fetched %d daily weather records.", len(df))
    return df


def calculate_water_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling lag metrics (8d, 16d, 30d) matching satellite cycles.

    8-day window aligns directly with Sentinel-2 observation intervals.
    """
    df = df.copy()

    # Cumulative rainfall windows
    df["rainfall_8day"] = df["rainfall_mm"].rolling(window=8, min_periods=1).sum().round(2)
    df["rainfall_16day"] = df["rainfall_mm"].rolling(window=16, min_periods=1).sum().round(2)
    df["rainfall_30day"] = df["rainfall_mm"].rolling(window=30, min_periods=1).sum().round(2)

    # Rolling mean temperature windows
    df["temp_8day_mean"] = df["temp_mean_c"].rolling(window=8, min_periods=1).mean().round(2)
    df["temp_30day_mean"] = df["temp_mean_c"].rolling(window=30, min_periods=1).mean().round(2)
    df["temp_8day_max"] = df["temp_max_c"].rolling(window=8, min_periods=1).max().round(2)

    # Rolling Evapotranspiration (ETo demand)
    df["et0_8day_sum"] = df["et0_mm"].rolling(window=8, min_periods=1).sum().round(2)
    df["et0_16day_sum"] = df["et0_mm"].rolling(window=16, min_periods=1).sum().round(2)

    # Atmospheric Water Deficit Proxy = Atmospheric Demand (ETo) - Rainfall
    df["climatic_water_deficit_8day"] = (df["et0_8day_sum"] - df["rainfall_8day"]).clip(lower=0.0).round(2)

    return df


def run_weather_pipeline(
    geojson_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-05-31",
) -> Path:
    """Execute complete weather data ingestion and processing pipeline."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if geojson_path is None:
        geojson_path = base_dir / "data" / "raw" / "study_area" / "command_area.geojson"
    if output_dir is None:
        output_dir = base_dir / "data" / "raw" / "weather"

    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon, metadata = load_command_area_bounds(geojson_path)
    df_raw = fetch_weather_timeseries(lat, lon, start_date=start_date, end_date=end_date)
    df_features = calculate_water_balance_features(df_raw)

    # Attach command area metadata
    df_features["aoi_id"] = metadata.get("id", "AOI_NLBC_PUNE")
    df_features["latitude"] = round(lat, 4)
    df_features["longitude"] = round(lon, 4)

    output_file = output_dir / "daily_weather.csv"
    df_features.to_csv(output_file, index=False)
    logger.info("Saved complete weather dataset to: %s", output_file)
    logger.info("Summary Statistics:\n%s", df_features[["rainfall_8day", "temp_8day_mean", "et0_8day_sum"]].describe().to_string())

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical weather data for AgroScope AOI.")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-05-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--geojson", type=str, default=None, help="Path to study area GeoJSON")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for weather CSV")

    args = parser.parse_args()
    geojson_p = Path(args.geojson) if args.geojson else None
    out_p = Path(args.outdir) if args.outdir else None

    run_weather_pipeline(
        geojson_path=geojson_p,
        output_dir=out_p,
        start_date=args.start_date,
        end_date=args.end_date,
    )

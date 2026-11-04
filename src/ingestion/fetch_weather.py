"""AgroScope Weather Data Ingestion Module.

Fetches daily weather for a grid of points across the Canal Command Area using
the Open-Meteo Historical Weather and Forecast APIs (free for non-commercial
use, no API key required, data licensed under CC BY 4.0).

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
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape

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
logger = logging.getLogger("agroscope.weather_ingestion")

WEATHER = settings["weather"]


def load_command_area(geojson_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load study area GeoJSON and return its polygon geometry and metadata."""
    if not geojson_path.exists():
        raise FileNotFoundError(f"Command area GeoJSON not found at: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for feat in data.get("features", []):
        if feat.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
            properties = feat.get("properties", {})
            logger.info("Study Area Loaded: %s", properties.get("name", "Unknown AOI"))
            return shape(feat["geometry"]), properties

    raise ValueError(f"No Polygon geometry found in {geojson_path}")


def build_weather_grid(geometry: Any, spacing_deg: float) -> List[Dict[str, Any]]:
    """Build evenly spaced weather points in and just around the study area."""
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    lons = np.arange(np.floor(min_lon / spacing_deg), np.ceil(max_lon / spacing_deg) + 1) * spacing_deg
    lats = np.arange(np.floor(min_lat / spacing_deg), np.ceil(max_lat / spacing_deg) + 1) * spacing_deg

    # A half-cell buffer keeps a nearby point for cells close to the boundary
    search_area = geometry.buffer(spacing_deg / 2)

    points: List[Dict[str, Any]] = []
    for lat in lats:
        for lon in lons:
            if search_area.contains(Point(lon, lat)):
                points.append({
                    "weather_point_id": f"WX_{len(points):03d}",
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                })

    if not points:
        raise ValueError(f"No weather points fell inside the study area at {spacing_deg} degree spacing.")

    logger.info("Built weather grid with %d points at %s degree spacing.", len(points), spacing_deg)
    return points


def request_daily_weather(
    url: str,
    lat: float,
    lon: float,
    extra_params: Dict[str, Any],
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> pd.DataFrame:
    """Fetch daily weather for one point from an Open-Meteo endpoint.

    Args:
        url: Open-Meteo API endpoint.
        lat: Latitude of the point.
        lon: Longitude of the point.
        extra_params: Date range or forecast length parameters.
        max_retries: Number of request retry attempts.
        backoff_factor: Exponential backoff factor.

    Returns:
        DataFrame containing date-indexed weather metrics.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(WEATHER["daily_variables"]),
        "timezone": WEATHER["timezone"],
        "wind_speed_unit": "ms",
        **extra_params,
    }

    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as err:
            logger.warning("Attempt %d failed: %s", attempt, err)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to fetch weather data after {max_retries} attempts.") from err
            rate_limited = err.response is not None and err.response.status_code == 429
            time.sleep(WEATHER["rate_limit_wait_s"] if rate_limited else backoff_factor * attempt)

    daily = response.json().get("daily", {})
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
    return df


def fetch_grid_weather(points: List[Dict[str, Any]], url: str, extra_params: Dict[str, Any]) -> pd.DataFrame:
    """Fetch weather for every grid point, one request per point."""
    frames = []
    for idx, point in enumerate(points, start=1):
        logger.info("[%d/%d] Fetching weather for %s...", idx, len(points), point["weather_point_id"])
        df = request_daily_weather(url, point["latitude"], point["longitude"], extra_params)
        for key, value in point.items():
            df[key] = value
        frames.append(df)
        time.sleep(WEATHER["request_delay_s"])
    return pd.concat(frames, ignore_index=True)


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


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling features separately for each weather point."""
    frames = [
        calculate_water_balance_features(group.sort_values("date"))
        for _, group in df.groupby("weather_point_id", sort=True)
    ]
    return pd.concat(frames, ignore_index=True)


def run_weather_pipeline(
    geojson_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    start_date: str = WEATHER["start_date"],
    end_date: str = WEATHER["end_date"],
) -> Path:
    """Fetch past weather across the grid and save it with rolling features."""
    geojson_path = geojson_path or (PROJECT_ROOT / "data" / "raw" / "study_area" / "command_area.geojson")
    output_dir = output_dir or (PROJECT_ROOT / "data" / "raw" / "weather")
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry, _ = load_command_area(geojson_path)
    points = build_weather_grid(geometry, WEATHER["grid_spacing_deg"])

    logger.info("Requesting weather history from Open-Meteo (%s to %s)...", start_date, end_date)
    df_raw = fetch_grid_weather(points, WEATHER["archive_url"], {"start_date": start_date, "end_date": end_date})
    df_features = add_grid_features(df_raw)
    df_features["aoi_id"] = settings["aoi"]["id"]

    output_file = output_dir / "daily_weather.csv"
    df_features.to_csv(output_file, index=False)
    logger.info("Saved weather history with %d rows to: %s", len(df_features), output_file)
    logger.info("Summary Statistics:\n%s", df_features[["rainfall_8day", "temp_8day_mean", "et0_8day_sum"]].describe().to_string())

    return output_file


def run_forecast_pipeline(
    geojson_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    forecast_days: int = WEATHER["forecast_days"],
) -> Path:
    """Fetch the latest weather forecast across the grid and save it."""
    geojson_path = geojson_path or (PROJECT_ROOT / "data" / "raw" / "study_area" / "command_area.geojson")
    output_dir = output_dir or (PROJECT_ROOT / "data" / "raw" / "weather")
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry, _ = load_command_area(geojson_path)
    points = build_weather_grid(geometry, WEATHER["grid_spacing_deg"])

    logger.info("Requesting %d-day forecast from Open-Meteo...", forecast_days)
    df = fetch_grid_weather(points, WEATHER["forecast_url"], {"forecast_days": forecast_days})
    df["forecast_issued_on"] = date.today().isoformat()
    df["aoi_id"] = settings["aoi"]["id"]

    output_file = output_dir / "forecast_weather.csv"
    df.to_csv(output_file, index=False)
    logger.info("Saved forecast with %d rows to: %s", len(df), output_file)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch weather data for AgroScope AOI.")
    parser.add_argument("--start-date", type=str, default=WEATHER["start_date"], help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=WEATHER["end_date"], help="End date (YYYY-MM-DD)")
    parser.add_argument("--geojson", type=str, default=None, help="Path to study area GeoJSON")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for weather CSV")
    parser.add_argument("--forecast", action="store_true", help="Fetch the latest forecast instead of past weather")

    args = parser.parse_args()
    geojson_p = Path(args.geojson) if args.geojson else None
    out_p = Path(args.outdir) if args.outdir else None

    if args.forecast:
        run_forecast_pipeline(geojson_path=geojson_p, output_dir=out_p)
    else:
        run_weather_pipeline(
            geojson_path=geojson_p,
            output_dir=out_p,
            start_date=args.start_date,
            end_date=args.end_date,
        )

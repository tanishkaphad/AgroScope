"""AgroScope Cloud Feature Extraction Module.

Calculates per-cell features directly from cloud-hosted Sentinel-2 scenes and keeps
only the results. A scene's results take far less space than its raw bands, so new
scenes can be added without downloading them.

Results are saved per scene with a manifest, so runs can be stopped and resumed.
They are not part of the DVC pipeline yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion.fetch_satellite import load_command_area_info, search_sentinel2_scenes
from src.preprocessing.preprocess_bands import (
    AOI,
    BOA_ADD_OFFSET,
    compute_scene_records,
    generate_spatial_grid,
    load_and_project_aoi,
    remove_boa_offset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.cloud_features")

SATELLITE = settings["satellite"]
TARGET_BANDS = SATELLITE["bands"]
FEATURE_MANIFEST_NAME = "feature_manifest.csv"
REFERENCE_BAND = "B04"
OFFSET_START_BASELINE = 4.0  # ESA added the reflectance offset from processing baseline 04.00


def offset_for_item(item: Any) -> int:
    """Return the reflectance offset that applies to a scene's processing baseline."""
    baseline = item.properties.get("s2:processing_baseline")
    if baseline is None:
        raise ValueError(f"Scene {item.id} has no processing baseline, so its offset is unknown.")
    return BOA_ADD_OFFSET if float(baseline) >= OFFSET_START_BASELINE else 0


def get_item_tile(item: Any) -> str:
    """Return a scene's MGRS tile from its properties or ID."""
    tile = item.properties.get("s2:mgrs_tile")
    if tile:
        return str(tile).lstrip("T").upper()
    for part in item.id.split("_"):
        if part.startswith("T") and len(part) == 6:
            return part[1:]
    return ""


def keep_latest_reprocessing(items: List[Any]) -> List[Any]:
    """Keep one version of each scene, because ESA can publish reprocessed copies."""
    latest: Dict[Tuple[str, str], Any] = {}
    for item in items:
        key = (item.datetime.date().isoformat(), get_item_tile(item))
        current = latest.get(key)
        if current is None or float(item.properties.get("s2:processing_baseline", 0)) > float(
            current.properties.get("s2:processing_baseline", 0)
        ):
            latest[key] = item
    return sorted(latest.values(), key=lambda it: (it.datetime, it.id))


def read_cloud_bands(item: Any, bbox_wgs84: List[float]) -> Tuple[Dict[str, np.ndarray], Any, Tuple[int, int]]:
    """Read the study area window of each band into memory, matched to the 10 m grid."""
    item = planetary_computer.sign(item)
    offset = offset_for_item(item)

    with rasterio.open(item.assets[REFERENCE_BAND].href) as ref:
        bounds = transform_bounds("EPSG:4326", ref.crs, *bbox_wgs84)
        window = (
            from_bounds(*bounds, transform=ref.transform)
            .intersection(Window(0, 0, ref.width, ref.height))
            .round_offsets()
            .round_lengths()
        )
        target_shape = (int(window.height), int(window.width))
        target_transform = window_transform(window, ref.transform)
        target_bounds = rasterio.windows.bounds(window, ref.transform)

    bands: Dict[str, np.ndarray] = {}
    for band_key, asset_name in TARGET_BANDS.items():
        asset = item.assets.get(band_key) or item.assets[asset_name]
        with rasterio.open(asset.href) as src:
            band_window = from_bounds(*target_bounds, transform=src.transform)
            data = src.read(
                1,
                window=band_window,
                out_shape=target_shape,
                resampling=Resampling.nearest if band_key == "SCL" else Resampling.bilinear,
            )
        bands[band_key] = data if band_key == "SCL" else remove_boa_offset(data, offset)

    return bands, target_transform, target_shape


def load_feature_manifest(output_dir: Path) -> pd.DataFrame:
    """Load the feature manifest, or an empty one on the first run."""
    path = output_dir / FEATURE_MANIFEST_NAME
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=["scene_id", "acquisition_date", "mgrs_tile", "cloud_cover_pct", "cells", "status", "feature_file"])


def save_feature_manifest(output_dir: Path, manifest: pd.DataFrame, record: Dict[str, Any]) -> pd.DataFrame:
    """Add one scene's outcome to the manifest and save it straight away."""
    new_row = pd.DataFrame([record])
    manifest = new_row if manifest.empty else pd.concat([manifest, new_row], ignore_index=True)
    manifest = manifest.drop_duplicates("scene_id", keep="last").sort_values(["acquisition_date", "scene_id"])
    manifest.to_csv(output_dir / FEATURE_MANIFEST_NAME, index=False)
    return manifest


def resolve_start_date(manifest: pd.DataFrame, raw_manifest_path: Path) -> str:
    """Start after the newest scene already covered by features or raw downloads."""
    dates = []
    if not manifest.empty:
        dates.append(str(manifest["acquisition_date"].max()))
    if raw_manifest_path.exists():
        dates.append(str(pd.read_csv(raw_manifest_path)["acquisition_date"].max()))
    return max(dates) if dates else SATELLITE["start_date"]


def run_cloud_feature_extraction(
    geojson_path: Optional[Path] = None,
    grid_geojson_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_scenes: Optional[int] = None,
    scene_ids: Optional[List[str]] = None,
    newest_first: bool = False,
) -> Path:
    """Calculate and save features for scenes that haven't been processed yet."""
    geojson_path = geojson_path or (PROJECT_ROOT / "data" / "raw" / "study_area" / "command_area.geojson")
    grid_geojson_path = grid_geojson_path or (PROJECT_ROOT / "data" / "processed" / "spatial_grid_250m.geojson")
    output_dir = output_dir or (PROJECT_ROOT / SATELLITE["feature_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_shape, aoi_bounds = load_and_project_aoi(geojson_path, target_crs=AOI["crs"])
    grid_cells = generate_spatial_grid(aoi_shape, aoi_bounds, output_geojson=grid_geojson_path)

    bbox, tiles, _ = load_command_area_info(geojson_path)
    manifest = load_feature_manifest(output_dir)
    raw_manifest_path = PROJECT_ROOT / "data" / "raw" / "satellite" / "scenes_manifest.csv"

    start = start_date or resolve_start_date(manifest, raw_manifest_path)
    end = end_date or date.today().isoformat()
    found = search_sentinel2_scenes(bbox, start, end, mgrs_tiles=tiles)

    if scene_ids is not None:
        items = [item for item in found if item.id in scene_ids]
        known = set()
    else:
        items = keep_latest_reprocessing(found)
        known = set(manifest.loc[manifest["status"] == "complete", "scene_id"])
        if raw_manifest_path.exists():
            known |= set(pd.read_csv(raw_manifest_path)["scene_id"])

    pending = [item for item in items if item.id not in known]
    if newest_first:
        pending.reverse()
    if max_scenes:
        pending = pending[:max_scenes]
    logger.info("%d scenes to process from %s to %s.", len(pending), start, end)

    for idx, item in enumerate(pending, start=1):
        acq_date = item.datetime.date().isoformat()
        feature_file = f"{acq_date}_{item.id}.csv.gz"
        record = {
            "scene_id": item.id,
            "acquisition_date": acq_date,
            "mgrs_tile": get_item_tile(item),
            "cloud_cover_pct": round(item.properties.get("eo:cloud_cover", float("nan")), 2),
            "cells": 0,
            "status": "failed",
            "feature_file": "",
        }
        started = time.time()
        logger.info("[%d/%d] Processing %s...", idx, len(pending), item.id)

        try:
            bands, transform, shape = read_cloud_bands(item, bbox)
            records = compute_scene_records(bands, transform, shape, acq_date, item.id, grid_cells)
            if records:
                pd.DataFrame(records).to_csv(output_dir / feature_file, index=False, compression="gzip")
                record.update(cells=len(records), status="complete", feature_file=feature_file)
            else:
                record["status"] = "no_clear_cells"
            logger.info("  %s: %d cells in %.0f s.", record["status"], len(records), time.time() - started)
        except Exception as exc:
            logger.error("  Failed: %s", exc)

        manifest = save_feature_manifest(output_dir, manifest, record)

    return output_dir / FEATURE_MANIFEST_NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate per-cell features directly from cloud-hosted Sentinel-2 scenes.")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD). Default: after the newest known scene")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit the number of scenes in this run")
    parser.add_argument("--scene-id", action="append", default=None, help="Process only this scene, even if it's already known")
    parser.add_argument("--newest-first", action="store_true", help="Process the newest scenes first")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for feature files")

    args = parser.parse_args()
    run_cloud_feature_extraction(
        output_dir=Path(args.outdir) if args.outdir else None,
        start_date=args.start_date,
        end_date=args.end_date,
        max_scenes=args.max_scenes,
        scene_ids=args.scene_id,
        newest_first=args.newest_first,
    )

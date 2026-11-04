"""AgroScope Sentinel-2 Satellite Data Ingestion Module.

Queries and downloads Sentinel-2 Level-2A (L2A) surface reflectance imagery
covering the Canal Command Area using Microsoft Planetary Computer STAC API (100% Free).

Features:
    - Queries Sentinel-2 L2A STAC collection for specific AOI and MGRS tile (e.g., 43QDF)
    - Filters by cloud cover threshold (< 20%)
    - Uses Cloud-Optimized GeoTIFF (COG) windowed reads to download ONLY the AOI extent
      (saving 95% bandwidth and disk space compared to downloading full scenes)
    - Extracts essential water stress bands:
        * B02 (Blue, 10m)
        * B03 (Green, 10m)
        * B04 (Red, 10m)
        * B08 (NIR, 10m)
        * B11 (SWIR1, 20m)
        * B12 (SWIR2, 20m)
        * SCL (Scene Classification Layer, 20m - cloud/shadow mask)
    - Generates a scene manifest CSV tracking acquisition dates, cloud cover, and file paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import shape

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
logger = logging.getLogger("agroscope.satellite_ingestion")

SATELLITE = settings["satellite"]
MANIFEST_NAME = "scenes_manifest.csv"

# Target bands for vegetation & water stress indices (NDVI, NDWI, EVI)
TARGET_BANDS = SATELLITE["bands"]


def load_command_area_info(geojson_path: Path) -> Tuple[List[float], List[str], Dict[str, Any]]:
    """Load study area GeoJSON and extract bounding box and metadata.

    Returns:
        Tuple of (bbox [min_lon, min_lat, max_lon, max_lat], mgrs_tiles, properties)
    """
    if not geojson_path.exists():
        raise FileNotFoundError(f"Study area GeoJSON not found at: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygon_feature = None
    for feat in data.get("features", []):
        if feat.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
            polygon_feature = feat
            break

    if not polygon_feature:
        raise ValueError(f"No Polygon geometry found in {geojson_path}")

    geom = shape(polygon_feature["geometry"])
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    bbox = [min_lon, min_lat, max_lon, max_lat]

    properties = polygon_feature.get("properties", {})
    raw_tiles = properties.get("sentinel2_mgrs_tiles")
    if not raw_tiles:
        single = properties.get("sentinel2_mgrs_tile")
        mgrs_tiles = [single] if single else settings["aoi"]["default_mgrs_tiles"]
    elif isinstance(raw_tiles, list):
        mgrs_tiles = raw_tiles
    else:
        mgrs_tiles = [t.strip() for t in str(raw_tiles).split(",")]

    logger.info(
        "Loaded AOI: %s | MGRS Tiles: %s | BBox: [%.4f, %.4f, %.4f, %.4f]",
        properties.get("name", "AOI"),
        mgrs_tiles,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )
    return bbox, mgrs_tiles, properties


def search_sentinel2_scenes(
    bbox: List[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = SATELLITE["max_cloud_pct"],
    mgrs_tiles: Optional[List[str]] = None,
) -> List[Any]:
    """Query Microsoft Planetary Computer STAC catalog for Sentinel-2 L2A scenes."""
    logger.info(
        "Searching Sentinel-2 L2A scenes (Date: %s to %s, Max Cloud: %.1f%%)...",
        start_date,
        end_date,
        max_cloud_cover,
    )

    catalog = pystac_client.Client.open(
        SATELLITE["stac_url"],
        modifier=planetary_computer.sign_inplace,
    )

    # Query by collection, bbox and datetime window
    search = catalog.search(
        collections=[SATELLITE["collection"]],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
    )

    all_items = list(search.items())
    logger.info("Retrieved %d candidate scenes from catalog. Filtering...", len(all_items))

    # Normalize allowed tiles
    allowed_tiles = {t.lstrip("T").upper() for t in mgrs_tiles if t} if mgrs_tiles else set()

    # Robust client-side filtering avoids STAC QUERY extension incompatibilities
    filtered_items = []
    discovered_tiles = set()

    for item in all_items:
        # Support various STAC cloud cover property keys
        cloud = item.properties.get("eo:cloud_cover")
        if cloud is None:
            cloud = item.properties.get("s2:high_proba_clouds_percentage", 100.0)

        # Extract MGRS tile from properties or parse from standard Sentinel-2 item ID
        tile = item.properties.get("s2:mgrs_tile")
        if not tile:
            for part in item.id.split("_"):
                if part.startswith("T") and len(part) == 6:
                    tile = part[1:]
                    break
                elif len(part) == 5 and part[:2].isdigit():
                    tile = part
                    break

        norm_tile = tile.lstrip("T").upper() if tile else None
        if norm_tile:
            discovered_tiles.add(norm_tile)

        if cloud > max_cloud_cover:
            continue
        # Filter by tile if specified and detected
        if allowed_tiles and norm_tile and norm_tile not in allowed_tiles:
            continue

        filtered_items.append(item)

    # Sort chronologically
    filtered_items.sort(key=lambda x: x.datetime or x.properties.get("datetime", ""))
    logger.info("Catalog candidate scenes cover tiles: %s", sorted(list(discovered_tiles)))
    logger.info(
        "Found %d cloud-filtered (< %.1f%%) Sentinel-2 scenes for target tiles: %s.",
        len(filtered_items),
        max_cloud_cover,
        sorted(list(allowed_tiles)) if allowed_tiles else "ALL INTERSECTING",
    )
    return filtered_items


def download_windowed_band(
    signed_asset_url: str,
    output_path: Path,
    aoi_bbox_wgs84: List[float],
) -> bool:
    """Read and save only the AOI bounding box window from a remote Cloud-Optimized GeoTIFF.

    This avoids downloading entire 100km x 100km scenes, saving substantial disk space and bandwidth.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(signed_asset_url) as src:
        # Transform WGS84 bbox (lon/lat) to the raster's native CRS (e.g. UTM EPSG:32643)
        min_lon, min_lat, max_lon, max_lat = aoi_bbox_wgs84
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat
        )

        # Compute pixel window covering the bounds
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        # Clamp window to raster extents
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )

        data = src.read(1, window=window)
        transform = rasterio.windows.transform(window, src.transform)

        profile = src.profile.copy()
        profile.update(
            width=int(window.width),
            height=int(window.height),
            transform=transform,
            compress="deflate",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    return True


def load_manifest(output_dir: Path) -> pd.DataFrame:
    """Load the existing scene manifest, or an empty one on the first run."""
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        return pd.read_csv(manifest_path)
    return pd.DataFrame(columns=["scene_id", "acquisition_date"])


def merge_manifest(existing: pd.DataFrame, new_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Add new scene records to the manifest, keeping the newest record per scene."""
    frames = [df for df in (existing, pd.DataFrame(new_records)) if not df.empty]
    if not frames:
        return existing
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("scene_id", keep="last")
    return merged.sort_values(["acquisition_date", "scene_id"]).reset_index(drop=True)


def is_scene_complete(scene_dir: Path) -> bool:
    """Check that every target band has already been downloaded for a scene."""
    return all(
        (scene_dir / f"{band}.tif").exists() and (scene_dir / f"{band}.tif").stat().st_size > 1000
        for band in TARGET_BANDS
    )


def resolve_since_last_start(manifest: pd.DataFrame, default_start: str) -> str:
    """Start a search from the newest scene date already in the manifest."""
    if manifest.empty:
        return default_start
    return str(manifest["acquisition_date"].max())


def fetch_satellite_data(
    geojson_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    start_date: str = SATELLITE["start_date"],
    end_date: str = SATELLITE["end_date"],
    max_cloud_cover: float = SATELLITE["max_cloud_pct"],
    mgrs_tiles: Optional[List[str]] = None,
    dry_run: bool = False,
    max_scenes: Optional[int] = None,
    since_last: bool = False,
) -> Path:
    """Execute complete satellite search and acquisition pipeline.

    With since_last, searches from the newest known scene up to today and leaves
    known scenes alone. The manifest keeps every scene seen across runs.

    Default date window focuses on the Rabi season (Oct 2023 - Mar 2024),
    which is the primary irrigated cropping season in Maharashtra.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    if geojson_path is None:
        geojson_path = base_dir / "data" / "raw" / "study_area" / "command_area.geojson"
    if output_dir is None:
        output_dir = base_dir / "data" / "raw" / "satellite"

    output_dir.mkdir(parents=True, exist_ok=True)

    bbox, default_tiles, properties = load_command_area_info(geojson_path)
    target_tiles = mgrs_tiles if mgrs_tiles is not None else default_tiles

    existing_manifest = load_manifest(output_dir)
    known_scene_ids = set(existing_manifest["scene_id"])
    if since_last:
        start_date = resolve_since_last_start(existing_manifest, start_date)
        end_date = date.today().isoformat()

    scenes = search_sentinel2_scenes(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        mgrs_tiles=target_tiles,
    )

    new_scenes = [item for item in scenes if item.id not in known_scene_ids]
    logger.info("%d of %d scenes are new.", len(new_scenes), len(scenes))

    if max_scenes and len(scenes) > max_scenes:
        logger.info("Limiting download to first %d scenes as requested.", max_scenes)
        scenes = scenes[:max_scenes]

    manifest_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(scenes, start=1):
        acq_datetime = item.datetime.isoformat() if item.datetime else str(item.properties.get("datetime"))
        acq_date = acq_datetime[:10]
        cloud_cover = item.properties.get("eo:cloud_cover", 0.0)
        scene_id = item.id

        logger.info(
            "[%d/%d] Scene: %s | Date: %s | Cloud: %.1f%%",
            idx,
            len(scenes),
            scene_id,
            acq_date,
            cloud_cover,
        )

        scene_tile = item.properties.get("s2:mgrs_tile")
        if not scene_tile:
            for part in scene_id.split("_"):
                if part.startswith("T") and len(part) == 6:
                    scene_tile = part[1:]
                    break

        record: Dict[str, Any] = {
            "scene_id": scene_id,
            "acquisition_date": acq_date,
            "cloud_cover_pct": round(cloud_cover, 2),
            "mgrs_tile": scene_tile or "",
        }

        if scene_id in known_scene_ids and (dry_run or is_scene_complete(output_dir / f"{acq_date}_{scene_id}")):
            logger.info("  Scene already known, skipping.")
            continue

        if dry_run:
            record["status"] = "found"
        else:
            scene_dir = output_dir / f"{acq_date}_{scene_id}"

            # Dynamically refresh SAS token per scene to prevent 403 token expiration on long runs
            try:
                active_item = planetary_computer.sign(item)
            except Exception as sign_err:
                logger.warning("Could not refresh token for scene %s (%s), falling back to existing token.", scene_id, sign_err)
                active_item = item

            for band_key, asset_name in TARGET_BANDS.items():
                if band_key in active_item.assets:
                    asset = active_item.assets[band_key]
                elif asset_name in active_item.assets:
                    asset = active_item.assets[asset_name]
                else:
                    logger.warning("Band %s not found in assets for scene %s", band_key, scene_id)
                    continue

                band_filename = f"{band_key}.tif"
                dest_path = scene_dir / band_filename

                if dest_path.exists() and dest_path.stat().st_size > 1000:
                    logger.info("  Band %s already exists (%.2f MB), skipping.", band_key, dest_path.stat().st_size / (1024 * 1024))
                    record[f"path_{band_key}"] = str(dest_path.relative_to(base_dir))
                    continue

                try:
                    logger.info("  Streaming windowed band %s...", band_key)
                    download_windowed_band(
                        signed_asset_url=asset.href,
                        output_path=dest_path,
                        aoi_bbox_wgs84=bbox,
                    )
                    record[f"path_{band_key}"] = str(dest_path.relative_to(base_dir))
                except Exception as err:
                    logger.error("  Failed to download band %s: %s", band_key, err)
                    if dest_path.exists():
                        dest_path.unlink(missing_ok=True)
                    record[f"path_{band_key}"] = None

            record["status"] = "complete" if is_scene_complete(scene_dir) else "incomplete"

            # Clean up empty scene directory if no bands were successfully downloaded
            if scene_dir.exists() and not any(scene_dir.iterdir()):
                try:
                    scene_dir.rmdir()
                except Exception:
                    pass

        manifest_records.append(record)

    manifest_df = merge_manifest(existing_manifest, manifest_records)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Saved satellite scenes manifest to: %s", manifest_path)
    logger.info(
        "Acquisition Summary:\n%s",
        manifest_df[["acquisition_date", "cloud_cover_pct"]].to_string() if not manifest_df.empty else "No scenes found.",
    )

    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Sentinel-2 L2A imagery for AgroScope Command Area."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=SATELLITE["start_date"],
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=SATELLITE["end_date"],
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=SATELLITE["max_cloud_pct"],
        help="Maximum cloud cover percentage allowed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search and catalog scenes only without downloading raster bands",
    )
    parser.add_argument(
        "--since-last",
        action="store_true",
        help="Search from the newest known scene up to today, ignoring the date options",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit number of scenes to download (useful for testing)",
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default=None,
        help="Comma-separated MGRS tiles (e.g. 43QDA,43QDV) or 'all' for all intersecting tiles",
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default=None,
        help="Path to study area GeoJSON",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for satellite rasters",
    )

    args = parser.parse_args()
    geojson_p = Path(args.geojson) if args.geojson else None
    out_p = Path(args.outdir) if args.outdir else None
    
    if args.tiles:
        if args.tiles.lower() == "all":
            parsed_tiles = []
        else:
            parsed_tiles = [t.strip() for t in args.tiles.split(",")]
    else:
        parsed_tiles = None

    fetch_satellite_data(
        geojson_path=geojson_p,
        output_dir=out_p,
        start_date=args.start_date,
        end_date=args.end_date,
        max_cloud_cover=args.max_cloud,
        mgrs_tiles=parsed_tiles,
        dry_run=args.dry_run,
        max_scenes=args.max_scenes,
        since_last=args.since_last,
    )

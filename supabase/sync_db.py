"""AgroScope Database Synchronization & Supabase PostGIS Loader.

Provides utilities to:
1. Initialize the PostgreSQL schema in Supabase.
2. Ingest the study area boundary polygon (Nira Left Bank Canal).
3. Ingest spatial units (250m grid cells).
4. Synchronize 8-day cycle predictions and irrigation advisories.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.supabase_sync")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_supabase_client():
    """Retrieve Supabase client if environment credentials are present."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.warning(
            "SUPABASE_URL or SUPABASE_KEY not found in environment. "
            "Database sync is in mock/dry-run mode."
        )
        return None
        
    try:
        from supabase import create_client, Client
        client: Client = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase project at: %s", supabase_url)
        return client
    except Exception as exc:
        logger.error("Failed to connect to Supabase: %s", exc)
        return None


def sync_study_area(client=None) -> Dict[str, Any]:
    """Ingest command area boundary GeoJSON."""
    boundary_path = PROJECT_ROOT / "data" / "raw" / "study_area" / "command_area.geojson"
    if not boundary_path.exists():
        logger.error("Boundary file not found at: %s", boundary_path)
        return {"status": "error", "message": "Boundary GeoJSON missing"}

    with open(boundary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    feature = data["features"][0]
    props = feature.get("properties", {})
    record = {
        "name": props.get("name", "Nira Left Bank Canal Command Area"),
        "project": props.get("project", "Nira Valley Irrigation Project"),
        "state": props.get("state", "Maharashtra"),
        "district": props.get("district", "Pune"),
        "culturable_command_area_ha": props.get("culturable_command_area_ha", 66250),
        "gross_command_area_ha": props.get("gross_command_area_ha", 84500),
        "main_canal_length_km": props.get("main_canal_length_km", 162.0),
        "metadata": props,
    }

    if client:
        try:
            res = client.table("study_areas").upsert(record, on_conflict="name").execute()
            logger.info("Successfully synced study area to Supabase: %s", res.data)
            return {"status": "synced", "data": res.data}
        except Exception as exc:
            logger.error("Failed to upsert study area to Supabase: %s", exc)
            return {"status": "error", "error": str(exc)}
    else:
        logger.info("Dry-run: Study area ready for ingestion: %s", record["name"])
        return {"status": "dry_run", "record": record}


if __name__ == "__main__":
    client = get_supabase_client()
    res = sync_study_area(client)
    print(json.dumps(res, indent=2))

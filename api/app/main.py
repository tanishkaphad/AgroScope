"""AgroScope Production FastAPI Application Entrypoint.

Satellite Crop Water Stress Estimation & Irrigation Advisory System.
Compatible with Railway Free Tier, Docker, and Cloudflare Pages frontend.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.app.config import settings
from api.app.routes import health, geo, stats, inference, mlops
from api.app.services.model_service import model_service
from api.app.services.geo_service import geo_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agroscope.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    logger.info("Starting AgroScope API server...")
    # Preload services
    _ = model_service
    _ = geo_service
    logger.info("AgroScope API initialized and ready to serve requests.")
    yield
    logger.info("Shutting down AgroScope API server...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware (permits local dev, Cloudflare Pages edge domain, Railway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api/v1
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(geo.router, prefix=settings.API_V1_STR)
app.include_router(stats.router, prefix=settings.API_V1_STR)
app.include_router(inference.router, prefix=settings.API_V1_STR)
app.include_router(mlops.router, prefix=settings.API_V1_STR)

# Register top-level aliases for direct access
app.include_router(health.router)
app.include_router(geo.router)
app.include_router(stats.router)
app.include_router(inference.router)
app.include_router(mlops.router)


@app.get("/", tags=["Root"])
async def root_endpoint():
    """Root metadata and service health status."""
    return {
        "system": "AgroScope — Satellite Crop Water Stress & Irrigation Advisory System",
        "command_area": "Nira Left Bank Canal, Pune / Maharashtra",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "api_v1": settings.API_V1_STR,
    }


# Mount static frontend directory if built
FRONTEND_DIST = settings.BASE_DIR / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="dashboard")


# AgroScope — Complete Project Workflow

## Satellite-Based Crop Water Stress Estimation and Irrigation Advisory for Canal Command Areas
### Fully Automated MLOps & 100% Free-Tier Cloud Deployment

> **"AgroScope is an end-to-end, reproducible MLOps framework. While piloted and calibrated on the Nira Left Bank Canal, the entire data-to-deployment pipeline is parameter-driven—meaning the system can be adapted and retrained for any canal command area in India simply by providing its boundary GeoJSON."**

---

## 1. Project Overview

**AgroScope** is an end-to-end geospatial AI and MLOps system designed to estimate crop water stress within canal command areas using satellite imagery, weather information, and crop-related data.

The system operates on an **8-day observation cycle** and produces:

- Crop water-stress maps
- Irrigation-priority maps
- Area-wise statistics
- Explainable predictions
- Irrigation advisories

The complete end-to-end workflow is:

```text
Data Acquisition (Sentinel-2, Weather API)
      ↓
Data Validation & AOI Selection
      ↓
Geospatial Preprocessing (GDAL, Rasterio, GeoPandas)
      ↓
Feature Engineering & Target Generation
      ↓
DVC Data Versioning (Remote: DagsHub Storage)
      ↓
ML Training (Random Forest baseline, XGBoost main)
      ↓
Experiment Tracking & Model Registry (DagsHub MLflow)
      ↓
Model Explainability (SHAP)
      ↓
ML Monitoring & Drift Detection (Evidently AI)
      ↓
Database Layer (Supabase PostgreSQL + PostGIS)
      ↓
Backend API (FastAPI containerized with Docker)
      ↓
Backend Free Cloud Deployment (Railway)
      ↓
Frontend Web Dashboard (React + Leaflet)
      ↓
Frontend Free Cloud Deployment (Cloudflare Pages)
      ↓
Application Monitoring (UptimeRobot pinging /health)
      ↓
Automated CI/CD & Retraining (GitHub Actions Cron + DVC + API Trigger)
```

---

# 2. Project Objective

Develop a satellite-based AI system that estimates crop water stress over an 8-day period for agricultural areas inside a canal command area and converts the prediction into an irrigation-priority advisory.

The system should answer:

1. **Where is crop water stress occurring?**
2. **How severe is the stress?**
3. **Which areas should receive irrigation priority?**
4. **Why was an area classified as stressed?**
5. **Can the complete ML workflow be reproduced, monitored, and retrained automatically at zero hosting cost?**

---

# 3. SDG Alignment

| SDG | Contribution |
|---|---|
| **SDG 2 – Zero Hunger** | Supports crop health and agricultural productivity |
| **SDG 6 – Clean Water and Sanitation** | Promotes efficient irrigation-water utilization |
| **SDG 13 – Climate Action** | Supports adaptation to agricultural water scarcity and climate variability |

---

# 4. High-Level Architecture & 100% Free Deployment Stack

```text
                               ┌─────────────────────────────┐
                               │  Canal Command Area (AOI)   │
                               │  Boundary / GIS Polygons    │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                ┌───────────────────────────────────────────────────────────┐
                │                    DATA ACQUISITION                       │
                │                                                           │
                │ Sentinel-2 L2A (Copernicus/STAC) │ Open-Meteo / NASA POWER │
                │ Reference / Target Water Data   │ Crop / Kc Guidelines    │
                └─────────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                ┌───────────────────────────────────────────────────────────┐
                │             GEOSPATIAL PROCESSING & DVC                   │
                │                                                           │
                │ Cloud Masking → Clipping → CRS → Alignment → Grids / Area │
                │ Versioned with DVC ──► Remote Storage on DagsHub (Free)   │
                └─────────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                ┌───────────────────────────────────────────────────────────┐
                │                   FEATURE ENGINEERING                     │
                │                                                           │
                │ NDVI │ NDWI │ EVI │ Spectral Bands │ 8/16/30d Weather    │
                │ Crop Stage │ Kc │ Soil / Moisture Proxy                   │
                └─────────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                ┌───────────────────────────────────────────────────────────┐
                │         ML MODEL EXPERIMENTATION & TRACKING               │
                │                                                           │
                │ XGBoost (Tuned) vs RF Baseline                            │
                │ Experiment Tracking & Model Registry: DagsHub MLflow      │
                └─────────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │          WATER STRESS & ADVISORY            │
                       │                                             │
                       │ Class: Low / Moderate / High Stress         │
                       │ Advisory: Priority 1 / Priority 2 / 3       │
                       │ Explainability: SHAP Feature Attributions   │
                       └──────────────────────┬──────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │          DATABASE LAYER (FREE TIER)         │
                       │                                             │
                       │ Supabase (Managed PostgreSQL + PostGIS)     │
                       │ GeoJSON Polygons │ Predictions │ Advisories │
                       └──────────────┬──────────────────────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────────────────────┐
                       │          BACKEND SERVICE (FREE TIER)        │
                       │                                             │
                       │ FastAPI Backend in Docker Container         │
                       │ Hosted on Railway (Free Tier)               │
                       │ Endpoints: /stress-map, /predict, /retrain  │
                       └──────────────┬──────────────────────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────────────────────┐
                       │          FRONTEND SERVICE (FREE TIER)       │
                       │                                             │
                       │ React + Leaflet Interactive GIS Dashboard   │
                       │ Hosted on Cloudflare Pages (Free CDN)       │
                       └─────────────────────────────────────────────┘
                                      │
                                      ▼
                ┌───────────────────────────────────────────────────────────┐
                │           AUTOMATED MLOps & CONTINUOUS MONITORING         │
                │                                                           │
                │ CI/CD Pipeline: GitHub Actions (Free 2,000 mins/mo)       │
                │ Data & Prediction Drift: Evidently AI (Open Source)       │
                │ Application Health & Keep-Alive: UptimeRobot (Free)       │
                │ Retraining Triggers: 8-Day Cron + DVC Hash + API Trigger  │
                └───────────────────────────────────────────────────────────┘
```

### 100% Free Tier Service Breakdown

| Component | Technology | Free Tier Details | Role in AgroScope |
|---|---|---|---|
| **Backend API** | **Railway** | Free trial credits ($5 monthly allowance), Docker support | Hosts containerized FastAPI REST API |
| **Frontend UI** | **Cloudflare Pages** | 100% Free, unlimited bandwidth, global edge CDN | Hosts React + Leaflet web dashboard |
| **Database & GIS** | **Supabase** | 100% Free tier (500MB DB, PostGIS, SSL connection) | Stores spatial units, prediction logs & advisories |
| **DVC Remote Storage** | **DagsHub Storage** | 100% Free storage for DVC datasets & artifacts | Stores raw/processed imagery & trained models |
| **MLflow Server** | **DagsHub MLflow** | 100% Free hosted MLflow tracking server & registry | Experiment metrics, parameter logging, model registry |
| **ML Drift Monitoring** | **Evidently AI** | Open-source Python library (zero cloud charge) | Detects data & prediction drift; generates HTML/JSON |
| **Automation & CI/CD** | **GitHub Actions** | 2,000 free runner minutes/month | Automated testing, builds, deployment & retrain cron |
| **App Uptime / Monitoring** | **UptimeRobot** | 50 free monitors, 5-minute ping interval | Pings `/health` to keep service awake and alert outages |

---

# 5. Complete Project Phases

| Phase | Module | Technology / Platform | Main Deliverable |
|---|---|---|---|
| 0 | Scope & Data Validation | Python, STAC API | Validated data inventory & study area definition |
| 1 | Study Area Selection | GeoJSON, QGIS/GeoPandas | Final AOI geometry (command area boundary) |
| 2 | Satellite Acquisition | Sentinel-2 L2A (STAC/Copernicus) | Cloud-filtered optical bands |
| 3 | Weather Acquisition | Open-Meteo API (Free) | 8-day / 16-day aggregated weather features |
| 4 | Crop & Reference Data | Agri Data / Land Cover Mask | Dominant crop map & growth coefficients |
| 5 | Geospatial Processing | Rasterio, GDAL, GeoPandas | Normalized, clipped, grid-aligned rasters |
| 6 | Feature Engineering | NumPy, Pandas | ML feature matrix (indices + weather + crop) |
| 7 | Target Generation | Deficit Proxy / Water Balance | Validated water stress labels (Low/Mod/High) |
| 8 | ML Dataset Construction | DVC + DagsHub | Final ML dataset versioned on DagsHub DVC |
| 9 | EDA | Seaborn, Plotly | Distributions, correlation analysis & report |
| 10 | ML Training | Scikit-learn, XGBoost | Random Forest baseline & tuned XGBoost model |
| 11 | Evaluation | Scikit-learn, DagsHub MLflow | Spatial/temporal cross-validation metrics |
| 12 | Explainability | SHAP | Global & local feature attribution plots |
| 13 | Prediction Pipeline | Python, PostGIS | Batch & on-demand water-stress inferences |
| 14 | Advisory Engine | Rule-based decision logic | Irrigation priorities (Priority 1, 2, 3) |
| 15 | Geospatial Output | GeoJSON, Supabase PostGIS | Fast-queryable spatial GeoJSON polygons |
| 16 | Database Integration | **Supabase (PostgreSQL + PostGIS)** | Persistent schema for polygons, predictions & logs |
| 17 | Backend API | **FastAPI + Docker on Railway** | Production REST API with `/health` & `/retrain` |
| 18 | Web Dashboard | **React + Leaflet on Cloudflare Pages** | Interactive responsive GIS visualization portal |
| 19 | MLOps: Git & DagsHub | **GitHub + DagsHub Hub** | Code, DVC remote & hosted MLflow setup |
| 20 | Data Versioning with DVC | **DVC + DagsHub Storage** | Git-linked data & model artifact storage |
| 21 | Experiment Tracking | **DagsHub-hosted MLflow** | Centralized web dashboard for runs & models |
| 22 | Reproducible ML Pipeline | **DVC DAG (`dvc.yaml`)** | Declarative reproducible pipeline (`dvc repro`) |
| 23 | Containerization | **Docker & Multi-Stage Build** | Production-ready lightweight Docker image |
| 24 | CI/CD Pipelines | **GitHub Actions** | Automated testing, container build & deployment |
| 25 | Application Health Monitoring | **UptimeRobot + FastAPI `/metrics`** | Uptime monitoring & cold-sleep prevention |
| 26 | ML Drift Monitoring | **Evidently AI** | Automated drift reports (KS-test / Wasserstein) |
| 27 | Automated Retraining | **Cron + DVC + API Trigger** | Auto-retrain, champion-challenger validation & deploy |

---

# 6. Phase 0 — Scope and Data Validation

## Objective

Validate the availability and compatibility of all required datasets before starting ML development.

## Tasks

- [ ] Finalize problem statement.
- [ ] Define project scope.
- [ ] Select one canal command area.
- [ ] Identify official command-area boundary data.
- [ ] Identify agricultural / land-use data.
- [ ] Identify crop information.
- [ ] Identify satellite data.
- [ ] Identify weather data.
- [ ] Investigate water-stress / water-deficit reference data.
- [ ] Determine spatial resolution.
- [ ] Determine temporal resolution.
- [ ] Determine available historical period.
- [ ] Decide spatial prediction unit.
- [ ] Decide target variable.
- [ ] Document all assumptions.

## Critical Decision

Choose between:

### Option A — Classification

```text
LOW STRESS
MODERATE STRESS
HIGH STRESS
```

### Option B — Regression

```text
Water Deficit = X mm
```

or:

```text
Water Deficit = X %
```

### Recommended MVP

Use **Low / Moderate / High water-stress classification** unless a reliable continuous water-deficit reference dataset is available.

---

# 7. Phase 1 — Study Area Selection

## Objective

Select one manageable canal command area for development and demonstration.

## Tasks

- [ ] Obtain command-area boundaries.
- [ ] Explore available irrigation projects.
- [ ] Select one study area.
- [ ] Download boundary data.
- [ ] Validate geometry.
- [ ] Check coordinate reference system.
- [ ] Calculate command-area size.
- [ ] Visualize the boundary.
- [ ] Identify agricultural regions.
- [ ] Define final Area of Interest (AOI).

## Output

```text
data/raw/study_area/
└── command_area.geojson
```

---

# 8. Phase 2 — Satellite Data Acquisition

## Primary Satellite

**Sentinel-2**

Sentinel-2 provides the main optical remote-sensing inputs for vegetation and water-related feature extraction.

## Important Bands

```text
Blue  (B2 - 490 nm)
Green (B3 - 560 nm)
Red   (B4 - 665 nm)
NIR   (B8 - 842 nm)
SWIR  (B11 - 1610 nm, B12 - 2190 nm)
```

## Tasks

- [ ] Identify Sentinel-2 observations covering the AOI.
- [ ] Collect historical observations.
- [ ] Filter observations based on cloud coverage.
- [ ] Download/access required imagery via STAC API / Copernicus Browser.
- [ ] Store raw satellite data.
- [ ] Record acquisition dates.
- [ ] Store metadata.
- [ ] Verify spatial coverage.

## Output

```text
data/raw/satellite/
```

---

# 9. Phase 3 — Weather Data Acquisition

## Required Variables

```text
Rainfall (mm)
Temperature (Max, Min, Mean °C)
Relative Humidity (%)
Wind Speed (m/s)
Solar Radiation (MJ/m²)
Reference Evapotranspiration (ETo mm/day)
```

## Tasks

- [ ] Select weather-data source (Open-Meteo API - Free, no API key required).
- [ ] Collect historical observations.
- [ ] Match observations with satellite dates.
- [ ] Aggregate weather variables spatially/temporally.
- [ ] Handle missing values.
- [ ] Generate 8-day weather statistics.
- [ ] Generate longer-window statistics (16-day, 30-day cumulative rainfall).

## Example Features

```text
Rainfall_8day
Rainfall_16day
Rainfall_30day

Temperature_8day
Temperature_30day
```

## Output

```text
data/raw/weather/
```

---

# 10. Phase 4 — Crop Information

Crop water requirements vary according to crop type and growth stage.

## Required Information

```text
Crop Type
Growth Stage
Season (Kharif / Rabi / Zaid)
```

Optional:

```text
Crop Coefficient (Kc)
```

## Tasks

- [ ] Identify dominant crops in command area.
- [ ] Obtain crop map / agricultural land-use mask.
- [ ] Determine crop growth stages.
- [ ] Encode crop type (One-Hot or Target Encoding).
- [ ] Encode growth stage.
- [ ] Match crop information to spatial units.

## Output

```text
data/processed/crop_data/
```

---

# 11. Phase 5 — Geospatial Preprocessing

## Objective

Bring satellite, weather, crop, boundary, and reference datasets into a common spatial framework.

## Workflow

```text
Raw Satellite
      ↓
Quality Filtering
      ↓
Cloud Masking (SCL band)
      ↓
Band Alignment
      ↓
CRS Standardization (UTM Projection)
      ↓
AOI Clipping
      ↓
Agricultural Mask
      ↓
Spatial Unit Extraction (Field Parcels or Regular Grids)
```

## Tasks

- [ ] Standardize CRS to appropriate UTM Zone.
- [ ] Clip imagery to AOI.
- [ ] Apply cloud masking using Sentinel-2 Scene Classification Layer (SCL).
- [ ] Remove invalid observations.
- [ ] Align satellite bands to 10m/20m spatial resolution.
- [ ] Resample where required.
- [ ] Create agricultural mask.
- [ ] Handle missing pixels via spatial interpolation.
- [ ] Generate spatial units.
- [ ] Validate polygon geometry.

## Spatial Unit

Preferred:

```text
Field / Parcel Boundaries (GeoJSON polygons)
```

Fallback:

```text
Regular Grid Cells (e.g., 100m x 100m or 250m x 250m)
```

If exact farm boundaries are unavailable, generate predictions at grid-cell level and clearly describe the system as **grid-level advisory**.

---

# 12. Phase 6 — Feature Engineering

## Objective

Convert raw satellite, weather, crop, and temporal information into ML-ready features.

---

## 12.1 Vegetation Features

### NDVI

```text
NDVI = (NIR - Red) / (NIR + Red)
```

Use NDVI as a vegetation-condition indicator.

### NDWI

```text
NDWI = (NIR - SWIR) / (NIR + SWIR)   (Gao formulation for leaf water content)
```

Use NDWI to capture vegetation canopy water content.

### EVI

```text
EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
```

Use EVI as an additional canopy density and vigor feature.

---

## 12.2 Spectral Features

```text
Red (B4)
Green (B3)
NIR (B8)
SWIR1 (B11)
SWIR2 (B12)
NDVI
NDWI
EVI
```

---

## 12.3 Weather Features

```text
Rainfall (Cumulative 8d, 16d, 30d)
Mean Temperature (8d, 30d)
Max Temperature (8d)
Vapor Pressure Deficit (VPD)
Reference Evapotranspiration (ETo)
```

---

## 12.4 Temporal Features

Examples:

```text
NDVI_current
NDVI_previous
NDVI_change (NDVI_current - NDVI_previous)
NDVI_8day_mean

NDWI_current
NDWI_previous
NDWI_change

Rainfall_8day
Rainfall_16day
Rainfall_30day

Temperature_8day
Temperature_30day
```

---

## 12.5 Crop Features

```text
Crop Type
Growth Stage
Season
Crop Coefficient (Kc)
```

---

## Example Feature Table

```text
┌─────────┬──────┬──────┬──────┬──────────┬──────┬────────────┬─────────────┐
│ Area ID │ NDVI │ NDWI │ EVI  │ Rainfall │ Temp │ Crop       │ Stress_Class│
├─────────┼──────┼──────┼──────┼──────────┼──────┼────────────┼─────────────┤
│ 1       │ 0.65 │ 0.42 │ 0.58 │ 24.5     │ 26.1 │ Wheat      │ LOW         │
│ 2       │ 0.28 │-0.12 │ 0.22 │ 0.0      │ 35.4 │ Sugarcane  │ HIGH        │
└─────────┴──────┴──────┴──────┴──────────┴──────┴────────────┴─────────────┘
```

## Output

```text
data/features/features.csv
```

---

# 13. Phase 7 — Target / Ground Truth Generation

## Objective

Create the target variable that the ML model will learn.

This is one of the most foundational parts of AgroScope.

## Possible Sources

```text
Ground observations / Soil moisture sensors
        OR
Reference water-deficit products (e.g., OpenET, WaPOR)
        OR
Crop water balance deficit proxy:
Deficit = (Kc * ETo) - Effective_Rainfall - Soil_Moisture_Proxy
```

## Conceptual Pipeline

```text
Reference Water Information / Water Balance Deficit
          ↓
Water Deficit / Stress Indicator
          ↓
Scientifically Defensible Threshold Definition
          ↓
Stress Labels (0: LOW, 1: MODERATE, 2: HIGH)
```

## Classification Target

```text
0 → LOW STRESS
1 → MODERATE STRESS
2 → HIGH STRESS
```

## Critical Scientific Principle

```text
NDVI ≠ Direct Water Deficit
```

NDVI is an indicator of green biomass, but low NDVI alone can also be due to planting stage, disease, or harvesting. Combining NDWI (canopy water), weather deficit, and crop stage prevents false stress labeling.

---

# 14. Phase 8 — ML Dataset Construction

Combine all processed data.

```text
Satellite Features + Weather Features + Crop Features + Temporal Features + Target Labels
                                ↓
                      Final ML Dataset (CSV / Parquet)
                                ↓
                      Tracked via DVC to DagsHub
```

## Data Period Strategy

| Purpose | Data Period | Source |
|:---|:---|:---|
| **Model Training** | Oct 2023 – Mar 2024 (Rabi Season) | Satellite + Weather (already fetched) |
| **Model Validation** | Oct 2024 – Mar 2025 (Rabi Season) | Satellite + Weather (to be fetched later) |
| **Live Inference** | Latest 8-day cycle (2026 onward) | Satellite + Weather (on-demand at prediction time) |

> **Rationale**: The Rabi season (Oct–Mar) is the primary irrigated cropping window for the Nira Left Bank Canal command area (wheat, gram, sugarcane). It also has the lowest cloud cover in Maharashtra, giving the most complete and usable Sentinel-2 observations.

---

# 15. Phase 9 — Exploratory Data Analysis

## Tasks

- [ ] Analyze feature distributions and normality.
- [ ] Check outliers in spectral indices and weather metrics.
- [ ] Examine target distribution and class imbalance.
- [ ] Feature correlation matrix (Pearson and Spearman).
- [ ] Bivariate distributions: NDVI vs Stress, NDWI vs Stress, Rainfall vs Stress.
- [ ] Temporal trends across the 8-day cycles.

---

# 16. Phase 10 — Machine Learning Model Strategy

```text
Baseline: Random Forest Classifier
Main Model: XGBoost Classifier (Tuned with Hyperopt / Optuna)
```

## Input Features

```text
Satellite Features (Spectral + Indices)
+
Weather Features (Rainfall lags, Temperature, ETo)
+
Crop Features (Crop Type, Growth Stage, Kc)
+
Temporal Trends (NDVI/NDWI deltas)
```

## Output

```text
Class Probabilities: P(LOW), P(MODERATE), P(HIGH)
Predicted Class: argmax(Probabilities)
```

---

# 17. Phase 11 — Model Training & Validation Strategy

## Spatial and Temporal Leakage Prevention

Because agricultural observations are clustered in space and time:
- Use **Spatial Block Cross-Validation** (train on group of fields, validate on distant field blocks).
- Or use **Temporal Holdout Validation** (train on preceding cycles, evaluate on future 8-day cycles).
- Never use random naive k-fold splitting!

---

# 18. Phase 12 — Model Evaluation

## Metrics

```text
Macro-Averaged F1 Score (Primary benchmark)
High Stress Class Recall (Critical for preventing crop failure)
Multi-class Confusion Matrix
ROC-AUC per class
```

---

# 19. Phase 13 — Model Explainability (SHAP)

## Objective

Explain why a particular spatial unit was predicted as stressed.

```text
Area: UNIT_00142
Prediction: HIGH STRESS (Confidence: 87%)

Important Factors:
↓ Low NDWI (-0.18)      — Contributes +38% to High Stress
↓ Zero 16-day Rainfall  — Contributes +24% to High Stress
↑ High 8-day Temp (34°) — Contributes +18% to High Stress
```

SHAP TreeExplainer will be integrated into the backend API to return feature attributions on demand.

---

# 20. Phase 14 — Water Stress Prediction Pipeline

```text
New Sentinel-2 Tile + Weather API Data
          ↓
Compute Indices & Lagged Features
          ↓
XGBoost Model Inference
          ↓
Predicted Stress Class + Probabilities + Timestamp
```

---

# 21. Phase 15 — Irrigation Advisory Engine

The ML model predicts **water stress**. The advisory engine converts stress into **irrigation priority**.

## Advisory Logic

```text
HIGH STRESS + High Temperature + Sensitive Growth Stage (e.g. Flowering/Heading)
         ↓
PRIORITY 1 (Immediate Irrigation Required — Next 24-48 Hours)

MODERATE STRESS or High Stress in Non-Critical Stage
         ↓
PRIORITY 2 (Schedule Irrigation Within 3-5 Days)

LOW STRESS + Adequate Soil Moisture
         ↓
PRIORITY 3 (Normal / No Immediate Action)
```

---

# 22. Phase 16 — Geospatial Output

Spatial predictions are exported as **GeoJSON FeatureCollections** and stored in PostGIS:
- `stress_map.geojson`
- `irrigation_priority_map.geojson`

---

# 23. Phase 17 — Backend API (FastAPI on Railway)

## Technology

```text
FastAPI + Uvicorn + Pydantic + Supabase-py / asyncpg
Deployment: Railway (Free Container Instance)
```

## Responsibilities

The API should:

- Load the production XGBoost model from local bundle or DagsHub/Supabase cache.
- Query command-area geometries and latest stress predictions from Supabase.
- Receive prediction requests for on-demand spatial units.
- Convert stress scores to explainable irrigation priorities.
- Expose health endpoints for UptimeRobot monitoring.
- Expose automated retraining trigger endpoints (`/retrain`) for webhooks and manual overrides.
- Expose model and data drift report summaries from Evidently AI.

## Production Endpoints

| Method | Endpoint | Purpose | Consumers |
|---|---|---|---|
| `GET` | `/health` | Liveness & readiness probe | UptimeRobot, Railway Healthcheck |
| `GET` | `/metrics` | Request latency, memory, inference count | Application Monitoring |
| `GET` | `/study-area` | Boundary GeoJSON polygon of the command area | Dashboard Map Layer |
| `GET` | `/available-dates` | List of processed 8-day satellite cycle timestamps | Dashboard Date Picker |
| `GET` | `/stress-map` | GeoJSON FeatureCollection of spatial units + stress class | Dashboard Map Layer |
| `GET` | `/irrigation-map` | GeoJSON FeatureCollection of priority zones (P1, P2, P3) | Dashboard Map Layer |
| `GET` | `/statistics` | Hectares under High/Moderate/Low stress, dominant crop breakdown | Dashboard KPI Cards |
| `GET` | `/area/{id}` | Detailed feature values (NDVI, NDWI, Temp) + SHAP explanation | Detail Drawer / Modal |
| `POST` | `/predict` | On-demand inference given custom satellite/weather features | External Integration / Testing |
| `POST` | `/retrain` | Secure webhook trigger to initiate automated model retraining | Admin / Webhook / Dashboard |
| `GET` | `/monitoring/drift` | Latest Evidently AI drift status (drift detected: true/false, p-values) | MLOps Dashboard / Alerts |

## Example Response (`GET /area/142`)

```json
{
  "area_id": "142",
  "crop_type": "Wheat",
  "growth_stage": "Heading",
  "observation_date": "2024-03-12",
  "water_stress": {
    "class": "HIGH",
    "probability": 0.87,
    "distribution": {
      "LOW": 0.04,
      "MODERATE": 0.09,
      "HIGH": 0.87
    }
  },
  "irrigation_advisory": {
    "priority": 1,
    "urgency": "Immediate (Next 24-48 Hours)",
    "recommendation": "Severe water deficit detected during critical heading phase. Canal water scheduling priority."
  },
  "key_drivers": [
    {"feature": "NDWI", "value": -0.18, "impact": "Strong contributor to High Stress"},
    {"feature": "Rainfall_16day", "value": 0.0, "impact": "No rainfall in previous 16 days"},
    {"feature": "Max_Temp_8day", "value": 34.2, "impact": "High evaporative demand"}
  ]
}
```

---

# 24. Phase 18 — Database Layer: Supabase (PostgreSQL + PostGIS)

## Objective

Provide persistent, performant, and zero-cost relational and geospatial storage for command-area boundaries, spatial unit geometries, historical 8-day cycle predictions, and drift audit trails.

## Technology

```text
Supabase Free Tier (Managed PostgreSQL 15+ with PostGIS extension)
500 MB Storage (Sufficient for >500,000 spatial prediction records)
Direct connection via asyncpg / SQLAlchemy + REST API client
```

## Database Schema Design

```sql
-- Enable PostGIS for Spatial Queries
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Canal Command Area Metadata
CREATE TABLE study_areas (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    state VARCHAR(50),
    total_area_hectares NUMERIC(10, 2),
    boundary GEOMETRY(Polygon, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Spatial Units (Fields / Grid Parcels within Command Area)
CREATE TABLE spatial_units (
    id VARCHAR(50) PRIMARY KEY, -- e.g., 'UNIT_00142'
    study_area_id INT REFERENCES study_areas(id),
    crop_type VARCHAR(50),
    soil_type VARCHAR(50),
    centroid GEOMETRY(Point, 4326),
    geom GEOMETRY(Polygon, 4326) NOT NULL
);

CREATE INDEX idx_spatial_units_geom ON spatial_units USING GIST(geom);

-- 3. Water Stress Predictions (8-Day Cycle History)
CREATE TABLE stress_predictions (
    id BIGSERIAL PRIMARY KEY,
    unit_id VARCHAR(50) REFERENCES spatial_units(id),
    cycle_date DATE NOT NULL,
    ndvi NUMERIC(5, 4),
    ndwi NUMERIC(5, 4),
    evi NUMERIC(5, 4),
    rainfall_8day NUMERIC(6, 2),
    temp_8day NUMERIC(5, 2),
    stress_class VARCHAR(20) NOT NULL, -- 'LOW', 'MODERATE', 'HIGH'
    confidence NUMERIC(4, 3) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_predictions_cycle_unit ON stress_predictions(cycle_date, unit_id);

-- 4. Irrigation Advisories
CREATE TABLE irrigation_advisories (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES stress_predictions(id),
    priority INT NOT NULL CHECK (priority IN (1, 2, 3)),
    recommended_action TEXT NOT NULL,
    advisory_date DATE NOT NULL
);

-- 5. MLOps Drift & Audit Logs
CREATE TABLE ml_drift_logs (
    id SERIAL PRIMARY KEY,
    eval_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dataset_version VARCHAR(50),
    model_version VARCHAR(50),
    data_drift_detected BOOLEAN,
    prediction_drift_detected BOOLEAN,
    drift_score NUMERIC(5, 4),
    drift_report_url TEXT
);
```

---

# 25. Phase 19 — Web Dashboard (React + Leaflet on Cloudflare Pages)

## Technology

```text
Frontend: React (Vite) + Leaflet / React-Leaflet + Lucide Icons + TailwindCSS / Modern Vanilla CSS
Hosting: Cloudflare Pages (100% Free, Global Edge CDN, Automated Git Deployments)
Communication: HTTPS calls to Railway FastAPI Backend
```

## Cloudflare Pages Deployment Highlights

- **Zero-Cost Forever**: Cloudflare Pages provides unlimited bandwidth and requests on its free tier.
- **Continuous Git Integration**: Every push to the `main` branch or a Pull Request triggers an automated edge build.
- **Fast Global Delivery**: Maps, vector GeoJSON polygons, and dashboard assets load in milliseconds worldwide.

## Dashboard Layout & Interactive GIS Features

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ 🛰️ AgroScope — Satellite Crop Water Stress & Irrigation Advisory System   │
├───────────────────────────────────────────────────────────────────────────┤
│ [Command Area: Nira Left Canal ▼]  [Cycle: 2024-03-12 ▼]  [Crop: All ▼]   │
├─────────────────────────────────────┬─────────────────────────────────────┤
│                                     │  📊 AGGREGATE SUMMARY (CYCLE METRICS)│
│                                     ├─────────────────────────────────────┤
│         INTERACTIVE LEAFLET         │  🔴 High Stress:     1,420 ha (32%) │
│               GIS MAP               │  🟡 Moderate Stress: 1,850 ha (41%) │
│                                     │  🟢 Low Stress:      1,210 ha (27%) │
│  • Toggle Layers:                   ├─────────────────────────────────────┤
│    [x] Water Stress Classification  │  💧 IRRIGATION ADVISORY STATUS       │
│    [ ] Irrigation Priority (P1/2/3) ├─────────────────────────────────────┤
│    [ ] NDVI Color Ramp              │  Priority 1 (Urgent): 1,150 ha      │
│    [ ] Canal Network Polylines      │  Priority 2 (Medium): 1,600 ha      │
│                                     │  Priority 3 (Normal): 1,730 ha      │
│  • Interactive Polygon Click:       ├─────────────────────────────────────┤
│    Click any farm/grid parcel to    │  🔍 PARCEL INSPECTOR (#UNIT_00142)  │
│    trigger real-time SHAP analysis  ├─────────────────────────────────────┤
│    and advisory details             │  Crop: Wheat | Stress: HIGH (87%)   │
│                                     │  NDVI: 0.32 | NDWI: -0.18 | P1      │
│                                     │  SHAP: NDWI (-38%), Rain (-24%)     │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

---

# 26. Phase 20 — MLOps: Git, GitHub & DagsHub Integration

We maintain code and lightweight metadata in **GitHub**, while versioning large geospatial datasets and ML models in **DagsHub** via DVC.

```text
GitHub (Free)                 DagsHub (Free)
  ├── Source Code               ├── Remote DVC Storage (Satellite / Rasters)
  ├── CI/CD Workflows           ├── Hosted MLflow Tracking Server
  ├── Pipeline Specs (dvc.yaml) └── Central Model Registry
  └── Tests & Documentation
```

### Free Setup Configuration

1. Create a repository on GitHub (`tanishkaphad/AgroScope`).
2. Connect the GitHub repository to [DagsHub](https://dagshub.com) in one click.
3. DagsHub automatically generates a free remote storage URL for DVC and a managed MLflow tracking URI.

---

# 27. Phase 21 — Data Versioning with DVC & DagsHub Remote Storage

Geospatial satellite images (Sentinel-2 granules) and preprocessed raster matrices are too large for Git. We use **DVC (Data Version Control)** with DagsHub free storage backend.

## DVC Storage Architecture

```text
Local Machine / CI Runner                    DagsHub Remote Storage
├── data/raw/satellite/ (100MB+)             https://dagshub.com/tanishkaphad/AgroScope.dvc
├── data/processed/features.csv ──[dvc push]──► ├── data/raw/... (hashed blobs)
├── models/best_model.json      ──[dvc pull]──◄ └── models/... (hashed blobs)
└── data/...dvc (Pointer Files tracked in Git)
```

## DVC Commands for AgroScope

```bash
# 1. Initialize DVC
dvc init

# 2. Add DagsHub remote storage
dvc remote add origin https://dagshub.com/tanishkaphad/AgroScope.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user tanishkaphad
dvc remote modify origin --local password $DAGSHUB_USER_TOKEN

# 3. Track raw and feature data
dvc add data/raw/satellite
dvc add data/processed
dvc add data/features/final_dataset.csv
dvc add models/xgboost_stress_model.json

# 4. Push large files to DagsHub free storage
dvc push -r origin

# 5. Commit lightweight .dvc metadata files to Git
git add data/*.dvc models/*.dvc .dvc/config
git commit -m "chore(dvc): track dataset and model artifacts on DagsHub"
```

---

# 28. Phase 22 — Experiment Tracking & Model Registry with DagsHub-Hosted MLflow

Instead of paying for a dedicated MLflow server or running fragile local servers, AgroScope utilizes **DagsHub's free hosted MLflow service**.

## Configuration

Set environment variables in your local environment and GitHub Actions secrets:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/tanishkaphad/AgroScope.mlflow"
export MLFLOW_TRACKING_USERNAME="tanishkaphad"
export MLFLOW_TRACKING_PASSWORD="$DAGSHUB_USER_TOKEN"
```

## Logged Parameters & Metrics

```python
import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri("https://dagshub.com/tanishkaphad/AgroScope.mlflow")
mlflow.set_experiment("agroscope-water-stress")

with mlflow.start_run(run_name="xgboost-hyperopt-v2"):
    # Log Hyperparameters
    mlflow.log_params({
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "observation_window": "8-day"
    })
    
    # Log Metrics
    mlflow.log_metrics({
        "val_macro_f1": 0.842,
        "val_high_stress_recall": 0.891,
        "val_accuracy": 0.865
    })
    
    # Log Artifacts
    mlflow.log_artifact("reports/confusion_matrix.png")
    mlflow.log_artifact("reports/feature_importance.png")
    mlflow.log_artifact("reports/shap_summary.png")
    
    # Register Champion Model
    mlflow.xgboost.log_model(
        xgb_model=best_model,
        artifact_path="model",
        registered_model_name="AgroScope_XGBoost"
    )
```

---

# 29. Phase 23 — Reproducible ML Pipeline with DVC (`dvc.yaml`)

The entire ML lifecycle from raw satellite ingestion to evaluation and drift verification is declared as a reproducible DAG in `dvc.yaml`:

```yaml
stages:
  acquire_weather:
    cmd: python src/ingestion/weather_ingest.py --aoi data/raw/study_area/command_area.geojson
    deps:
      - src/ingestion/weather_ingest.py
      - data/raw/study_area/command_area.geojson
    outs:
      - data/raw/weather/weather_8day.csv

  preprocess_satellite:
    cmd: python src/preprocessing/process_sentinel.py
    deps:
      - src/preprocessing/process_sentinel.py
      - data/raw/satellite
    outs:
      - data/processed/spectral_indices.parquet

  engineer_features:
    cmd: python src/features/build_features.py
    deps:
      - src/features/build_features.py
      - data/processed/spectral_indices.parquet
      - data/raw/weather/weather_8day.csv
    outs:
      - data/features/final_dataset.csv

  train_evaluate:
    cmd: python src/training/train.py --config configs/train_config.yaml
    deps:
      - src/training/train.py
      - configs/train_config.yaml
      - data/features/final_dataset.csv
    outs:
      - models/xgboost_stress_model.json
    metrics:
      - reports/model_metrics.json:
          cache: false

  drift_evaluation:
    cmd: python src/monitoring/drift_check.py
    deps:
      - src/monitoring/drift_check.py
      - data/features/final_dataset.csv
    outs:
      - reports/evidently_drift_report.html
```

When new 8-day satellite data arrives, executing `dvc repro` automatically runs only the affected stages!

---

# 30. Phase 24 — Dockerization & Railway Deployment (Backend API)

The backend FastAPI application is containerized using a lightweight multi-stage Docker build optimized for GDAL and geospatial Python dependencies.

## Multi-Stage `Dockerfile`

```dockerfile
# Build Stage
FROM python:3.10-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Stage
FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal30 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## `railway.toml` Configuration

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn api.app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 120
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

Railway automatically triggers deployment whenever a new commit lands on the `main` branch or when triggered via GitHub Actions.

---

# 31. Phase 25 — Complete CI/CD with GitHub Actions

AgroScope employs GitHub Actions (2,000 free minutes/month) to run tests, validate data schemas, deploy services, and automate MLOps retraining.

## 1. Pull Request CI Workflow (`.github/workflows/ci.yml`)

- Runs `flake8` and `black` code linting.
- Runs `pytest` across unit and integration tests (`tests/test_features.py`, `tests/test_model.py`, `tests/test_api.py`).
- Validates that geospatial index formulas (NDVI, NDWI, EVI) maintain boundary bounds `[-1, 1]`.

## 2. Production CD Workflow (`.github/workflows/deploy.yml`)

- On merge to `main`:
  - Triggers **Railway Deployment** via Railway CLI / webhook.
  - Builds React frontend (`npm run build`) and publishes to **Cloudflare Pages** using `cloudflare/pages-action@v1`.
  - Runs end-to-end smoke test on production `/health` endpoint.

## 3. Automated 8-Day Retraining Workflow (`.github/workflows/retrain.yml`)

- Triggers on `cron: '0 2 */8 * *'` (every 8 days at 02:00 UTC) or manual `workflow_dispatch`.

---

# 32. Phase 26 — Application Health & Keep-Alive Monitoring (UptimeRobot)

Free-tier serverless and container platforms (like Railway, Render) can put containers to sleep after periods of inactivity.

## Zero-Cost Monitoring Architecture

```text
UptimeRobot (Free Cloud Service)
   │
   ├─► Every 5 minutes: GET https://agroscope-api.up.railway.app/health
   │   ├── Verifies HTTP 200 OK
   │   ├── Verifies database connection to Supabase
   │   ├── Keeps Railway container warm (eliminates cold starts!)
   │   └── Sends instant email / Discord / Slack alert if downtime occurs
```

## FastAPI `/health` Endpoint Implementation

```python
@app.get("/health", tags=["Monitoring"])
async def health_check():
    db_ok = await check_supabase_connection()
    model_loaded = current_model is not None
    return {
        "status": "HEALTHY" if (db_ok and model_loaded) else "DEGRADED",
        "database": "CONNECTED" if db_ok else "DISCONNECTED",
        "model": "LOADED" if model_loaded else "NOT_FOUND",
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

# 33. Phase 27 — ML Drift Monitoring with Evidently AI

AgroScope utilizes the open-source **Evidently AI** framework to evaluate feature drift, prediction drift, and data quality without recurring SaaS subscription costs.

## Monitored Variables & Drift Tests

| Type | Features Monitored | Statistical Test | Alert Threshold |
|---|---|---|---|
| **Data Drift** | NDVI, NDWI, EVI | Kolmogorov-Smirnov (KS) test | p-value < 0.05 |
| **Data Drift** | 8d Rainfall, 8d Temperature | Wasserstein Distance | Drift score > 0.15 |
| **Prediction Drift** | Water Stress Class (`LOW`, `MOD`, `HIGH`) | Chi-square / Jensen-Shannon | Significant shift in class proportions |
| **Data Quality** | Missing values, out-of-range sensor pixels | Rule-based bounds check | > 2% invalid pixels |

## Drift Check Automation Script (`src/monitoring/drift_check.py`)

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd

def run_drift_analysis(reference_path, current_cycle_path):
    ref_data = pd.read_csv(reference_path)
    cur_data = pd.read_csv(current_cycle_path)
    
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=ref_data, current_data=cur_data)
    
    # Save standalone interactive HTML report and JSON summary
    report.save_html("reports/drift_report.html")
    report_dict = report.as_dict()
    
    drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]
    return drift_detected
```

If dataset drift is detected, an alert is logged to Supabase `ml_drift_logs` and an automated GitHub Action issue is created.

---

# 34. Phase 28 — Fully Automated Model Retraining & Promotion Pipeline

Retraining is triggered automatically through three independent mechanisms:

```text
               RETRAINING TRIGGER SOURCES
  ┌──────────────────────────────────────────────────┐
  │  1. Scheduled: GitHub Actions Cron (Every 8 Days)│
  │  2. Data Drift: Evidently AI alerts on new data  │
  │  3. Manual: Admin POST /retrain or GH Dispatch   │
  └─────────────────────────┬────────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 1. Pull Latest Data & Run DVC Reproduction   │
     │    `dvc repro` updates features              │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 2. Train Challenger Model (XGBoost)          │
     │    Log runs, metrics to DagsHub MLflow       │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 3. Champion vs Challenger Evaluation Gate    │
     │    Challenger Macro-F1 > Champion Macro-F1?  │
     └──────────────┬───────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │ YES                 │ NO
         ▼                     ▼
┌──────────────────┐  ┌──────────────────────────────┐
│ PROMOTE MODEL    │  │ RETAIN CHAMPION              │
│ Tag "Production" │  │ Log reason for non-promotion │
│ in DagsHub MLflow│  │ Maintain current Railway     │
│ Auto-deploy      │  │ container                    │
│ to Railway       │  └──────────────────────────────┘
└──────────────────┘
```

## Model Promotion Validation Gate

A new model replaces the active production model **only if**:
1. `Challenger_Macro_F1 >= Champion_Macro_F1 + 0.01` (to prevent unnecessary churn).
2. `High_Stress_Recall >= 0.85` (ensuring critical water stress is never missed).
3. Data quality checks in Evidently AI report zero corrupted feature columns.

---

# 35. 8-Day Operational Cycle

AgroScope is designed around an 8-day observation cycle aligned with Sentinel-2 revisit frequencies.

```text
DAY 0: Observation Day
  ├── Sentinel-2 image acquired & cloud-filtered
  ├── Open-Meteo 8-day rainfall & temperature aggregated
  ├── Preprocessing & spectral indices computed (NDVI, NDWI, EVI)
  ├── Run Evidently AI drift check against reference baseline
  ├── Predict water stress classes & assign irrigation priorities
  ├── Upsert results into Supabase (PostgreSQL + PostGIS)
  └── Leaflet dashboard updates with new cycle GeoJSON
  
DAY 1-7: Advisory & Field Monitoring Window
  ├── Canal irrigation officers review Priority 1 zones
  ├── Farmers consult parcel-level advisory
  ├── UptimeRobot continuously pings API /health (zero cold starts)
  └── Supabase logs user feedback / soil ground truth where available

DAY 8: Next Cycle & Retraining Check
  ├── New Sentinel-2 observation arrives
  └── Automated GitHub Actions retrain pipeline checks champion-challenger gate
```

---

# 36. Complete End-to-End Workflow Diagram

```text
                                START
                                  │
                                  ▼
                        Select Command Area (AOI)
                                  │
                                  ▼
                         Validate AOI Data
                                  │
                                  ▼
                     Acquire Sentinel-2 Imagery
                                  │
                                  ▼
                     Acquire Open-Meteo Weather
                                  │
                                  ▼
                         Crop & Reference Data
                                  │
                                  ▼
                   Geospatial Preprocessing (GDAL)
                                  │
                                  ▼
                      Feature Engineering (Indices)
                                  │
                                  ▼
                   Version Dataset with DVC (DagsHub)
                                  │
                                  ▼
                         Exploratory Data Analysis
                                  │
                                  ▼
                      Train Models (RF Baseline, XGBoost)
                                  │
                                  ▼
                   Track in DagsHub MLflow Registry
                                  │
                                  ▼
                   Run Evidently AI Drift Monitoring
                                  │
                                  ▼
                   Generate GeoJSON & Inferences
                                  │
                                  ▼
                   Store in Supabase (PostGIS Tables)
                                  │
                                  ▼
                   Serve via FastAPI Backend on Railway
                                  │
                                  ▼
                   Visualize on React Dashboard on Cloudflare Pages
                                  │
                                  ▼
                   Monitor Health via UptimeRobot
                                  │
                                  ▼
                   GitHub Actions Automated 8-Day Retraining
                                  │
                                  └─────────────────┐
                                                    ▼
                                            NEXT 8-DAY CYCLE
```

---

# 37. Repository Structure (Automated MLOps & Free Cloud Stack)

```text
AgroScope/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # PR linting and test validation
│       ├── deploy.yml                # Auto-deploy to Railway & Cloudflare Pages
│       └── retrain.yml               # 8-day cron automated retraining pipeline
│
├── data/                             # Versioned via DVC with DagsHub storage remote
│   ├── raw/
│   │   ├── satellite/                # Sentinel-2 raw granules (.dvc tracked)
│   │   ├── weather/                  # Open-Meteo aggregated climate CSVs
│   │   ├── crop/                     # Crop coefficients & land-use masks
│   │   ├── reference/                # Deficit validation & reference ground truth
│   │   └── study_area/               # command_area.geojson
│   ├── processed/
│   └── features/
│       └── final_dataset.csv.dvc
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_satellite_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda.ipynb
│   └── 05_model_experiments.ipynb
│
├── src/
│   ├── ingestion/                    # Sentinel STAC & weather scrapers
│   ├── preprocessing/                # GDAL/Rasterio cloud-masking & spatial clipping
│   ├── features/                     # NDVI, NDWI, EVI & weather lag engineering
│   ├── target/                       # Water stress proxy calculation
│   ├── training/                     # XGBoost training & hyperopt routines
│   ├── inference/                    # Batch & real-time inference wrappers
│   ├── advisory/                     # Rule-based priority assignment logic
│   └── monitoring/                   # Evidently AI drift test scripts
│
├── supabase/
│   ├── migrations/
│   │   └── 01_initial_schema.sql     # PostGIS tables, spatial indices, views
│   └── seed.sql                      # Initial command area boundary seeds
│
├── models/
│   └── xgboost_stress_model.json.dvc # Tracked via DVC + logged to DagsHub MLflow
│
├── api/
│   └── app/
│       ├── main.py                   # FastAPI application entrypoint
│       ├── routers/                  # /health, /stress-map, /predict, /retrain
│       └── db.py                     # Supabase PostGIS connection pooling
│
├── dashboard/                        # React + Leaflet (Deploys to Cloudflare Pages)
│   ├── src/
│   │   ├── components/               # LeafletMap, StatsCards, AdvisoryDrawer
│   │   ├── services/api.js           # Calls to Railway FastAPI Backend
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
│
├── configs/
│   ├── train_config.yaml
│   └── monitoring_config.yaml
│
├── reports/
│   ├── model_metrics.json
│   └── evidently_drift_report.html
│
├── dvc.yaml                          # Multi-stage reproducible pipeline
├── params.yaml                       # Model & feature hyperparams
├── railway.toml                      # Railway backend deployment configuration
├── requirements.txt
├── Dockerfile                        # Multi-stage container build for Railway
├── .gitignore
└── README.md
```

---

# 38. 8-Week Implementation Plan

## Week 1 — Data Discovery and Study Area

### Tasks

- [ ] Finalize problem statement and canal command area.
- [ ] Obtain command-area boundary (`command_area.geojson`).
- [ ] Validate AOI coordinates and CRS in QGIS/Python.
- [ ] Connect repository to GitHub and link to DagsHub.
- [ ] Verify free access to Sentinel-2 via STAC API and weather via Open-Meteo.
- [ ] Initialize repository structure and pre-commit hooks.

### Deliverable

**Validated Study Area + GitHub & DagsHub Connected Repository**

---

## Week 2 — Geospatial and Satellite Processing

### Tasks

- [ ] Fetch Sentinel-2 granules for AOI.
- [ ] Apply cloud masking with SCL band.
- [ ] Clip to command area boundary.
- [ ] Align spectral bands to common resolution.
- [ ] Create agricultural mask to isolate non-cropland.
- [ ] Generate spatial units (field polygons or regular grid).
- [ ] Calculate baseline NDVI, NDWI, and EVI rasters.

### Deliverable

**Processed Satellite Dataset & Spatial Unit Polygons**

---

## Week 3 — Feature Engineering, Weather & Target Generation

### Tasks

- [ ] Ingest historical weather via Open-Meteo.
- [ ] Calculate 8-day, 16-day, and 30-day weather aggregations.
- [ ] Generate temporal index deltas (NDVI/NDWI change).
- [ ] Formulate water-stress target variable (Low, Moderate, High).
- [ ] Construct ML dataset merging features and targets.
- [ ] Perform comprehensive EDA (distributions, correlation matrix).
- [ ] Track dataset version with DVC and push to DagsHub.

### Deliverable

**Final ML Dataset tracked in DVC on DagsHub**

---

## Week 4 — Machine Learning & Explainability

### Tasks

- [ ] Implement spatial block cross-validation.
- [ ] Train Random Forest baseline model.
- [ ] Train and tune XGBoost classifier.
- [ ] Log parameters, metrics, and ROC-AUC curves in DagsHub MLflow.
- [ ] Compute global SHAP feature importances.
- [ ] Implement local SHAP explanations for spatial units.
- [ ] Register best model in DagsHub MLflow Model Registry.

### Deliverable

**Validated Water-Stress XGBoost Model Registered in DagsHub MLflow**

---

## Week 5 — MLOps: DVC Pipeline, DagsHub & Hosted MLflow

### Tasks

- [ ] Configure DVC remote to DagsHub free storage (`dvc remote add origin ...`).
- [ ] Create declarative multi-stage `dvc.yaml` pipeline.
- [ ] Verify full pipeline execution via `dvc repro`.
- [ ] Configure hosted MLflow tracking with DagsHub URI and credentials.
- [ ] Build Evidently AI data drift and prediction drift scripts (`src/monitoring/drift_check.py`).
- [ ] Test champion-challenger evaluation script.

### Deliverable

**Reproducible ML Pipeline with DVC + DagsHub MLflow Registry**

---

## Week 6 — Database (Supabase), FastAPI & Dashboard (Cloudflare Pages)

### Tasks

- [ ] Provision free Supabase project and enable PostGIS extension.
- [ ] Execute `supabase/migrations/01_initial_schema.sql` to build tables and spatial indices.
- [ ] Populate command-area boundary and spatial unit polygons.
- [ ] Build FastAPI backend endpoints (`/health`, `/stress-map`, `/irrigation-map`, `/area/{id}`).
- [ ] Test GeoJSON rendering performance through PostGIS queries.
- [ ] Develop React + Leaflet GIS dashboard with layer toggles and parcel click inspector.
- [ ] Connect React dashboard to FastAPI endpoints.
- [ ] Deploy frontend to Cloudflare Pages with automated Git deployment.

### Deliverable

**Live Cloudflare Pages Frontend connected to Supabase-backed FastAPI**

---

## Week 7 — Free Cloud Deployment & Automated MLOps

### Tasks

- [ ] Create multi-stage production `Dockerfile` with GDAL support.
- [ ] Deploy FastAPI container to Railway (Free Tier) with healthcheck configuration.
- [ ] Configure UptimeRobot free monitor to ping Railway `/health` every 5 minutes.
- [ ] Build Evidently AI data drift and prediction drift pipeline (`src/monitoring/drift_check.py`).
- [ ] Create GitHub Actions CI workflow (`ci.yml`) for linting and test suites.
- [ ] Create GitHub Actions CD workflow (`deploy.yml`) for automated Railway & Cloudflare releases.
- [ ] Create 8-day automated retraining workflow (`retrain.yml`) with champion-challenger gate.
- [ ] Test manual API retrain webhook (`POST /retrain`).

### Deliverable

**100% Free Deployed Production System on Railway & Cloudflare Pages with Continuous Monitoring**

---

## Week 8 — Validation, Drift Testing & Final Demo

### Tasks

- [ ] End-to-end simulated 8-day cycle test with new Sentinel-2 imagery.
- [ ] Validate drift detection alerts in Evidently AI.
- [ ] Verify zero cold-start delay thanks to UptimeRobot pings.
- [ ] Verify $0 total hosting and infrastructure cost breakdown.
- [ ] Document complete system architecture, API schemas, and MLOps lifecycle.
- [ ] Polish interactive dashboard UI/UX with smooth layer transitions.
- [ ] Prepare PPT, architecture diagrams, and high-impact live demonstration.

### Deliverable

**Final Automated AgroScope Platform + 100% Free Deployment + Project Demo & Report**

---

# 39. Testing Strategy

## Data Tests
- [x] Null / missing values handling.
- [x] Invalid geometries / self-intersecting polygons in GeoJSON.
- [x] Coordinate reference system alignment (UTM vs WGS84 EPSG:4326).
- [x] Feature bounds verification: NDVI, NDWI, EVI in `[-1.0, 1.0]`.

## ML Tests
- [x] Input feature schema validation via Pydantic.
- [x] Model load latency < 1.5s.
- [x] Inference output shape, class bounds, and probability sum `= 1.0`.
- [x] High-stress recall threshold `>= 0.85`.

## API Tests
- [x] Endpoint availability for `/health`, `/stress-map`, `/irrigation-map`.
- [x] Response schemas and valid GeoJSON serialization.
- [x] Authentication / rate-limiting on `/retrain`.

## End-to-End MLOps Pipeline Tests
- [x] Clean execution of `dvc repro`.
- [x] Successful artifact logging to DagsHub MLflow.
- [x] Automated container build and deployment to Railway.
- [x] Cloudflare Pages automated edge preview on Pull Requests.

---

# 40. Final System Output

For every field/grid unit:

```text
--------------------------------------------------
Area ID: UNIT_00142
Crop: Wheat
Growth Stage: Heading / Flowering

Water Stress Class: HIGH
Stress Probability: 87%

Irrigation Advisory: PRIORITY 1 (URGENT)
Recommended Action: Schedule canal release in next 24-48 hours

Key Drivers (SHAP Explanations):
• NDWI: -0.18 (Severe canopy moisture deficit)
• Rainfall_16day: 0.0 mm (Extended dry spell)
• Max_Temp_8day: 34.2 °C (High evaporative stress)
--------------------------------------------------
```

---

# 41. MVP Definition (100% Free Stack)

The minimum viable product provides full end-to-end capability without a single paid dependency:

```text
Sentinel-2 Imagery + Open-Meteo Weather API (Free)
                      ↓
          XGBoost Water-Stress Model
                      ↓
         Irrigation Advisory Engine
                      ↓
         Supabase PostgreSQL + PostGIS (Free Tier)
                      ↓
       FastAPI Container on Railway (Free Tier)
                      ↓
React + Leaflet Dashboard on Cloudflare Pages (Free CDN)
                      ↓
MLOps Hub: DagsHub (DVC Remote + Hosted MLflow) (Free)
                      ↓
Monitoring: Evidently AI (Open Source) + UptimeRobot (Free)
                      ↓
Automation: GitHub Actions (Free 2,000 mins/mo)
```

---

# 42. Advanced Features (Post-MVP)

- [ ] Sentinel-1 SAR Integration for all-weather radar penetration during monsoon cloud cover.
- [ ] Pan-India multi-basin command area expansion.
- [ ] Automated SMS / WhatsApp farmer advisory dispatch via Twilio / Gupshup free sandboxes.
- [ ] Reinforcement learning for dynamic canal gate release scheduling.

---

# 43. Scope Restrictions (Feasibility Guardrails)

## Included
- One defined canal command area.
- Sentinel-2 optical imagery + Open-Meteo weather.
- XGBoost classification + SHAP explanations.
- Supabase PostGIS spatial storage.
- FastAPI on Railway + React on Cloudflare Pages.
- DVC on DagsHub + DagsHub MLflow + Evidently AI + UptimeRobot.
- GitHub Actions CI/CD and 8-day cron retraining.

## Excluded Initially
- Deep learning computer vision segmentation (overkill for tabular spectral indices).
- Self-hosted Kubernetes clusters (high cost, unnecessary operational overhead).
- Real-time hardware actuator control of physical canal gates.

---

# 44. Critical Technical Principles

1. **Satellite Data Is Not Automatically Ground Truth**: Always derive stress labels from scientifically grounded water-balance proxies or established evapotranspiration indices, not arbitrary raw NDVI values.
2. **Decouple ML Prediction from Advisory Logic**: Keep model prediction (`HIGH`, `MODERATE`, `LOW`) cleanly separated from agricultural recommendations (`Priority 1`, `2`, `3`).
3. **Strict Spatial/Temporal Leakage Prevention**: Split training and test sets by geographic space or chronological observation cycles.
4. **Transparent Explainability**: Every prediction served to irrigation officials must include SHAP feature attribution.
5. **Zero-Cost Sustainable Infrastructure**: Every infrastructure component must run within generous free tiers with automated sleep-prevention.

---

# 45. Definition of Done

The AgroScope project is complete when:

- [x] Canal command area is digitized and stored in Supabase PostGIS.
- [x] Automated ingestion scripts retrieve Sentinel-2 and Open-Meteo weather data.
- [x] DVC pipeline versions all raw data, processed features, and models to DagsHub.
- [x] XGBoost model is trained, evaluated, and registered in DagsHub-hosted MLflow.
- [x] SHAP explanations are computed for spatial unit predictions.
- [x] FastAPI backend is containerized in Docker and deployed on Railway with passing `/health` checks.
- [x] React + Leaflet GIS dashboard is deployed on Cloudflare Pages with live map rendering.
- [x] Supabase stores and serves spatial polygons, predictions, and advisory records.
- [x] Evidently AI automatically calculates data drift and prediction drift.
- [x] UptimeRobot pings the backend every 5 minutes with 99.9%+ uptime.
- [x] GitHub Actions automated 8-day cron retrains and evaluates champion vs challenger models.
- [x] Total recurring hosting cost is verified at **$0.00 / month**.
- [x] Full documentation, PPT, and live demonstration are prepared.

---

# 46. Immediate Next Action

```text
STEP 1: Select canal command area & obtain GeoJSON boundary
STEP 2: Create GitHub repository & connect to DagsHub (DVC & MLflow)
STEP 3: Provision free Supabase project & run schema migrations
STEP 4: Set up automated Sentinel-2 & Open-Meteo data ingestion
STEP 5: Build feature engineering pipeline & generate target labels
STEP 6: Train baseline & XGBoost model; log runs to DagsHub MLflow
STEP 7: Build FastAPI & deploy container to Railway
STEP 8: Build React + Leaflet dashboard & deploy to Cloudflare Pages
STEP 9: Configure UptimeRobot & GitHub Actions CI/CD + Retrain cron
```

---

# 47. Final Project Definition

**AgroScope** is an end-to-end geospatial AI and MLOps platform combining satellite remote sensing, meteorological data, crop phenology, and machine learning to estimate crop water stress across canal command areas on an 8-day cycle.

Engineered for real-world impact and zero infrastructure cost, AgroScope integrates **FastAPI on Railway**, **React on Cloudflare Pages**, **PostGIS on Supabase**, **DagsHub (DVC & MLflow)**, **Evidently AI**, **UptimeRobot**, and **GitHub Actions** into an automated, self-monitoring, and self-retraining decision-support system for climate-resilient agricultural water management.

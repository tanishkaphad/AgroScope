-- AgroScope Supabase PostGIS Database Schema
-- Migration: 001_initial_schema.sql
-- Enables spatial data storage, 8-day satellite water stress tracking,
-- and irrigation advisory logs for canal command areas.

-- 1. Enable PostGIS Extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 2. Study Areas / Canal Command Area Boundary
CREATE TABLE IF NOT EXISTS study_areas (
    id SERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    project VARCHAR(150),
    state VARCHAR(100) DEFAULT 'Maharashtra',
    district VARCHAR(100) DEFAULT 'Pune',
    culturable_command_area_ha NUMERIC(12, 2),
    gross_command_area_ha NUMERIC(12, 2),
    main_canal_length_km NUMERIC(8, 2),
    boundary GEOMETRY(Polygon, 4326),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_study_areas_boundary ON study_areas USING GIST(boundary);

-- 3. Spatial Units (250m x 250m Grids / Farm Parcels within Command Area)
CREATE TABLE IF NOT EXISTS spatial_units (
    id VARCHAR(64) PRIMARY KEY, -- e.g., 'NLBC_CELL_0008'
    study_area_id INT REFERENCES study_areas(id) ON DELETE CASCADE,
    centroid GEOMETRY(Point, 4326),
    geom GEOMETRY(Polygon, 4326) NOT NULL,
    area_ha NUMERIC(6, 2) DEFAULT 6.25,
    dominant_crop VARCHAR(80) DEFAULT 'Sugarcane / Wheat',
    soil_type VARCHAR(100) DEFAULT 'Vertisols (Black Cotton Soil)',
    taluka VARCHAR(80),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_spatial_units_geom ON spatial_units USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_spatial_units_study_area ON spatial_units(study_area_id);

-- 4. 8-Day Cycle Water Stress Predictions
CREATE TABLE IF NOT EXISTS stress_predictions (
    id BIGSERIAL PRIMARY KEY,
    unit_id VARCHAR(64) REFERENCES spatial_units(id) ON DELETE CASCADE,
    cycle_date DATE NOT NULL,
    stress_class VARCHAR(20) NOT NULL CHECK (stress_class IN ('LOW', 'MODERATE', 'HIGH')),
    stress_code SMALLINT NOT NULL CHECK (stress_code IN (0, 1, 2)),
    confidence NUMERIC(5, 4) NOT NULL,
    probability_low NUMERIC(5, 4),
    probability_moderate NUMERIC(5, 4),
    probability_high NUMERIC(5, 4),
    ndvi_mean NUMERIC(5, 4),
    ndwi_mean NUMERIC(5, 4),
    evi_mean NUMERIC(5, 4),
    temp_max_c NUMERIC(5, 2),
    rainfall_8day NUMERIC(6, 2),
    rainfall_16day NUMERIC(6, 2),
    et0_8day_sum NUMERIC(6, 2),
    model_version VARCHAR(64) DEFAULT 'xgboost-v1.0',
    shap_top_drivers JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stress_predictions_cycle_unit ON stress_predictions(cycle_date, unit_id);
CREATE INDEX IF NOT EXISTS idx_stress_predictions_stress_class ON stress_predictions(stress_class);

-- 5. Actionable Irrigation Advisories
CREATE TABLE IF NOT EXISTS irrigation_advisories (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES stress_predictions(id) ON DELETE CASCADE,
    unit_id VARCHAR(64) REFERENCES spatial_units(id) ON DELETE CASCADE,
    cycle_date DATE NOT NULL,
    priority SMALLINT NOT NULL CHECK (priority IN (1, 2, 3)),
    urgency_en VARCHAR(100) NOT NULL,
    urgency_mr VARCHAR(100),
    urgency_hi VARCHAR(100),
    title_en VARCHAR(200) NOT NULL,
    title_mr VARCHAR(200),
    title_hi VARCHAR(200),
    recommendation_en TEXT NOT NULL,
    recommendation_mr TEXT,
    recommendation_hi TEXT,
    action_steps JSONB DEFAULT '[]'::jsonb,
    advisory_date DATE NOT NULL,
    color_code VARCHAR(16) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_advisories_priority_cycle ON irrigation_advisories(priority, cycle_date);

-- 6. MLOps Model Performance & Drift Logs
CREATE TABLE IF NOT EXISTS ml_drift_logs (
    id SERIAL PRIMARY KEY,
    eval_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dataset_version VARCHAR(64) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    data_drift_detected BOOLEAN DEFAULT FALSE,
    prediction_drift_detected BOOLEAN DEFAULT FALSE,
    dataset_drift_share NUMERIC(5, 4),
    ks_test_pvalue NUMERIC(6, 5),
    wasserstein_distance NUMERIC(6, 4),
    monitored_features JSONB DEFAULT '{}'::jsonb,
    report_html_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. Ground Truth & Farmer Field Feedback (for Continual Learning)
CREATE TABLE IF NOT EXISTS ground_truth_feedback (
    id BIGSERIAL PRIMARY KEY,
    unit_id VARCHAR(64) REFERENCES spatial_units(id) ON DELETE CASCADE,
    observation_date DATE NOT NULL,
    reporter_role VARCHAR(50) DEFAULT 'Farmer', -- 'Farmer', 'Canal_Officer', 'Agronomist'
    actual_stress VARCHAR(20) CHECK (actual_stress IN ('LOW', 'MODERATE', 'HIGH')),
    soil_cracking_observed BOOLEAN,
    last_irrigation_days_ago INT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8. Spatial GeoJSON View for Dashboard Queries
CREATE OR REPLACE VIEW view_latest_stress_geojson AS
SELECT 
    su.id AS cell_id,
    sp.cycle_date,
    sp.stress_class,
    sp.stress_code,
    sp.confidence,
    ia.priority,
    ia.urgency_en,
    ia.title_en,
    ia.recommendation_en,
    ia.color_code,
    sp.ndvi_mean,
    sp.ndwi_mean,
    ST_AsGeoJSON(su.geom)::jsonb AS geometry
FROM spatial_units su
JOIN stress_predictions sp ON su.id = sp.unit_id
LEFT JOIN irrigation_advisories ia ON sp.id = ia.prediction_id
WHERE sp.cycle_date = (SELECT MAX(cycle_date) FROM stress_predictions);

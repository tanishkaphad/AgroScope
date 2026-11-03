import React, { useState } from 'react';
import { Play, RotateCcw, Sliders, Sparkles, Thermometer, Umbrella, Wind, Zap } from 'lucide-react';
import { runPredict } from '../services/api';

export default function PredictionStudio({ currentLang = 'en', initialValues = null }) {
  const [ndvi, setNdvi] = useState(initialValues?.ndvi ?? 0.28);
  const [ndwi, setNdwi] = useState(initialValues?.ndwi ?? 0.04);
  const [tempMax, setTempMax] = useState(34.0);
  const [rainfall, setRainfall] = useState(0.0);
  const [et0, setEt0] = useState(36.0);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleRunInference = async () => {
    setLoading(true);
    const payload = {
      ndvi_mean: parseFloat(ndvi),
      ndwi_mean: parseFloat(ndwi),
      temp_max_c: parseFloat(tempMax),
      rainfall_8day: parseFloat(rainfall),
      et0_8day_sum: parseFloat(et0),
      lang: currentLang,
    };

    const res = await runPredict(payload);
    setResult(res);
    setLoading(false);
  };

  const handleReset = () => {
    setNdvi(0.28);
    setNdwi(0.04);
    setTempMax(34.0);
    setRainfall(0.0);
    setEt0(36.0);
    setResult(null);
  };

  return (
    <div className="sidebar-card" id="prediction-studio-card">
      <div className="card-title-group">
        <h2>
          <Sliders size={17} color="var(--brand-cyan)" />
          <span>What-If Stress Simulator</span>
        </h2>
        <button
          type="button"
          className="icon-btn"
          onClick={handleReset}
          title="Reset parameters to command area averages"
          id="sim-reset-btn"
        >
          <RotateCcw size={13} />
        </button>
      </div>

      <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: 1.4 }}>
        Adjust satellite & meteorological variables to simulate crop water stress response under drought or heatwaves.
      </p>

      {/* Sliders */}
      <div className="sim-slider-group">
        <div className="slider-label-row">
          <span>NDWI (Canopy Water Content)</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--brand-cyan)' }}>{Number(ndwi).toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="-0.25"
          max="0.25"
          step="0.01"
          value={ndwi}
          onChange={(e) => setNdwi(e.target.value)}
          className="slider-input"
          id="sim-slider-ndwi"
        />
      </div>

      <div className="sim-slider-group">
        <div className="slider-label-row">
          <span>NDVI (Vegetation Greenness)</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: '#34d399' }}>{Number(ndvi).toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.05"
          max="0.75"
          step="0.01"
          value={ndvi}
          onChange={(e) => setNdvi(e.target.value)}
          className="slider-input"
          id="sim-slider-ndvi"
        />
      </div>

      <div className="sim-slider-group">
        <div className="slider-label-row">
          <span>Peak Temperature (°C)</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: '#f87171' }}>{tempMax}°C</span>
        </div>
        <input
          type="range"
          min="22"
          max="46"
          step="0.5"
          value={tempMax}
          onChange={(e) => setTempMax(e.target.value)}
          className="slider-input"
          id="sim-slider-temp"
        />
      </div>

      <div className="sim-slider-group">
        <div className="slider-label-row">
          <span>8-Day Rainfall (mm)</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{rainfall} mm</span>
        </div>
        <input
          type="range"
          min="0"
          max="80"
          step="1"
          value={rainfall}
          onChange={(e) => setRainfall(e.target.value)}
          className="slider-input"
          id="sim-slider-rain"
        />
      </div>

      <button
        type="button"
        className="primary-action-btn"
        onClick={handleRunInference}
        disabled={loading}
        id="run-simulation-btn"
      >
        <Zap size={16} />
        <span>{loading ? 'Evaluating Model...' : 'Run AI Stress Inference'}</span>
      </button>

      {/* Inference Result Box */}
      {result && (
        <div style={{ marginTop: '1.25rem', animation: 'fadeIn 0.25s ease-out' }}>
          <div className="parcel-meta-header" style={{ borderColor: result.advisory?.color || 'var(--brand-cyan)' }}>
            <div className="parcel-id-row">
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#fff' }}>SIMULATED PREDICTION</span>
              <span className={`stress-badge-pill ${result.predicted_class?.toLowerCase()}`}>
                {result.predicted_class} ({Math.round((result.confidence || 0.98) * 100)}%)
              </span>
            </div>
            <div style={{ fontSize: '0.78rem', marginTop: '0.35rem', color: result.advisory?.color, fontWeight: 700 }}>
              {result.advisory?.title}
            </div>
            <div style={{ fontSize: '0.75rem', marginTop: '0.25rem', color: 'var(--text-secondary)' }}>
              {result.advisory?.recommendation}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

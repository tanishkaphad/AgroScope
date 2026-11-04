import React, { useState } from 'react';
import { CheckSquare, RotateCcw, Sliders, Zap } from 'lucide-react';
import { errorText } from '../i18n';
import { runPredict } from '../services/api';

const DEFAULTS = { ndvi: 0.28, ndwi: 0.04, tempMax: 34.0, rainfall: 0.0 };
const ET0_8DAY_DEFAULT = 36.0;

const SLIDERS = [
  { key: 'ndwi', label: 'simNdwi', min: -0.25, max: 0.25, step: 0.01, format: (v) => Number(v).toFixed(2) },
  { key: 'ndvi', label: 'simNdvi', min: 0.05, max: 0.75, step: 0.01, format: (v) => Number(v).toFixed(2) },
  { key: 'tempMax', label: 'simTemp', min: 22, max: 46, step: 0.5, format: (v, t) => `${v}${t.unitCelsius}` },
  { key: 'rainfall', label: 'simRain', min: 0, max: 80, step: 1, format: (v, t) => `${v} ${t.unitMm}` },
];

export default function PredictionStudio({ currentLang = 'en', t, initialValues = null }) {
  const [values, setValues] = useState({
    ...DEFAULTS,
    ndvi: initialValues?.ndvi ?? DEFAULTS.ndvi,
    ndwi: initialValues?.ndwi ?? DEFAULTS.ndwi,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // The API sends the advice in every language, so a result already shown follows the language button.
  const advice = (field) => result?.advisory?.translations?.[field]?.[currentLang] ?? result?.advisory?.[field];

  const handleRunInference = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await runPredict({
        ndvi_mean: parseFloat(values.ndvi),
        ndwi_mean: parseFloat(values.ndwi),
        temp_max_c: parseFloat(values.tempMax),
        rainfall_8day: parseFloat(values.rainfall),
        et0_8day_sum: ET0_8DAY_DEFAULT,
        lang: currentLang,
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setValues(DEFAULTS);
    setResult(null);
    setError(null);
  };

  return (
    <div className="sidebar-card" id="prediction-studio-card">
      <div className="card-title-group">
        <h2>
          <Sliders size={17} aria-hidden="true" />
          <span>{t.simTitle}</span>
        </h2>
        <button type="button" className="icon-btn" onClick={handleReset} title={t.simReset} aria-label={t.simReset} id="sim-reset-btn">
          <RotateCcw size={13} aria-hidden="true" />
        </button>
      </div>

      <p className="sim-intro">{t.simIntro}</p>

      {SLIDERS.map(({ key, label, min, max, step, format }) => (
        <div className="sim-slider-group" key={key}>
          <div className="slider-label-row">
            <label htmlFor={`sim-slider-${key}`}>{t[label]}</label>
            <span className="slider-value">{format(values[key], t)}</span>
          </div>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={values[key]}
            onChange={(e) => setValues((current) => ({ ...current, [key]: e.target.value }))}
            className="slider-input"
            id={`sim-slider-${key}`}
          />
        </div>
      ))}

      <button type="button" className="primary-action-btn" onClick={handleRunInference} disabled={loading} id="run-simulation-btn">
        <Zap size={16} aria-hidden="true" />
        <span>{loading ? t.simRunning : t.simRun}</span>
      </button>

      {error && (
        <div className="data-error-banner modal-error" role="alert">
          <span>{errorText(error, t)}</span>
        </div>
      )}

      {result && (
        <div className="sim-result fade-in" aria-live="polite">
          <div className="parcel-meta-header">
            <div className="parcel-id-row">
              <span className="sim-result-title">{t.simResult}</span>
              <span className={`stress-badge-pill ${result.predicted_class?.toLowerCase()}`}>
                {t.stressBadge(t.stress[result.predicted_class] ?? result.predicted_class, Math.round(result.confidence * 100))}
              </span>
            </div>
            <div className="advisory-urgency">{advice('urgency')}</div>
            <div className="advisory-title">{advice('title')}</div>
            <div className="advisory-rec">{advice('recommendation')}</div>
            <h3 className="status-subheading">{t.actionSteps}</h3>
            <ul className="action-steps-list">
              {(advice('actionable_steps') || []).map((step) => (
                <li key={step} className="action-step-item">
                  <CheckSquare size={13} className="action-step-icon" aria-hidden="true" />
                  <span>{step}</span>
                </li>
              ))}
            </ul>
            <p className="status-note">{t.simNote}</p>
          </div>
        </div>
      )}
    </div>
  );
}

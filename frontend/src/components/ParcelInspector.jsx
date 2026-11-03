import React, { useEffect, useState } from 'react';
import { AlertCircle, CheckSquare, Droplet, HelpCircle, MapPin, Sliders, Sparkles } from 'lucide-react';
import { fetchParcelDetail } from '../services/api';

export default function ParcelInspector({
  selectedParcel,
  currentLang = 'en',
  onSendToSimulator,
}) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!selectedParcel || !selectedParcel.cell_id) {
      setDetail(null);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchParcelDetail(selectedParcel.cell_id, currentLang)
      .then((res) => {
        if (isMounted) setDetail(res);
      })
      .catch((err) => {
        if (isMounted) {
          setDetail(null);
          setError(err.message);
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [selectedParcel, currentLang]);

  if (!selectedParcel) {
    return (
      <div className="sidebar-card" id="parcel-inspector-card">
        <div className="card-title-group">
          <h2>
            <MapPin size={17} color="var(--brand-cyan)" />
            <span>Parcel Inspector</span>
          </h2>
        </div>
        <div className="inspector-empty">
          <div className="inspector-empty-icon">🛰️</div>
          <p style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>No Parcel Selected</p>
          <p style={{ fontSize: '0.78rem', marginTop: '0.4rem', lineHeight: 1.4 }}>
            Click a 250m grid cell on the map to see its water stress estimate and irrigation advice.
          </p>
        </div>
      </div>
    );
  }

  if (loading || error || !detail) {
    return (
      <div className="sidebar-card" id="parcel-inspector-card">
        <div className="card-title-group">
          <h2>
            <MapPin size={17} color="var(--brand-cyan)" />
            <span>Parcel Inspector</span>
          </h2>
        </div>
        <div className="inspector-empty" role={error ? 'alert' : undefined}>
          <p style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
            {error ? "Couldn't load this parcel" : 'Loading parcel...'}
          </p>
          {error && <p style={{ fontSize: '0.78rem', marginTop: '0.4rem', lineHeight: 1.4 }}>{error}</p>}
        </div>
      </div>
    );
  }

  const stressClass = detail.water_stress.class;
  const confidence = detail.water_stress.confidence;
  const advisory = detail.irrigation_advisory;
  const keyDrivers = detail.key_drivers;
  const { ndvi, ndwi } = detail.canopy_indices;

  const pClass = advisory.priority === 1 ? 'p1' : advisory.priority === 3 ? 'p3' : 'p2';

  return (
    <div className="sidebar-card" id="parcel-inspector-card">
      <div className="card-title-group">
        <h2>
          <MapPin size={17} color="var(--brand-cyan)" />
          <span>Parcel Inspector</span>
        </h2>
        {onSendToSimulator && (
          <button
            type="button"
            className="layer-btn"
            style={{ fontSize: '0.72rem', padding: '0.25rem 0.5rem' }}
            onClick={() => onSendToSimulator(selectedParcel)}
            title="Simulate what-if changes for this cell"
          >
            <Sliders size={12} />
            <span>Simulate</span>
          </button>
        )}
      </div>

      {/* Meta Header */}
      <div className="parcel-meta-header">
        <div className="parcel-id-row">
          <span className="parcel-id">{selectedParcel.cell_id}</span>
          <span className={`stress-badge-pill ${stressClass.toLowerCase()}`}>
            {stressClass} STRESS ({Math.round(confidence * 100)}%)
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
          <span>Crop: {detail.crop_type ?? 'Not available'} | Date: {detail.date}</span>
          <span>NDVI: {ndvi.toFixed(2)} | NDWI: {ndwi.toFixed(2)}</span>
        </div>
      </div>

      {/* Irrigation Advisory Box */}
      <div className={`advisory-box ${pClass}`}>
        <div className="advisory-urgency">{advisory.urgency}</div>
        <div className="advisory-title">{advisory.title}</div>
        <div className="advisory-rec">{advisory.recommendation}</div>

        <div style={{ marginTop: '0.6rem' }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.4rem', textTransform: 'uppercase' }}>
            Action Steps:
          </div>
          <ul className="action-steps-list">
            {(advisory.actionable_steps || []).map((step, idx) => (
              <li key={idx} className="action-step-item">
                <CheckSquare size={13} color="var(--brand-cyan)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{step}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* SHAP Feature Attribution Waterfall */}
      <div className="shap-container">
        <div className="shap-header" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Sparkles size={13} color="var(--brand-cyan)" />
          <span>SHAP Key Stress Drivers</span>
        </div>

        {keyDrivers.length === 0 && (
          <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>Not available for stored results yet.</p>
        )}

        {keyDrivers.map((driver, idx) => (
          <div key={idx} className="shap-driver-row">
            <div className="shap-label-row">
              <span style={{ color: 'var(--text-primary)' }}>
                {driver.arrow} {driver.label}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--brand-cyan)' }}>
                {driver.impact_pct}%
              </span>
            </div>
            <div className="driver-bar-bg">
              <div
                className={`driver-bar-fill ${driver.arrow === '↓' ? 'red' : 'green'}`}
                style={{ width: `${Math.min(100, driver.impact_pct * 2.2)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

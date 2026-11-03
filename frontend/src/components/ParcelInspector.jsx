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

  useEffect(() => {
    if (!selectedParcel || !selectedParcel.cell_id) {
      setDetail(null);
      return;
    }

    let isMounted = true;
    setLoading(true);

    fetchParcelDetail(selectedParcel.cell_id, currentLang).then((res) => {
      if (isMounted) {
        setDetail(res);
        setLoading(false);
      }
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
            Click any 250m agricultural grid cell on the map to inspect real-time AI water stress, SHAP drivers, and irrigation advisories.
          </p>
        </div>
      </div>
    );
  }

  const stressClass = detail?.water_stress?.class || selectedParcel.stress_class || 'MODERATE';
  const confidence = detail?.water_stress?.confidence || selectedParcel.confidence || 0.99;
  const advisory = detail?.irrigation_advisory || {
    priority: selectedParcel.priority || 2,
    urgency: selectedParcel.urgency || 'Schedule Within 3-5 Days',
    title: selectedParcel.title || 'Moderate Water Stress Observed',
    recommendation: selectedParcel.recommendation || 'Crop is experiencing water deficit under moderate temperatures. Plan irrigation in upcoming canal rotation.',
    actionable_steps: selectedParcel.actionable_steps || [
      'Confirm your turn in the next canal water distribution roster',
      'Prepare field channels and furrows for delivery',
    ],
  };

  const keyDrivers = detail?.key_drivers || [
    { feature: 'ndwi_mean', label: 'Canopy Leaf Water Content', value: selectedParcel.ndwi || 0.04, impact_pct: 38.0, direction: 'increases_stress', arrow: '↓' },
    { feature: 'rainfall_8day', label: '8-Day Cumulative Rainfall', value: 0.0, impact_pct: 26.5, direction: 'increases_stress', arrow: '↓' },
    { feature: 'temp_max_c', label: 'Peak Daily Temperature', value: 34.0, impact_pct: 20.1, direction: 'increases_stress', arrow: '↑' },
    { feature: 'ndvi_mean', label: 'Vegetation Canopy Vigor', value: selectedParcel.ndvi || 0.28, impact_pct: 15.4, direction: 'mitigates_stress', arrow: '↑' },
  ];

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
          <span>Crop: Sugarcane / Wheat</span>
          <span>NDVI: {selectedParcel.ndvi?.toFixed(2) || '0.28'} | NDWI: {selectedParcel.ndwi?.toFixed(2) || '0.04'}</span>
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

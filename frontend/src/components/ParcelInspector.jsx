import React, { useEffect, useState } from 'react';
import { CheckSquare, MapPin, Sliders, Sparkles } from 'lucide-react';
import { errorText, formatDate } from '../i18n';
import { fetchParcelDetail } from '../services/api';

function CardTitle({ t, children }) {
  return (
    <div className="card-title-group">
      <h2>
        <MapPin size={17} aria-hidden="true" />
        <span>{t.tabDetails}</span>
      </h2>
      {children}
    </div>
  );
}

// Map cells can't take keyboard focus, so this is the keyboard route to a cell's details.
function CellLookup({ t, onSelectParcel }) {
  const [cellId, setCellId] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    const trimmed = cellId.trim();
    if (trimmed) onSelectParcel({ cell_id: trimmed });
  };

  return (
    <form className="cell-lookup" onSubmit={handleSubmit}>
      <label htmlFor="cell-lookup-input">{t.cellLookup}</label>
      <div className="cell-lookup-row">
        <input
          id="cell-lookup-input"
          type="text"
          placeholder="NLBC_CELL_0008"
          value={cellId}
          onChange={(e) => setCellId(e.target.value)}
          autoComplete="off"
          spellCheck="false"
        />
        <button type="submit" className="layer-btn" id="cell-lookup-btn">{t.cellLookupButton}</button>
      </div>
    </form>
  );
}

function DetailSkeleton() {
  return (
    <div aria-hidden="true">
      <div className="parcel-meta-header">
        <span className="skeleton skeleton-text" style={{ width: '55%', marginBottom: '0.6rem' }} />
        <span className="skeleton skeleton-text" style={{ width: '85%' }} />
      </div>
      <div className="advisory-box">
        <span className="skeleton skeleton-text" style={{ width: '35%', marginBottom: '0.5rem' }} />
        <span className="skeleton skeleton-text" style={{ width: '70%', marginBottom: '0.5rem' }} />
        <span className="skeleton skeleton-text" style={{ width: '95%', marginBottom: '0.4rem' }} />
        <span className="skeleton skeleton-text" style={{ width: '90%' }} />
      </div>
    </div>
  );
}

export default function ParcelInspector({ selectedParcel, currentLang = 'en', date, t, onSelectParcel, onSendToSimulator }) {
  const [loaded, setLoaded] = useState({ key: null, detail: null, error: null });

  const cellId = selectedParcel?.cell_id;
  const requestKey = cellId ? `${cellId}|${currentLang}|${date}` : null;
  const isLoading = requestKey !== null && loaded.key !== requestKey;

  useEffect(() => {
    if (!requestKey) return undefined;

    let isMounted = true;
    fetchParcelDetail(cellId, currentLang, date)
      .then((detail) => isMounted && setLoaded({ key: requestKey, detail, error: null }))
      .catch((err) => isMounted && setLoaded({ key: requestKey, detail: null, error: err.message }));

    return () => {
      isMounted = false;
    };
  }, [requestKey, cellId, currentLang, date]);

  if (!selectedParcel) {
    return (
      <div className="sidebar-card" id="parcel-inspector-card">
        <CardTitle t={t} />
        <div className="inspector-empty">
          <div className="inspector-empty-icon" aria-hidden="true">🛰️</div>
          <p className="inspector-empty-title">{t.noCell}</p>
          <p className="inspector-empty-help">{t.noCellHelp}</p>
        </div>
        <CellLookup t={t} onSelectParcel={onSelectParcel} />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="sidebar-card" id="parcel-inspector-card" aria-busy="true">
        <CardTitle t={t} />
        <DetailSkeleton />
      </div>
    );
  }

  if (loaded.error || !loaded.detail) {
    return (
      <div className="sidebar-card" id="parcel-inspector-card">
        <CardTitle t={t} />
        <div className="inspector-empty" role="alert">
          <p className="inspector-empty-title">{t.loadCellError}</p>
          <p className="inspector-empty-help">{errorText(loaded.error, t)}</p>
          <button type="button" className="layer-btn banner-retry" onClick={() => setLoaded({ key: null, detail: null, error: null })} id="cell-retry-btn">
            {t.retry}
          </button>
        </div>
        <CellLookup t={t} onSelectParcel={onSelectParcel} />
      </div>
    );
  }

  const { detail } = loaded;
  const stressClass = detail.water_stress.class;
  const advisory = detail.irrigation_advisory;
  const keyDrivers = detail.key_drivers;
  const { ndvi, ndwi } = detail.canopy_indices;
  const pClass = advisory.priority === 1 ? 'p1' : advisory.priority === 3 ? 'p3' : 'p2';

  return (
    <div className="sidebar-card fade-in" id="parcel-inspector-card">
      <CardTitle t={t}>
        {onSendToSimulator && (
          <button type="button" className="layer-btn" onClick={() => onSendToSimulator(selectedParcel)} title={t.simulateHelp}>
            <Sliders size={12} aria-hidden="true" />
            <span>{t.simulate}</span>
          </button>
        )}
      </CardTitle>

      <div className="parcel-meta-header">
        <div className="parcel-id-row">
          <span className="parcel-id">{cellId}</span>
          <span className={`stress-badge-pill ${stressClass.toLowerCase()}`}>
            {t.stressBadge(t.stress[stressClass] ?? stressClass, Math.round(detail.water_stress.confidence * 100))}
          </span>
        </div>
        <div className="parcel-meta-row">
          <span>{t.crop}: {detail.crop_type ?? t.notAvailable}</span>
          <span>{t.date}: {formatDate(detail.date, currentLang)}</span>
          <span>NDVI {ndvi.toFixed(2)}, NDWI {ndwi.toFixed(2)}</span>
        </div>
      </div>

      <div className={`advisory-box ${pClass}`}>
        <div className="advisory-urgency">{advisory.urgency}</div>
        <div className="advisory-title">{advisory.title}</div>
        <div className="advisory-rec">{advisory.recommendation}</div>

        <h3 className="status-subheading">{t.actionSteps}</h3>
        <ul className="action-steps-list">
          {(advisory.actionable_steps || []).map((step) => (
            <li key={step} className="action-step-item">
              <CheckSquare size={13} className="action-step-icon" aria-hidden="true" />
              <span>{step}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="shap-container">
        <div className="shap-header">
          <Sparkles size={13} aria-hidden="true" />
          <span>{t.shapTitle}</span>
        </div>

        {keyDrivers.length === 0 && <p className="status-note">{t.shapNone}</p>}

        {keyDrivers.map((driver) => (
          <div key={driver.feature} className="shap-driver-row">
            <div className="shap-label-row">
              <span>{driver.arrow} {driver.label}</span>
              <span className="shap-value">{driver.impact_pct}%</span>
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

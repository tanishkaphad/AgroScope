import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Calendar, Clock, MapIcon } from 'lucide-react';
import Header from './components/Header';
import KpiBar from './components/KpiBar';
import MapContainer from './components/MapContainer';
import ParcelInspector from './components/ParcelInspector';
import PredictionStudio from './components/PredictionStudio';
import MlopsModal from './components/MlopsModal';
import { errorText, formatDate, formatNumber, getText } from './i18n';
import {
  fetchAvailableDates,
  fetchHealth,
  fetchStatistics,
  fetchStudyArea,
  fetchStressMap,
  fetchIrrigationMap,
} from './services/api';

const DATE_DATA = { stats: null, stressMap: null, irrigationMap: null };
const MAP_CELL_LIMIT = 1500;

export default function App() {
  const [currentLang, setCurrentLang] = useState('en');
  const [activeLayer, setActiveLayer] = useState('stress'); // 'stress' | 'priority'
  const [activeTab, setActiveTab] = useState('inspector'); // 'inspector' | 'simulator'

  const [data, setData] = useState(DATE_DATA);
  const [studyArea, setStudyArea] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);
  const [publishedDate, setPublishedDate] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null); // null means the published date
  const [pending, setPending] = useState(Object.keys(DATE_DATA));
  const [loadErrors, setLoadErrors] = useState([]);
  const [systemStatus, setSystemStatus] = useState('checking'); // 'checking' | 'online' | 'degraded' | 'offline'
  const [loadAttempt, setLoadAttempt] = useState(0);

  const [selectedParcel, setSelectedParcel] = useState(null);
  const [isMlopsOpen, setIsMlopsOpen] = useState(false);

  const t = useMemo(() => getText(currentLang), [currentLang]);

  useEffect(() => {
    document.documentElement.lang = currentLang;
  }, [currentLang]);

  const reportError = useCallback((err) => {
    setLoadErrors((errors) => (errors.includes(err.message) ? errors : [...errors, err.message]));
  }, []);

  useEffect(() => {
    fetchHealth()
      .then((h) => setSystemStatus(h.status === 'HEALTHY' ? 'online' : 'degraded'))
      .catch(() => setSystemStatus('offline'));
    fetchStudyArea().then(setStudyArea).catch(reportError);
    fetchAvailableDates().then(setAvailableDates).catch(reportError);
  }, [loadAttempt, reportError]);

  // Results for the chosen date. A slower reply for an earlier choice must not replace a newer one.
  useEffect(() => {
    let isCurrent = true;
    const load = (key, promise) => {
      promise
        .then((value) => {
          if (!isCurrent) return;
          setData((current) => ({ ...current, [key]: value }));
          if (key === 'stats' && !selectedDate) setPublishedDate(value.cycle_date);
        })
        .catch((err) => isCurrent && reportError(err))
        .finally(() => isCurrent && setPending((current) => current.filter((item) => item !== key)));
    };

    load('stats', fetchStatistics(selectedDate));
    load('stressMap', fetchStressMap(MAP_CELL_LIMIT, selectedDate));
    load('irrigationMap', fetchIrrigationMap(MAP_CELL_LIMIT, selectedDate));
    return () => {
      isCurrent = false;
    };
  }, [selectedDate, loadAttempt, reportError]);

  const resetDateData = () => {
    setLoadErrors([]);
    setData(DATE_DATA);
    setPending(Object.keys(DATE_DATA));
  };

  const handleDateChange = (date) => {
    resetDateData();
    setSelectedDate(date === publishedDate ? null : date);
  };

  const handleRetry = () => {
    resetDateData();
    setSystemStatus('checking');
    setLoadAttempt((attempt) => attempt + 1);
  };

  const handleSelectParcel = useCallback((parcelData) => {
    setSelectedParcel(parcelData);
    setActiveTab('inspector');
  }, []);

  const handleSendToSimulator = (parcelData) => {
    setSelectedParcel(parcelData);
    setActiveTab('simulator');
  };

  const { stats, stressMap, irrigationMap } = data;
  const shownDate = selectedDate ?? publishedDate;
  const mapMeta = stressMap?.metadata;
  const shownCells = mapMeta?.sampled_cells ?? stressMap?.features?.length;
  const totalCells = mapMeta?.total_cells ?? stressMap?.features?.length;
  const isLoading = pending.length > 0;

  return (
    <div className="app-container" id="agroscope-app">
      <Header
        t={t}
        currentLang={currentLang}
        onLangChange={setCurrentLang}
        onOpenMlops={() => setIsMlopsOpen(true)}
        systemStatus={systemStatus}
      />

      {loadErrors.length > 0 && (
        <div className="data-error-banner" role="alert" id="data-error-banner">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>{t.loadError} {loadErrors.map((message) => errorText(message, t)).join(' ')}</span>
          <button type="button" className="layer-btn banner-retry" onClick={handleRetry} id="retry-load-btn">
            {t.retry}
          </button>
        </div>
      )}

      <section className="data-context-bar" id="data-context-bar" aria-live="polite">
        <div className="context-item context-date">
          <Calendar size={15} aria-hidden="true" />
          <label className="context-label" htmlFor="data-date-select">{t.dataDate}:</label>
          {availableDates.length > 0 ? (
            <select
              id="data-date-select"
              className="date-select"
              value={shownDate ?? ''}
              onChange={(e) => handleDateChange(e.target.value)}
            >
              {availableDates.map((date) => (
                <option key={date} value={date}>
                  {formatDate(date, currentLang)}{date === publishedDate ? ` (${t.publishedDate})` : ''}
                </option>
              ))}
            </select>
          ) : isLoading ? (
            <span className="skeleton skeleton-text" style={{ width: '9rem' }} />
          ) : (
            <strong>{t.noData}</strong>
          )}
        </div>
        <div className="context-pill" title={t.pastDataHelp} id="past-data-pill">
          <Clock size={13} aria-hidden="true" />
          <span>{t.pastData}</span>
        </div>
        <span className="visually-hidden">{t.pastDataHelp}</span>
        <div className="context-item context-cells" id="map-cell-count">
          <MapIcon size={15} aria-hidden="true" />
          {stressMap ? (
            <span>{t.showingCells(formatNumber(shownCells), formatNumber(totalCells))}</span>
          ) : isLoading ? (
            <span className="skeleton skeleton-text" style={{ width: '14rem' }} />
          ) : (
            <span>{t.noData}</span>
          )}
        </div>
      </section>

      <KpiBar stats={stats} isLoading={pending.includes('stats')} t={t} />

      <main className="dashboard-main">
        <MapContainer
          studyAreaGeoJson={studyArea}
          stressMapGeoJson={stressMap}
          irrigationMapGeoJson={irrigationMap}
          stats={stats}
          isLoading={pending.includes('stressMap') || pending.includes('irrigationMap')}
          activeLayer={activeLayer}
          onLayerChange={setActiveLayer}
          onSelectParcel={handleSelectParcel}
          selectedParcelId={selectedParcel?.cell_id}
          t={t}
        />

        <aside className="sidebar-panel">
          <div className="sidebar-tabs">
            <button
              type="button"
              className={`layer-btn ${activeTab === 'inspector' ? 'active' : ''}`}
              aria-pressed={activeTab === 'inspector'}
              onClick={() => setActiveTab('inspector')}
              id="tab-inspector-btn"
            >
              {t.tabDetails}
            </button>
            <button
              type="button"
              className={`layer-btn ${activeTab === 'simulator' ? 'active' : ''}`}
              aria-pressed={activeTab === 'simulator'}
              onClick={() => setActiveTab('simulator')}
              id="tab-simulator-btn"
            >
              {t.tabSimulator}
            </button>
          </div>

          {activeTab === 'inspector' ? (
            <ParcelInspector
              selectedParcel={selectedParcel}
              currentLang={currentLang}
              date={selectedDate}
              t={t}
              onSelectParcel={handleSelectParcel}
              onSendToSimulator={handleSendToSimulator}
            />
          ) : (
            <PredictionStudio currentLang={currentLang} t={t} initialValues={selectedParcel} />
          )}
        </aside>
      </main>

      {isMlopsOpen && <MlopsModal t={t} currentLang={currentLang} onClose={() => setIsMlopsOpen(false)} />}
    </div>
  );
}

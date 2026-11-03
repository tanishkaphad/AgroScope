import React, { useEffect, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import Header from './components/Header';
import KpiBar from './components/KpiBar';
import MapContainer from './components/MapContainer';
import ParcelInspector from './components/ParcelInspector';
import PredictionStudio from './components/PredictionStudio';
import MlopsModal from './components/MlopsModal';
import {
  fetchAvailableDates,
  fetchHealth,
  fetchStatistics,
  fetchStudyArea,
  fetchStressMap,
  fetchIrrigationMap,
} from './services/api';

export default function App() {
  const [cycleDate, setCycleDate] = useState('');
  const [availableDates, setAvailableDates] = useState([]);

  const [currentLang, setCurrentLang] = useState('en');
  const [activeLayer, setActiveLayer] = useState('stress'); // 'stress' | 'priority'
  const [activeTab, setActiveTab] = useState('inspector'); // 'inspector' | 'simulator'

  const [stats, setStats] = useState(null);
  const [studyArea, setStudyArea] = useState(null);
  const [stressMap, setStressMap] = useState(null);
  const [irrigationMap, setIrrigationMap] = useState(null);
  const [loadErrors, setLoadErrors] = useState([]);

  const [selectedParcel, setSelectedParcel] = useState(null);
  const [isMlopsOpen, setIsMlopsOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState('CONNECTING');

  // Load Initial Datasets
  useEffect(() => {
    const reportError = (err) => {
      setLoadErrors((errors) => (errors.includes(err.message) ? errors : [...errors, err.message]));
    };

    fetchHealth()
      .then((h) => setSystemStatus(h.status === 'HEALTHY' ? 'API ONLINE' : 'API DEGRADED'))
      .catch(() => setSystemStatus('API OFFLINE'));

    fetchAvailableDates()
      .then((dates) => {
        setAvailableDates(dates);
        setCycleDate(dates[dates.length - 1] ?? '');
      })
      .catch(reportError);
    fetchStatistics().then(setStats).catch(reportError);
    fetchStudyArea().then(setStudyArea).catch(reportError);
    fetchStressMap(1500).then(setStressMap).catch(reportError);
    fetchIrrigationMap(1500).then(setIrrigationMap).catch(reportError);
  }, []);

  const handleSelectParcel = (parcelData) => {
    setSelectedParcel(parcelData);
    setActiveTab('inspector');
  };

  const handleSendToSimulator = (parcelData) => {
    setSelectedParcel(parcelData);
    setActiveTab('simulator');
  };

  return (
    <div className="app-container" id="agroscope-app">
      {/* Top Header */}
      <Header
        cycleDate={cycleDate}
        onDateChange={setCycleDate}
        availableDates={availableDates}
        currentLang={currentLang}
        onLangChange={setCurrentLang}
        onOpenMlops={() => setIsMlopsOpen(true)}
        systemStatus={systemStatus}
      />

      {loadErrors.length > 0 && (
        <div className="data-error-banner" role="alert" id="data-error-banner">
          <AlertTriangle size={16} />
          <span>Some data couldn't be loaded: {loadErrors.join(' ')}</span>
        </div>
      )}

      {/* KPI Overview Metrics */}
      <KpiBar stats={stats} currentLang={currentLang} />

      {/* Main Interactive Grid */}
      <main className="dashboard-main">
        {/* Interactive GIS Map */}
        <MapContainer
          studyAreaGeoJson={studyArea}
          stressMapGeoJson={stressMap}
          irrigationMapGeoJson={irrigationMap}
          stats={stats}
          activeLayer={activeLayer}
          onLayerChange={setActiveLayer}
          onSelectParcel={handleSelectParcel}
          selectedParcelId={selectedParcel?.cell_id}
          currentLang={currentLang}
        />

        {/* Right Sidebar: Parcel Inspector or What-If Simulator */}
        <aside className="sidebar-panel">
          {/* Tab Switcher */}
          <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '-0.35rem' }}>
            <button
              type="button"
              className={`layer-btn ${activeTab === 'inspector' ? 'active' : ''}`}
              style={{ flex: 1, justifyContent: 'center' }}
              onClick={() => setActiveTab('inspector')}
              id="tab-inspector-btn"
            >
              Parcel Inspector
            </button>
            <button
              type="button"
              className={`layer-btn ${activeTab === 'simulator' ? 'active' : ''}`}
              style={{ flex: 1, justifyContent: 'center' }}
              onClick={() => setActiveTab('simulator')}
              id="tab-simulator-btn"
            >
              What-If Simulator
            </button>
          </div>

          {activeTab === 'inspector' ? (
            <ParcelInspector
              selectedParcel={selectedParcel}
              currentLang={currentLang}
              onSendToSimulator={handleSendToSimulator}
            />
          ) : (
            <PredictionStudio
              currentLang={currentLang}
              initialValues={selectedParcel}
            />
          )}
        </aside>
      </main>

      {/* MLOps Drift & Retrain Modal */}
      <MlopsModal
        isOpen={isMlopsOpen}
        onClose={() => setIsMlopsOpen(false)}
      />
    </div>
  );
}

import React, { useEffect, useState } from 'react';
import Header from './components/Header';
import KpiBar from './components/KpiBar';
import MapContainer from './components/MapContainer';
import ParcelInspector from './components/ParcelInspector';
import PredictionStudio from './components/PredictionStudio';
import MlopsModal from './components/MlopsModal';
import {
  fetchHealth,
  fetchStatistics,
  fetchStudyArea,
  fetchStressMap,
  fetchIrrigationMap,
  fetchDriftReport,
} from './services/api';

export default function App() {
  const [cycleDate, setCycleDate] = useState('2023-12-10');
  const availableDates = [
    '2023-12-10',
    '2024-03-09',
    '2024-03-14',
    '2024-03-19',
    '2024-03-24',
    '2024-03-29',
  ];

  const [currentLang, setCurrentLang] = useState('en');
  const [activeLayer, setActiveLayer] = useState('stress'); // 'stress' | 'priority'
  const [activeTab, setActiveTab] = useState('inspector'); // 'inspector' | 'simulator'

  const [stats, setStats] = useState(null);
  const [studyArea, setStudyArea] = useState(null);
  const [stressMap, setStressMap] = useState(null);
  const [irrigationMap, setIrrigationMap] = useState(null);
  const [driftReport, setDriftReport] = useState(null);

  const [selectedParcel, setSelectedParcel] = useState(null);
  const [isMlopsOpen, setIsMlopsOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState('CONNECTED');

  // Load Initial Datasets
  useEffect(() => {
    fetchHealth().then((h) => {
      if (h.status === 'HEALTHY') setSystemStatus('API ONLINE');
      else setSystemStatus('LOCAL READY');
    });

    fetchStatistics().then(setStats);
    fetchStudyArea().then(setStudyArea);
    fetchStressMap(1500).then(setStressMap);
    fetchIrrigationMap(1500).then(setIrrigationMap);
    fetchDriftReport().then(setDriftReport);
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

      {/* KPI Overview Metrics */}
      <KpiBar stats={stats} currentLang={currentLang} />

      {/* Main Interactive Grid */}
      <main className="dashboard-main">
        {/* Interactive GIS Map */}
        <MapContainer
          studyAreaGeoJson={studyArea}
          stressMapGeoJson={stressMap}
          irrigationMapGeoJson={irrigationMap}
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
        driftData={driftReport}
      />
    </div>
  );
}

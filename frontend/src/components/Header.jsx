import React from 'react';
import { Activity, Calendar, Globe, Satellite, ShieldCheck } from 'lucide-react';

export default function Header({
  cycleDate,
  onDateChange,
  availableDates = [],
  currentLang,
  onLangChange,
  onOpenMlops,
  systemStatus = 'HEALTHY',
}) {
  const titles = {
    en: {
      title: 'AgroScope',
      tagline: 'Satellite Crop Water Stress & Irrigation Advisory',
      area: 'Nira Left Bank Canal Command Area (Pune)',
      mlopsBtn: 'Model Status',
    },
    mr: {
      title: 'ॲग्रोस्कोप',
      tagline: 'उपग्रह पीक जल ताण व सिंचन सल्ला प्रणाली',
      area: 'निरा डावा कालवा लाभक्षेत्र (पुणे)',
      mlopsBtn: 'एमएल मॉनिटरिंग',
    },
    hi: {
      title: 'एग्रोस्कोप',
      tagline: 'उपग्रह फसल जल तनाव एवं सिंचाई सलाह प्रणाली',
      area: 'नीरा बायां नहर कमान क्षेत्र (पुणे)',
      mlopsBtn: 'एमएल मॉनिटरिंग',
    },
  };

  const t = titles[currentLang] || titles.en;

  return (
    <header className="header-bar" id="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon-box" id="brand-logo">
          🛰️
        </div>
        <div className="brand-title-group">
          <h1>{t.title}</h1>
          <div className="brand-subtitle">
            {t.tagline} • <span style={{ color: 'var(--brand-cyan)' }}>{t.area}</span>
          </div>
        </div>
      </div>

      <div className="header-actions">
        {/* Cycle Date Selector */}
        <div className="selector-pill" id="cycle-date-selector">
          <Calendar size={15} color="var(--brand-cyan)" />
          <select
            value={cycleDate}
            onChange={(e) => onDateChange(e.target.value)}
            disabled={availableDates.length <= 1}
            id="cycle-select-input"
            aria-label="Select satellite observation cycle"
          >
            {availableDates.length === 0 && <option value="">No data</option>}
            {availableDates.map((d) => (
              <option key={d} value={d}>
                Cycle: {d}
              </option>
            ))}
          </select>
        </div>

        {/* Trilingual Switcher */}
        <div className="lang-switch" id="lang-switch-group">
          <button
            type="button"
            className={`lang-btn ${currentLang === 'en' ? 'active' : ''}`}
            onClick={() => onLangChange('en')}
            id="lang-btn-en"
          >
            EN
          </button>
          <button
            type="button"
            className={`lang-btn ${currentLang === 'mr' ? 'active' : ''}`}
            onClick={() => onLangChange('mr')}
            id="lang-btn-mr"
          >
            मराठी
          </button>
          <button
            type="button"
            className={`lang-btn ${currentLang === 'hi' ? 'active' : ''}`}
            onClick={() => onLangChange('hi')}
            id="lang-btn-hi"
          >
            हिन्दी
          </button>
        </div>

        {/* MLOps Trigger Button */}
        <button
          type="button"
          className="layer-btn"
          onClick={onOpenMlops}
          id="mlops-modal-open-btn"
          style={{ borderColor: 'rgba(56, 189, 248, 0.3)' }}
        >
          <Activity size={14} color="#38bdf8" />
          <span>{t.mlopsBtn}</span>
        </button>

        {/* Live System Status Pill */}
        <div className={`status-badge ${systemStatus === 'API ONLINE' ? '' : 'problem'}`} id="system-status-pill" title="API connection status">
          <div className="pulse-dot"></div>
          <span>{systemStatus}</span>
        </div>
      </div>
    </header>
  );
}

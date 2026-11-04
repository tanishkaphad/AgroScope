import React from 'react';
import { Activity, Sprout } from 'lucide-react';

const LANGUAGES = [
  { code: 'en', label: 'EN' },
  { code: 'mr', label: 'मराठी' },
  { code: 'hi', label: 'हिन्दी' },
];

const STATUS_TEXT = { checking: 'apiChecking', online: 'apiOnline', degraded: 'apiDegraded', offline: 'apiOffline' };

export default function Header({ t, currentLang, onLangChange, onOpenMlops, systemStatus }) {
  return (
    <header className="header-bar" id="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon-box" id="brand-logo" aria-hidden="true">
          <Sprout size={21} strokeWidth={2.2} />
        </div>
        <div className="brand-title-group">
          <h1>{t.appTitle}</h1>
          <div className="brand-subtitle">
            {t.tagline}
          </div>
        </div>
      </div>

      <div className="header-actions">
        <div className="lang-switch" id="lang-switch-group" role="group" aria-label={t.languageGroup}>
          {LANGUAGES.map(({ code, label }) => (
            <button
              key={code}
              type="button"
              className={`lang-btn ${currentLang === code ? 'active' : ''}`}
              aria-pressed={currentLang === code}
              lang={code}
              onClick={() => onLangChange(code)}
              id={`lang-btn-${code}`}
            >
              {label}
            </button>
          ))}
        </div>

        <button type="button" className="layer-btn" onClick={onOpenMlops} id="mlops-modal-open-btn">
          <Activity size={14} aria-hidden="true" />
          <span>{t.modelStatus}</span>
        </button>

        <div
          className={`status-badge ${systemStatus === 'online' ? '' : systemStatus === 'checking' ? 'checking' : 'problem'}`}
          id="system-status-pill"
          title={t.apiStatusHelp}
          role="status"
        >
          <div className="pulse-dot" aria-hidden="true"></div>
          <span>{t[STATUS_TEXT[systemStatus]]}</span>
        </div>
      </div>

    </header>
  );
}

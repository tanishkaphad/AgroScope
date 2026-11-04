import React from 'react';
import { Activity, ExternalLink } from 'lucide-react';
import { DASHBOARD_LINKS } from '../services/api';

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
          🛰️
        </div>
        <div className="brand-title-group">
          <h1>{t.appTitle}</h1>
          <div className="brand-subtitle">
            {t.tagline} <span className="brand-area">{t.area}</span>
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

      <nav className="tools-bar" id="tools-bar" aria-label={t.tools}>
        <span className="tools-label">{t.tools}:</span>
        {DASHBOARD_LINKS.map(({ key, label, url }) => (
          <a key={key} id={`tool-link-${key}`} className="tool-link" href={url} target="_blank" rel="noreferrer" title={t.toolHelp[key]}>
            <span>{label}</span>
            <ExternalLink size={11} aria-hidden="true" />
            <span className="visually-hidden">{t.toolHelp[key]}</span>
          </a>
        ))}
      </nav>
    </header>
  );
}

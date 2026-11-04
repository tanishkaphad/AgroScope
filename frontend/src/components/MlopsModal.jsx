import React, { useEffect, useRef, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle, ExternalLink, X } from 'lucide-react';
import { errorText, formatDate } from '../i18n';
import { driftReportUrl, fetchDriftReport, fetchMetrics, fetchPromotionDecision, TOOL_LINKS } from '../services/api';

const TOP_DRIFT_INPUTS = 5;

function Row({ label, children }) {
  return (
    <div className="status-row">
      <span>{label}</span>
      <span className="status-value">{children}</span>
    </div>
  );
}

function PanelSkeleton({ rows }) {
  return (
    <div className="status-panel" aria-hidden="true">
      {Array.from({ length: rows }, (_, index) => (
        <div className="status-row" key={index}>
          <span className="skeleton skeleton-text" style={{ width: '45%' }} />
          <span className="skeleton skeleton-text" style={{ width: '20%' }} />
        </div>
      ))}
    </div>
  );
}

export default function MlopsModal({ t, currentLang, onClose }) {
  const [results, setResults] = useState({});
  const [errors, setErrors] = useState([]);
  const [attempt, setAttempt] = useState(0);
  const closeButtonRef = useRef(null);
  const dialogRef = useRef(null);

  useEffect(() => {
    let isMounted = true;
    const load = (key, promise) => {
      promise
        .then((value) => isMounted && setResults((current) => ({ ...current, [key]: value })))
        .catch((err) => {
          if (!isMounted) return;
          setResults((current) => ({ ...current, [key]: null }));
          setErrors((current) => (current.includes(err.message) ? current : [...current, err.message]));
        });
    };
    load('metrics', fetchMetrics());
    load('drift', fetchDriftReport());
    load('promotion', fetchPromotionDecision());
    return () => {
      isMounted = false;
    };
  }, [attempt]);

  // Keep keyboard focus inside the dialog, and return it to the opening button afterwards.
  useEffect(() => {
    const previouslyFocused = document.activeElement;
    closeButtonRef.current?.focus();

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
      } else if (event.key === 'Tab') {
        const focusable = dialogRef.current.querySelectorAll('a[href], button:not([disabled])');
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      previouslyFocused?.focus?.();
    };
  }, [onClose]);

  const handleRetry = () => {
    setErrors([]);
    setResults({});
    setAttempt((current) => current + 1);
  };

  const { metrics, drift, promotion } = results;
  const score = (value) => value?.toFixed(4) ?? t.notAvailable;

  const driftedInputs = (drift?.features ?? []).filter((feature) => feature.drift).sort((a, b) => b.score - a.score);
  const topInputs = driftedInputs.slice(0, TOP_DRIFT_INPUTS);
  const largestScore = topInputs[0]?.score || 1;

  return (
    <div className="modal-overlay" id="mlops-modal-backdrop" onClick={onClose}>
      <div
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="mlops-modal-title"
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title">
            <Activity size={18} aria-hidden="true" />
            <h2 id="mlops-modal-title">{t.modelStatus}</h2>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} id="modal-close-btn" ref={closeButtonRef} aria-label={t.close} title={`${t.close} (Esc)`}>
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        <div className="modal-body">
          {errors.length > 0 && (
            <div className="data-error-banner modal-error" role="alert">
              <AlertTriangle size={16} aria-hidden="true" />
              <span>{errors.map((message) => errorText(message, t)).join(' ')}</span>
              <button type="button" className="layer-btn banner-retry" onClick={handleRetry} id="modal-retry-btn">
                {t.retry}
              </button>
            </div>
          )}

          <h3 className="status-heading">{t.currentModel}</h3>
          {metrics === undefined ? <PanelSkeleton rows={4} /> : metrics && (
            <div className="status-panel fade-in">
              <Row label={t.version}><strong className="status-accent">{metrics.active_model ?? t.notAvailable}</strong></Row>
              <Row label={t.f1Unseen}>{score(metrics.macro_f1_score)}</Row>
              <Row label={t.f1Later}>{score(metrics.temporal_macro_f1)}</Row>
              <Row label={t.f1NoLabels}>{score(metrics.temporal_macro_f1_without_label_inputs)}</Row>
              <p className="status-note">{t.scoreNote}</p>
            </div>
          )}

          <h3 className="status-heading">{t.promotion}</h3>
          {promotion === undefined ? <PanelSkeleton rows={3} /> : promotion && (
            <div className="status-panel fade-in">
              <Row label={t.result}>
                <strong className={promotion.promoted ? 'status-good' : 'status-bad'}>
                  {promotion.promoted ? t.promoted : t.rejected}
                </strong>
              </Row>
              <Row label={t.candidateModel}>{promotion.candidate_model}</Row>
              <Row label={t.candidateScore}>{promotion.candidate_score}</Row>
              <Row label={t.minimumScore}>{promotion.min_score}</Row>
              <Row label={t.activeVersion}>{promotion.active_version}</Row>
            </div>
          )}

          <h3 className="status-heading">{t.driftTitle}</h3>
          {drift === undefined ? <PanelSkeleton rows={3} /> : drift && (
            <div className="status-panel fade-in">
              <Row label={
                <span className="status-with-icon">
                  {drift.drift_detected ? <AlertTriangle size={15} className="status-bad" aria-hidden="true" /> : <CheckCircle size={15} className="status-good" aria-hidden="true" />}
                  {drift.drift_detected ? t.driftDetected : t.driftNone}
                </span>
              }>
                <strong>{t.driftShare(Math.round(drift.drifted_share * 100))}</strong>
              </Row>
              <Row label={t.compared}>{t.comparedValue(formatDate(drift.current_date, currentLang))}</Row>

              {topInputs.length > 0 && (
                <>
                  <h4 className="status-subheading">{t.mostChanged}</h4>
                  <ul className="drift-list" id="drift-top-inputs">
                    {topInputs.map((input) => (
                      <li key={input.feature} className="drift-item" title={`${input.test}: ${input.score}`}>
                        <span>{t.features[input.feature] ?? input.feature}</span>
                        <span className="driver-bar-bg drift-bar" aria-hidden="true">
                          <span className="driver-bar-fill red" style={{ width: `${(input.score / largestScore) * 100}%`, display: 'block' }} />
                        </span>
                      </li>
                    ))}
                  </ul>
                  {driftedInputs.length > topInputs.length && (
                    <p className="status-note">{t.andMore(driftedInputs.length - topInputs.length)}</p>
                  )}
                </>
              )}

              <p className="status-note">{t.driftNote}</p>
              <a href={driftReportUrl} target="_blank" rel="noreferrer" id="drift-report-link" className="status-link">
                {t.openReport} <ExternalLink size={12} aria-hidden="true" />
              </a>
            </div>
          )}

          <h3 className="status-heading">{t.tools}</h3>
          <div className="status-panel">
            <div className="tool-links" id="tool-links">
              {TOOL_LINKS.map(({ label, url }) => (
                <a key={label} href={url} target="_blank" rel="noreferrer" className="layer-btn">
                  {label} <ExternalLink size={12} aria-hidden="true" />
                </a>
              ))}
            </div>
            <p className="status-note">{t.toolsNote}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

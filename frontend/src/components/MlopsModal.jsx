import React, { useEffect, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle, ExternalLink, X } from 'lucide-react';
import { driftReportUrl, fetchDriftReport, fetchMetrics, fetchPromotionDecision } from '../services/api';

const sectionHeading = { fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.6rem' };
const panel = { background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '0.75rem', marginBottom: '1.25rem' };
const row = { display: 'flex', justifyContent: 'space-between', gap: '1rem', padding: '0.35rem 0', fontSize: '0.8rem' };

export default function MlopsModal({ isOpen, onClose }) {
  const [metrics, setMetrics] = useState(null);
  const [drift, setDrift] = useState(null);
  const [promotion, setPromotion] = useState(null);
  const [errors, setErrors] = useState([]);

  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;
    const addError = (err) => {
      if (isMounted) setErrors((current) => (current.includes(err.message) ? current : [...current, err.message]));
    };

    setErrors([]);
    fetchMetrics().then((res) => isMounted && setMetrics(res)).catch(addError);
    fetchDriftReport().then((res) => isMounted && setDrift(res)).catch(addError);
    fetchPromotionDecision().then((res) => isMounted && setPromotion(res)).catch(addError);

    return () => {
      isMounted = false;
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const driftedFeatures = drift?.features?.filter((feature) => feature.drift) ?? [];

  return (
    <div className="modal-overlay" id="mlops-modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Activity size={18} color="var(--brand-cyan)" />
            <h3 style={{ fontFamily: 'var(--font-heading)', color: '#fff', fontSize: '1.15rem' }}>
              Model Status
            </h3>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} id="modal-close-btn">
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          {errors.map((message) => (
            <div key={message} className="data-error-banner" role="alert" style={{ margin: '0 0 0.75rem' }}>
              <AlertTriangle size={16} />
              <span>{message}</span>
            </div>
          ))}

          {metrics && (
            <>
              <h4 style={sectionHeading}>Current Model</h4>
              <div style={panel}>
                <div style={row}>
                  <span>Version</span>
                  <strong style={{ color: 'var(--brand-cyan)' }}>{metrics.active_model ?? 'Not available'}</strong>
                </div>
                <div style={row}>
                  <span>Macro F1, unseen cells</span>
                  <span>{metrics.macro_f1_score?.toFixed(4) ?? 'Not available'}</span>
                </div>
                <div style={row}>
                  <span>Macro F1, later dates</span>
                  <span>{metrics.temporal_macro_f1?.toFixed(4) ?? 'Not available'}</span>
                </div>
                <div style={row}>
                  <span>Later dates, without label inputs</span>
                  <span>{metrics.temporal_macro_f1_without_label_inputs?.toFixed(4) ?? 'Not available'}</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                  {metrics.score_note}
                </div>
              </div>
            </>
          )}

          {promotion && (
            <>
              <h4 style={sectionHeading}>Latest Promotion Decision</h4>
              <div style={panel}>
                <div style={row}>
                  <span>Result</span>
                  <strong style={{ color: promotion.promoted ? '#34d399' : '#fca5a5' }}>
                    {promotion.promoted ? 'Promoted' : 'Rejected'}
                  </strong>
                </div>
                <div style={row}>
                  <span>Active version</span>
                  <span>{promotion.active_version}</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>{promotion.reason}</div>
              </div>
            </>
          )}

          {drift && (
            <>
              <h4 style={sectionHeading}>Data Drift (Evidently)</h4>
              <div style={panel}>
                <div style={{ ...row, alignItems: 'center' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    {drift.drift_detected ? <AlertTriangle size={15} color="#fca5a5" /> : <CheckCircle size={15} color="#34d399" />}
                    {drift.drift_detected ? 'Drift detected' : 'No drift detected'}
                  </span>
                  <strong>{Math.round(drift.drifted_share * 100)}% of inputs</strong>
                </div>
                <div style={row}>
                  <span>Compared</span>
                  <span>{drift.current_date} with data {drift.reference_period}</span>
                </div>
                {driftedFeatures.length > 0 && (
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.3rem' }}>
                    Drifted: {driftedFeatures.map((feature) => feature.feature).join(', ')}
                  </div>
                )}
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>{drift.note}</div>
                <a
                  href={driftReportUrl}
                  target="_blank"
                  rel="noreferrer"
                  id="drift-report-link"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.78rem', color: 'var(--brand-cyan)', marginTop: '0.6rem' }}
                >
                  Open full report <ExternalLink size={12} />
                </a>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

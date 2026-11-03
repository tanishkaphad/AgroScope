import React, { useEffect, useState } from 'react';
import { Activity, AlertTriangle, X } from 'lucide-react';
import { fetchMetrics } from '../services/api';

export default function MlopsModal({ isOpen, onClose }) {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;
    setError(null);

    fetchMetrics()
      .then((res) => {
        if (isMounted) setMetrics(res);
      })
      .catch((err) => {
        if (isMounted) setError(err.message);
      });

    return () => {
      isMounted = false;
    };
  }, [isOpen]);

  if (!isOpen) return null;

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
          {error && (
            <div className="data-error-banner" role="alert" style={{ marginBottom: '1.25rem' }}>
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}

          {metrics && (
            <div style={{ marginBottom: '1.25rem' }}>
              <h4 style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.6rem' }}>
                Current Model
              </h4>
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.8rem' }}>
                  <span>Version:</span>
                  <strong style={{ color: 'var(--brand-cyan)' }}>{metrics.active_model ?? 'Not available'}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', fontSize: '0.8rem' }}>
                  <span>Macro F1:</span>
                  <span style={{ color: 'var(--text-secondary)' }}>{metrics.macro_f1_score?.toFixed(4) ?? 'Not available'}</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                  {metrics.score_note}
                </div>
              </div>
            </div>
          )}

          <h4 style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.6rem' }}>
            Not Built Yet
          </h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Drift monitoring and automated retraining aren't available yet.
          </p>
        </div>
      </div>
    </div>
  );
}

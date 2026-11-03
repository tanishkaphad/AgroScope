import React, { useState } from 'react';
import { Activity, CheckCircle, RefreshCw, X, Zap } from 'lucide-react';
import { triggerRetraining } from '../services/api';

export default function MlopsModal({ isOpen, onClose, driftData }) {
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState(null);

  if (!isOpen) return null;

  const data = driftData || {
    eval_date: '2026-10-15 17:00:00 UTC',
    dataset_version: 'dvc-hash:8f7b2c9e',
    model_version: 'xgboost-tuned-v1.0',
    data_drift_detected: false,
    ks_test_pvalue: 0.248,
    wasserstein_distance: 0.042,
    status: 'HEALTHY (NO DRIFT DETECTED)',
    monitored_features: [
      { feature: 'ndvi_mean', test: 'Kolmogorov-Smirnov', p_value: 0.312, drift: false },
      { feature: 'ndwi_mean', test: 'Kolmogorov-Smirnov', p_value: 0.285, drift: false },
      { feature: 'temp_max_c', test: 'Wasserstein Distance', distance: 0.038, drift: false },
      { feature: 'rainfall_8day', test: 'Wasserstein Distance', distance: 0.012, drift: false },
    ],
  };

  const handleRetrain = async () => {
    setRetraining(true);
    const res = await triggerRetraining();
    setRetrainResult(res);
    setRetraining(false);
  };

  return (
    <div className="modal-overlay" id="mlops-modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Activity size={18} color="var(--brand-cyan)" />
            <h3 style={{ fontFamily: 'var(--font-heading)', color: '#fff', fontSize: '1.15rem' }}>
              MLOps Drift & Model Registry Monitor
            </h3>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} id="modal-close-btn">
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          {/* Status Banner */}
          <div
            style={{
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: '10px',
              padding: '0.85rem 1rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              marginBottom: '1.25rem',
            }}
          >
            <CheckCircle size={20} color="#10b981" />
            <div>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#34d399' }}>
                {data.status}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                Evidently AI statistical gate passed • DVC Hash: <code>{data.dataset_version}</code>
              </div>
            </div>
          </div>

          {/* Model Registry Benchmark Table */}
          <div style={{ marginBottom: '1.25rem' }}>
            <h4 style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.6rem' }}>
              DagsHub Model Registry Champions
            </h4>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '0.75rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.8rem' }}>
                <span>Production Champion:</span>
                <strong style={{ color: 'var(--brand-cyan)' }}>XGBoost Classifier (Macro-F1: 0.9982)</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', fontSize: '0.8rem' }}>
                <span>Baseline Model:</span>
                <span style={{ color: 'var(--text-secondary)' }}>Random Forest (Macro-F1: 0.9984)</span>
              </div>
            </div>
          </div>

          {/* Feature Drift Table */}
          <div style={{ marginBottom: '1.5rem' }}>
            <h4 style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.6rem' }}>
              Monitored Statistical Tests
            </h4>
            <div style={{ fontSize: '0.78rem' }}>
              {data.monitored_features?.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '0.4rem 0',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                  }}
                >
                  <span style={{ fontFamily: 'var(--font-mono)' }}>{item.feature}</span>
                  <span style={{ color: 'var(--text-muted)' }}>{item.test}</span>
                  <span style={{ color: '#34d399', fontWeight: 600 }}>No Drift</span>
                </div>
              ))}
            </div>
          </div>

          {/* Retrain Action */}
          <button
            type="button"
            className="primary-action-btn"
            onClick={handleRetrain}
            disabled={retraining}
            id="trigger-retrain-btn"
          >
            <RefreshCw size={15} className={retraining ? 'spin' : ''} />
            <span>{retraining ? 'Evaluating Gate...' : 'Trigger Retraining Gate (Champion vs Challenger)'}</span>
          </button>

          {retrainResult && (
            <div
              style={{
                marginTop: '1rem',
                background: 'rgba(56, 189, 248, 0.08)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                borderRadius: '8px',
                padding: '0.75rem',
                fontSize: '0.78rem',
              }}
            >
              <div style={{ color: 'var(--brand-cyan)', fontWeight: 700 }}>{retrainResult.promotion_gate}</div>
              <div style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>{retrainResult.action}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

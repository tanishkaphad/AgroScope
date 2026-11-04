/**
 * AgroScope Frontend API Client.
 * Every function throws on failure, so the UI can show a real error.
 */

import { API_UNREACHABLE } from '../i18n';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const TOOL_LINKS = [
  { key: 'airflow', label: 'Airflow', url: import.meta.env.VITE_AIRFLOW_URL || 'http://localhost:8080' },
  { key: 'mlflow', label: 'MLflow', url: import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000' },
  { key: 'grafana', label: 'Grafana', url: import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3000' },
  { key: 'prometheus', label: 'Prometheus', url: import.meta.env.VITE_PROMETHEUS_URL || 'http://localhost:9090' },
  { key: 'apiDocs', label: 'API docs', url: `${new URL(API_BASE).origin}/docs` },
];

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options).catch(() => {
    throw new Error(API_UNREACHABLE);
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed with HTTP ${res.status}`);
  }
  return res.json();
}

export function fetchHealth() {
  return request('/health');
}

export function fetchMetrics() {
  return request('/metrics');
}

export function fetchDriftReport() {
  return request('/monitoring/drift');
}

export function fetchPromotionDecision() {
  return request('/monitoring/promotion');
}

export const driftReportUrl = `${API_BASE}/monitoring/drift/report`;
export const DASHBOARD_LINKS = [...TOOL_LINKS, { key: 'driftReport', label: 'Evidently', url: driftReportUrl }];

export function fetchAvailableDates() {
  return request('/available-dates');
}

export function fetchStatistics(date) {
  return request(`/statistics${date ? `?date=${date}` : ''}`);
}

export function fetchStudyArea() {
  return request('/study-area');
}

export function fetchStressMap(limit, date) {
  return request(`/stress-map?sample=true&limit=${limit}${date ? `&date=${date}` : ''}`);
}

export function fetchIrrigationMap(limit, date) {
  return request(`/irrigation-map?sample=true&limit=${limit}${date ? `&date=${date}` : ''}`);
}

export function fetchParcelDetail(cellId, lang = 'en', date) {
  return request(`/area/${encodeURIComponent(cellId)}?lang=${lang}${date ? `&date=${date}` : ''}`);
}

export function runPredict(payload) {
  return request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

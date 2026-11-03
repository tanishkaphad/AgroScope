/**
 * AgroScope Frontend API Client.
 * Every function throws on failure, so the UI can show a real error.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options).catch(() => {
    throw new Error("Can't reach the AgroScope API.");
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

export function fetchAvailableDates() {
  return request('/available-dates');
}

export function fetchStatistics() {
  return request('/statistics');
}

export function fetchStudyArea() {
  return request('/study-area');
}

export function fetchStressMap(limit = 1200) {
  return request(`/stress-map?sample=true&limit=${limit}`);
}

export function fetchIrrigationMap(limit = 1200) {
  return request(`/irrigation-map?sample=true&limit=${limit}`);
}

export function fetchParcelDetail(cellId, lang = 'en') {
  return request(`/area/${encodeURIComponent(cellId)}?lang=${lang}`);
}

export function runPredict(payload) {
  return request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

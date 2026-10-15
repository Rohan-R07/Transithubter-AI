/**
 * TransitHunter AI — API Service Module
 * 
 * Centralized communication layer connecting the React frontend to the FastAPI backend.
 * Base URL defaults to http://127.0.0.1:8000 (configurable via VITE_API_BASE_URL in .env).
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const response = await fetch(url, {
    headers: {
      'Accept': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let errorData = null;
    try {
      errorData = await response.json();
    } catch (_) {}

    const errorDetail =
      (typeof errorData?.detail === 'string' && errorData.detail) ||
      (Array.isArray(errorData?.detail) && errorData.detail.map(d => d.msg || JSON.stringify(d)).join('; ')) ||
      errorData?.message ||
      `HTTP ${response.status} (${response.statusText})`;

    throw new ApiError(errorDetail, response.status, errorData);
  }

  // 204 No Content
  if (response.status === 204) return null;

  return response.json();
}

// ============================================================
// HEALTH & STATUS
// ============================================================

/**
 * GET /health
 * Returns backend health status and loaded model readiness.
 */
export async function getHealth() {
  return request('/health');
}

/**
 * GET /model/metadata (or /model/info)
 * Returns model metadata, hyperparams, feature count (3,197), and decision thresholds.
 */
export async function getModelInfo() {
  return request('/model/metadata');
}

/**
 * GET /model/metrics
 * Returns evaluation metrics (accuracy, precision, recall, F1, confusion matrix).
 */
export async function getModelMetrics() {
  return request('/model/metrics');
}

/**
 * GET /dataset/info
 * Returns dataset metadata (sample counts, features, class distribution).
 */
export async function getDatasetInfo() {
  return request('/dataset/info');
}

// ============================================================
// EXOPLANET HUNTER / ML PREDICTION
// ============================================================

/**
 * POST /predict
 * Run exoplanet transit inference on 3,197 photometric flux values.
 * 
 * @param {number[]} fluxValues - Array of exactly 3,197 numeric flux values
 * @param {'balanced'|'high_recall'} [mode='balanced'] - Operating decision threshold
 */
export async function predict(fluxValues, mode = 'balanced') {
  return request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      flux_values: fluxValues,
      mode: mode,
    }),
  });
}

/**
 * POST /predict/sample
 * Run prediction on verified backend Kepler light-curve sample.
 * Returns sample chart points, 3,197 flux values, and prediction response.
 * 
 * @param {number} [sampleId=0]
 * @param {'balanced'|'high_recall'} [mode='balanced']
 */
export async function predictFromSample(sampleId = 0, mode = 'balanced') {
  return request(`/predict/sample?sample_id=${sampleId}&mode=${mode}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
}

// ============================================================
// COMMUNICATION NETWORK & DIJKSTRA ROUTING
// ============================================================

/**
 * GET /network/topology
 * Returns full network topology (5 space nodes + 6 links with operational latencies).
 */
export async function getNetworkTopology() {
  return request('/network/topology');
}

/**
 * POST /network/route
 * Calculate optimal communication path using Dijkstra's algorithm.
 * 
 * @param {string} source - Source node ID or Name (e.g. 'earth_ground_station')
 * @param {string} destination - Destination node ID or Name (e.g. 'exoplanet_observatory')
 */
export async function calculateRoute(source, destination) {
  return request('/network/route', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source,
      destination,
    }),
  });
}

/**
 * POST /network/interference (or /network/fail-link)
 * Simulate space signal disruption disabling a link.
 * 
 * @param {string} linkId - The link ID to disable
 * @param {string} [reason='Solar flare electromagnetic interference']
 */
export async function failLink(linkId, reason = 'Solar flare electromagnetic interference') {
  return request('/network/interference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      link_id: linkId,
      reason: reason,
    }),
  });
}

/**
 * POST /network/restore
 * Restore a disabled link back to operational status.
 * 
 * @param {string} linkId - The link ID to restore
 */
export async function restoreLink(linkId) {
  return request('/network/restore', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      link_id: linkId,
    }),
  });
}

/**
 * POST /network/reset
 * Reset all communication links back to default operational state.
 */
export async function resetNetwork() {
  return request('/network/reset', {
    method: 'POST',
  });
}

export { BASE_URL, ApiError };

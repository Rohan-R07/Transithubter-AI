import { useState, useCallback } from 'react';

/**
 * Hook for making API requests with loading/error/data state.
 * @param {Function} apiFn - The API function to call
 * @returns {{ data, loading, error, execute, reset }}
 */
export function useApiCall(apiFn) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(async (...args) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiFn(...args);
      setData(result);
      return result;
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [apiFn]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

/**
 * Format timestamp to a human-readable string.
 * @param {Date|string|number} ts
 * @returns {string}
 */
export function formatTimestamp(ts) {
  const d = ts instanceof Date ? ts : new Date(ts);
  return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/**
 * Format a number as a percentage string.
 * @param {number} value - value between 0 and 1
 * @param {number} [decimals=1]
 */
export function formatPct(value, decimals = 1) {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Generate a unique event ID.
 */
let _eventCounter = 0;
export function genEventId() {
  return `evt-${Date.now()}-${++_eventCounter}`;
}

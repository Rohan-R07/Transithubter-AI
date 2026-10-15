import { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Navigation,
  Radio,
  RotateCcw,
  Wifi,
  WifiOff,
  Zap,
} from 'lucide-react';
import { PageHeader } from '../components/layout/Sidebar';
import { SectionCard, LoadingState, ErrorState } from '../components/ui/UIComponents';
import { NetworkGraph, LinkStatus, mapBackendTopology } from '../components/network/NetworkGraph';
import { useActivity } from '../components/activity/ActivityLog';
import * as api from '../services/api';

export default function NetworkPage() {
  const { addEvent } = useActivity();
  const [topology, setTopology] = useState(null);
  const [disabled, setDisabled] = useState([]);
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [routingLoading, setRoutingLoading] = useState(false);
  const [error, setError] = useState('');

  // ── Fetch live topology from backend ────────────────────────────────────
  const loadTopology = useCallback(async () => {
    try {
      const data = await api.getNetworkTopology();
      const mapped = mapBackendTopology(data);
      setTopology(mapped);

      // Detect any disabled links from backend state
      const disabledIds = (data.links || [])
        .filter((l) => l.available === false)
        .map((l) => l.id);
      setDisabled(disabledIds);
      setError('');
    } catch (err) {
      setError(`Failed to connect to backend network service: ${err.message}`);
      // Fallback to local default map so page doesn't crash
      setTopology(mapBackendTopology(null));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTopology();
  }, [loadTopology]);

  // ── Simulate Link Failure ───────────────────────────────────────────────
  const fail = async (linkId) => {
    try {
      setError('');
      await api.failLink(linkId, 'Coronal mass ejection signal disruption');
      setDisabled((prev) => (prev.includes(linkId) ? prev : [...prev, linkId]));
      setResult(null);
      addEvent({
        type: 'warning',
        message: `Signal interference: ${linkId} disabled`,
        detail: 'Live FastAPI network state updated',
      });
      await loadTopology();
    } catch (err) {
      setError(`Failed to disable link: ${err.message}`);
    }
  };

  // ── Restore Link ────────────────────────────────────────────────────────
  const restore = async (linkId) => {
    try {
      setError('');
      await api.restoreLink(linkId);
      setDisabled((prev) => prev.filter((id) => id !== linkId));
      setResult(null);
      addEvent({
        type: 'info',
        message: `Telemetry restored: ${linkId} operational`,
        detail: 'Dijkstra route re-enabled',
      });
      await loadTopology();
    } catch (err) {
      setError(`Failed to restore link: ${err.message}`);
    }
  };

  // ── Reset Network to Operational ────────────────────────────────────────
  const reset = async () => {
    try {
      setError('');
      await api.resetNetwork();
      setDisabled([]);
      setSelected(null);
      setResult(null);
      addEvent({
        type: 'info',
        message: 'Network reset to factory operational state',
        detail: 'All links available',
      });
      await loadTopology();
    } catch (err) {
      setError(`Failed to reset network: ${err.message}`);
    }
  };

  // ── Calculate Shortest Path via Live Dijkstra ───────────────────────────
  const calculate = async () => {
    setRoutingLoading(true);
    setError('');
    try {
      const res = await api.calculateRoute('earth_ground_station', 'exoplanet_observatory');

      const routeEdges = (res.links || []).map((l) => l.id);
      setResult({
        reachable: res.reachable,
        nodes: res.path || [],
        edges: routeEdges,
        latency: res.total_latency_ms,
        distance: res.total_distance_km,
        hops: res.hops,
        message: res.message,
      });

      addEvent({
        type: 'network',
        message: res.reachable
          ? `Optimal route calculated (${res.total_latency_ms} ms, ${res.hops} hops)`
          : 'Destination unreachable (network partitioned)',
        detail: res.reachable ? res.path.join(' → ') : res.message,
      });
    } catch (err) {
      setError(`Dijkstra calculation error: ${err.message}`);
    } finally {
      setRoutingLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-container animate-fade-in" style={{ padding: '60px 0' }}>
        <LoadingState message="Connecting to Deep-Space Communication Service..." />
      </div>
    );
  }

  const activeLinkCount = (topology?.edges?.length || 0) - disabled.length;

  return (
    <div className="page-container animate-fade-in">
      <PageHeader
        title="Communication Network"
        subtitle="Dynamic Dijkstra shortest-path routing over deep-space telemetry links"
        actions={
          <button className="btn btn-secondary btn-sm" onClick={reset}>
            <RotateCcw size={14} /> Reset network
          </button>
        }
      />

      <div style={{ padding: '20px 0' }}>
        {/* Status Banner */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '10px 14px',
            marginBottom: 16,
            borderRadius: 10,
            border: error
              ? '1px solid rgba(239,68,68,.3)'
              : '1px solid rgba(16,185,129,.3)',
            background: error ? 'rgba(239,68,68,.06)' : 'rgba(16,185,129,.06)',
            color: error ? 'var(--color-red)' : 'var(--color-green)',
            fontSize: 13,
          }}
        >
          {error ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
          <span>
            {error ||
              'Connected to FastAPI Network Engine · Dijkstra Latency Optimization Live · Real-Time Dynamic Rerouting'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 320px', gap: 16 }}>
          {/* Interactive Topology Graph */}
          <SectionCard
            title="Relay Topology (Interactive)"
            titleIcon={Radio}
            headerRight={
              <span style={{ fontSize: 12, color: 'var(--color-cyan)', fontFamily: 'var(--font-mono)' }}>
                {activeLinkCount} / {topology?.edges?.length || 0} Links Active
              </span>
            }
          >
            <div
              style={{
                border: '1px solid var(--color-border-muted)',
                borderRadius: 10,
                overflow: 'hidden',
              }}
            >
              {topology && (
                <NetworkGraph
                  topology={topology}
                  disabledLinks={disabled}
                  routeEdges={result?.edges || []}
                  selectedEdgeId={selected}
                  onEdgeSelect={setSelected}
                />
              )}
            </div>

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginTop: 14 }}
              onClick={calculate}
              disabled={routingLoading}
            >
              <Navigation size={15} />{' '}
              {routingLoading ? 'Calculating optimal path…' : 'Calculate Dijkstra Route (Earth → Observatory)'}
            </button>

            {/* Routing Result Banner */}
            {result && (
              <div
                style={{
                  marginTop: 12,
                  padding: 14,
                  borderRadius: 8,
                  border: result.reachable
                    ? '1px solid rgba(0,212,255,.3)'
                    : '1px solid rgba(239,68,68,.3)',
                  background: result.reachable
                    ? 'rgba(0,212,255,.07)'
                    : 'rgba(239,68,68,.07)',
                  color: result.reachable ? 'var(--color-cyan)' : 'var(--color-red)',
                  fontSize: 13,
                }}
              >
                <div style={{ fontWeight: 650, display: 'flex', alignItems: 'center', gap: 6 }}>
                  {result.reachable ? '✓ Optimal Route Found' : '✕ Destination Unreachable'}
                </div>
                {result.reachable ? (
                  <div style={{ marginTop: 6, lineHeight: 1.5 }}>
                    <div style={{ color: 'var(--color-text-primary)' }}>
                      <strong>Path:</strong> {result.nodes.join(' → ')}
                    </div>
                    <div style={{ marginTop: 4, display: 'flex', gap: 14, fontSize: 12, color: 'var(--color-text-muted)' }}>
                      <span>Latency: <strong style={{ color: 'var(--color-cyan)' }}>{result.latency} ms</strong></span>
                      <span>Hops: <strong style={{ color: 'var(--color-cyan)' }}>{result.hops}</strong></span>
                      {result.distance && (
                        <span>Distance: <strong>{(result.distance).toLocaleString()} km</strong></span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div style={{ marginTop: 6, color: 'var(--color-text-secondary)', fontSize: 12 }}>
                    {result.message}
                  </div>
                )}
              </div>
            )}
          </SectionCard>

          {/* Controls Panel */}
          <div style={{ display: 'grid', gap: 14 }}>
            <SectionCard title="Link Status & Fault Injection" titleIcon={Activity}>
              <p style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 10 }}>
                Click <strong>Fail</strong> on any link to simulate space signal disruption. The live backend will dynamically reroute around the failure.
              </p>
              {topology && (
                <LinkStatus
                  edges={topology.edges}
                  disabledLinks={disabled}
                  selectedEdgeId={selected}
                  onEdgeSelect={setSelected}
                  onFailLink={fail}
                  onRestoreLink={restore}
                />
              )}
            </SectionCard>

            <SectionCard title="Mission Telemetry" titleIcon={Zap}>
              <div style={{ display: 'grid', gap: 8, fontSize: 13 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--color-text-muted)' }}><Wifi size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} /> Operational Links:</span>
                  <span style={{ fontWeight: 600, color: 'var(--color-green)' }}>{activeLinkCount}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--color-text-muted)' }}><WifiOff size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} /> Disrupted Links:</span>
                  <span style={{ fontWeight: 600, color: disabled.length > 0 ? 'var(--color-red)' : 'var(--color-text-dim)' }}>{disabled.length}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--color-text-muted)' }}>Routing Algorithm:</span>
                  <span style={{ fontWeight: 600, color: 'var(--color-cyan)' }}>Dijkstra (Latency Weighted)</span>
                </div>
                <div style={{ marginTop: 6, fontSize: 11, color: 'var(--color-text-dim)', lineHeight: 1.4 }}>
                  Disrupting <code>link_earth_relay_a</code> triggers dynamic rerouting via Orbital Relay B (370.0 ms).
                </div>
              </div>
            </SectionCard>
          </div>
        </div>
      </div>
    </div>
  );
}

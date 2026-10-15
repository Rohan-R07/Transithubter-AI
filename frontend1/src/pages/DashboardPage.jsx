import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Cpu, Database, Radio, Telescope, Wifi, WifiOff, RefreshCw, CheckCircle, Zap } from 'lucide-react';
import { PageHeader } from '../components/layout/Sidebar';
import { ActivityLog } from '../components/activity/ActivityLog';
import { MetricCard, SectionCard, StatusBadge, InlineLoader } from '../components/ui/UIComponents';
import { getHealth, getModelInfo, getNetworkTopology, getDatasetInfo } from '../services/api';

export default function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [backendStatus, setBackendStatus] = useState({
    connected: false,
    health: null,
    model: null,
    dataset: null,
    network: null,
    error: null,
  });

  const fetchDashboardData = async (isManual = false) => {
    if (isManual) setRefreshing(true);
    else setLoading(true);

    try {
      const [healthRes, modelRes, datasetRes, networkRes] = await Promise.allSettled([
        getHealth(),
        getModelInfo(),
        getDatasetInfo(),
        getNetworkTopology(),
      ]);

      const health = healthRes.status === 'fulfilled' ? healthRes.value : null;
      const model = modelRes.status === 'fulfilled' ? modelRes.value : null;
      const dataset = datasetRes.status === 'fulfilled' ? datasetRes.value : null;
      const network = networkRes.status === 'fulfilled' ? networkRes.value : null;

      const isConnected = !!health;

      setBackendStatus({
        connected: isConnected,
        health,
        model,
        dataset,
        network,
        error: isConnected ? null : 'Cannot reach backend at http://127.0.0.1:8000',
      });
    } catch (err) {
      setBackendStatus(prev => ({
        ...prev,
        connected: false,
        error: err.message,
      }));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const modules = [
    {
      to: '/exoplanet',
      icon: Telescope,
      title: 'Exoplanet Hunter',
      text: 'Analyze 3,197-point Kepler light curves with trained Random Forest classifier and tunable recall thresholds.',
      color: 'var(--color-cyan)',
      status: backendStatus.model ? 'Active (Phase 1.5)' : 'Ready',
    },
    {
      to: '/network',
      icon: Radio,
      title: 'Resilient Space Mesh',
      text: 'Simulate space communication links, compute Dijkstra shortest-paths, and inject link interference with auto-rerouting.',
      color: 'var(--color-violet)',
      status: backendStatus.network ? `${backendStatus.network.nodes?.length || 5} Nodes Active` : 'Ready',
    },
  ];

  const activeLinksCount = backendStatus.network?.links?.filter(l => l.available !== false).length ?? 7;
  const totalLinksCount = backendStatus.network?.links?.length ?? 7;

  return (
    <div className="page-container animate-fade-in">
      <PageHeader
        title="Mission Control"
        subtitle="TRANSITHUBTER AI · Resilient Exoplanet Detection & Space Communication Mesh"
      />

      <div style={{ padding: '24px 0', display: 'flex', flexDirection: 'column', gap: 20 }}>
        {/* Connection Notice Banner */}
        {backendStatus.connected ? (
          <div
            style={{
              padding: '12px 16px',
              border: '1px solid rgba(16, 185, 129, 0.35)',
              borderRadius: 10,
              background: 'rgba(16, 185, 129, 0.08)',
              color: '#34d399',
              fontSize: 13,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <CheckCircle size={16} />
              <span>
                <strong>FastAPI Backend Connected:</strong> Live ML inference (Phase 1.5), NASA Kepler dataset telemetry, and Dijkstra routing online.
              </span>
            </div>
            <button
              onClick={() => fetchDashboardData(true)}
              disabled={refreshing}
              className="btn btn-secondary btn-sm"
              style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, padding: '4px 10px' }}
            >
              <RefreshCw size={12} className={refreshing ? 'spin' : ''} />
              {refreshing ? 'Syncing...' : 'Refresh Telemetry'}
            </button>
          </div>
        ) : (
          <div
            style={{
              padding: '12px 16px',
              border: '1px solid rgba(239, 68, 68, 0.35)',
              borderRadius: 10,
              background: 'rgba(239, 68, 68, 0.08)',
              color: '#f87171',
              fontSize: 13,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <WifiOff size={16} />
              <span>
                <strong>Backend Disconnected:</strong> Could not reach <code>http://127.0.0.1:8000</code>. Ensure Uvicorn is running.
              </span>
            </div>
            <button
              onClick={() => fetchDashboardData(true)}
              disabled={refreshing}
              className="btn btn-secondary btn-sm"
              style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, padding: '4px 10px' }}
            >
              <RefreshCw size={12} className={refreshing ? 'spin' : ''} />
              Retry Connection
            </button>
          </div>
        )}

        {/* 4 Metric Cards */}
        <div className="grid-cols-4">
          <MetricCard
            label="Model Status"
            value={
              loading ? (
                <InlineLoader size={16} />
              ) : backendStatus.model ? (
                <StatusBadge variant="online" label={backendStatus.model.model_type || 'Phase 1.5 Active'} />
              ) : (
                <StatusBadge variant="warning" label="Model Offline" />
              )
            }
            sub={
              backendStatus.model
                ? `Default: ${(backendStatus.model.decision_threshold * 100).toFixed(0)}% · Recall: ${(backendStatus.model.high_recall_threshold * 100).toFixed(0)}%`
                : 'Backend model uninitialized'
            }
            icon={Cpu}
            colorClass="violet"
          />

          <MetricCard
            label="Kepler Dataset"
            value={
              loading ? (
                <InlineLoader size={16} />
              ) : backendStatus.dataset ? (
                `${backendStatus.dataset.total_samples?.toLocaleString() || '5,657'} Curves`
              ) : (
                'exoTest.csv (3,197 pts)'
              )
            }
            sub={
              backendStatus.dataset
                ? `${backendStatus.dataset.test_samples} test · ${backendStatus.dataset.exoplanet_candidates_in_test} verified transit`
                : '3,197 flux features per star'
            }
            icon={Database}
            colorClass="cyan"
          />

          <MetricCard
            label="Network Mesh"
            value={
              loading ? (
                <InlineLoader size={16} />
              ) : backendStatus.network ? (
                <StatusBadge
                  variant={activeLinksCount === totalLinksCount ? 'online' : 'warning'}
                  label={`${activeLinksCount}/${totalLinksCount} Links Up`}
                />
              ) : (
                <StatusBadge variant="info" label="Mesh Simulation" />
              )
            }
            sub="Dijkstra Shortest-Path Latency"
            icon={Radio}
            colorClass="amber"
          />

          <MetricCard
            label="API Connection"
            value={
              loading ? (
                <InlineLoader size={16} />
              ) : backendStatus.connected ? (
                <StatusBadge variant="online" label="FastAPI 8000" />
              ) : (
                <StatusBadge variant="offline" label="Offline" />
              )
            }
            sub={backendStatus.connected ? 'CORS & REST Endpoints Active' : 'Start backend server'}
            icon={backendStatus.connected ? Wifi : WifiOff}
            colorClass={backendStatus.connected ? 'cyan' : 'red'}
          />
        </div>

        {/* Main Grid */}
        <div
          className="dashboard-main-grid"
          style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) minmax(280px, .8fr)', gap: 20 }}
        >
          <SectionCard title="Mission Operational Modules" titleIcon={Zap}>
            <div style={{ display: 'grid', gap: 14 }}>
              {modules.map(({ to, icon: Icon, title, text, color, status }) => (
                <Link key={to} to={to} style={{ textDecoration: 'none' }}>
                  <div
                    style={{
                      display: 'flex',
                      gap: 16,
                      padding: 18,
                      border: '1px solid var(--color-border-muted)',
                      borderRadius: 12,
                      background: 'var(--color-bg-elevated)',
                      transition: 'border-color 0.2s, transform 0.2s',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.borderColor = color;
                      e.currentTarget.style.transform = 'translateY(-2px)';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.borderColor = 'var(--color-border-muted)';
                      e.currentTarget.style.transform = 'none';
                    }}
                  >
                    <div
                      style={{
                        padding: 10,
                        borderRadius: 10,
                        background: 'rgba(255, 255, 255, 0.04)',
                        height: 'fit-content',
                      }}
                    >
                      <Icon size={24} style={{ color }} />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ color: 'var(--color-text-primary)', fontWeight: 650, fontSize: 16 }}>
                          {title}
                        </div>
                        <span style={{ fontSize: 11, color: 'var(--color-text-muted)', background: 'rgba(255, 255, 255, 0.06)', padding: '2px 8px', borderRadius: 6 }}>
                          {status}
                        </span>
                      </div>
                      <div style={{ marginTop: 6, color: 'var(--color-text-muted)', fontSize: 13, lineHeight: 1.5 }}>
                        {text}
                      </div>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="Mission Activity" titleIcon={Activity}>
            <ActivityLog compact />
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

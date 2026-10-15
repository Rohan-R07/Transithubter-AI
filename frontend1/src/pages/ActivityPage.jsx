import { PageHeader } from '../components/layout/Sidebar';
import { SectionCard, EmptyState } from '../components/ui/UIComponents';
import { useActivity } from '../components/activity/ActivityLog';

export default function ActivityPage() {
  const { events } = useActivity();
  return <div className="page-container animate-fade-in"><PageHeader title="Mission Activity" subtitle="Frontend interaction log for this browser session" />
    <div style={{ padding: '24px 0' }}><SectionCard title="Recent events">
      {!events.length ? <EmptyState title="No activity recorded" description="Use Exoplanet Hunter or Communication Network to create local demo events." /> : <div style={{ display: 'grid', gap: 8 }}>{events.map(event => <div key={event.id} style={{ padding: '12px 14px', borderRadius: 8, border: '1px solid var(--color-border-muted)', background: 'var(--color-bg-elevated)' }}><div style={{ color: 'var(--color-text-primary)', fontSize: 13, fontWeight: 600 }}>{event.message}</div>{event.detail && <div style={{ marginTop: 3, color: 'var(--color-text-muted)', fontSize: 12 }}>{event.detail}</div>}<div style={{ marginTop: 6, color: 'var(--color-text-dim)', fontSize: 11 }}>{new Date(event.timestamp).toLocaleTimeString()}</div></div>)}</div>}
    </SectionCard></div>
  </div>;
}

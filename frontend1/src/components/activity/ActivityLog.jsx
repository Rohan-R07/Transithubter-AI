import { useState, createContext, useContext, useCallback, useRef } from 'react';
import { genEventId, formatTimestamp } from '../../hooks/useApiCall';

const ActivityContext = createContext(null);

export function ActivityProvider({ children }) {
  const [events, setEvents] = useState([]);

  const addEvent = useCallback(({ type = 'info', message, detail }) => {
    const event = {
      id: genEventId(),
      type,
      message,
      detail,
      timestamp: new Date(),
    };
    setEvents(prev => [event, ...prev].slice(0, 50));
  }, []);

  return (
    <ActivityContext.Provider value={{ events, addEvent }}>
      {children}
    </ActivityContext.Provider>
  );
}

export function useActivity() {
  const ctx = useContext(ActivityContext);
  if (!ctx) throw new Error('useActivity must be used within ActivityProvider');
  return ctx;
}

// Color for each event type
function eventColor(type) {
  switch (type) {
    case 'success': return 'var(--color-green)';
    case 'error': return 'var(--color-red)';
    case 'warning': return 'var(--color-amber)';
    case 'network': return 'var(--color-violet)';
    case 'predict': return 'var(--color-cyan)';
    default: return 'var(--color-text-muted)';
  }
}

export function ActivityLog({ maxItems = 20, compact = false }) {
  const { events } = useActivity();
  const displayed = events.slice(0, maxItems);

  if (displayed.length === 0) {
    return (
      <div style={{ padding: '24px 0', textAlign: 'center', color: 'var(--color-text-dim)', fontSize: 13 }}>
        No activity recorded yet.
      </div>
    );
  }

  return (
    <div>
      {displayed.map(evt => (
        <div key={evt.id} className="activity-item animate-fade-in">
          <div
            className="activity-dot"
            style={{ background: eventColor(evt.type), marginTop: compact ? 6 : 7 }}
          />
          <div className="activity-content">
            <div className="activity-message">{evt.message}</div>
            {evt.detail && !compact && (
              <div style={{ fontSize: 12, color: 'var(--color-text-dim)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                {evt.detail}
              </div>
            )}
            <div className="activity-time">{formatTimestamp(evt.timestamp)}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

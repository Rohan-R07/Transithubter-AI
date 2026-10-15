import { Loader2 } from 'lucide-react';

/**
 * Full loading state with spinner and optional message.
 */
export function LoadingState({ message = 'Processing...', size = 'md' }) {
  return (
    <div className="loading-state animate-fade-in">
      <div className={`loading-spinner ${size === 'lg' ? 'loading-spinner-lg' : ''}`} />
      <p style={{ color: 'var(--color-text-secondary)', fontSize: '14px' }}>{message}</p>
    </div>
  );
}

/**
 * Inline loading indicator.
 */
export function InlineLoader({ size = 16 }) {
  return (
    <Loader2
      size={size}
      style={{ color: 'var(--color-cyan)', animation: 'spin 0.8s linear infinite' }}
    />
  );
}

/**
 * Error display box.
 */
export function ErrorState({ message, onRetry }) {
  return (
    <div className="error-state animate-fade-in">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--color-red)' }}>
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
      <p style={{ color: 'var(--color-red)', fontSize: '14px', fontWeight: 500 }}>
        {message || 'An error occurred.'}
      </p>
      {onRetry && (
        <button className="btn btn-secondary btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

/**
 * Empty state placeholder.
 */
export function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="empty-state animate-fade-in">
      {Icon && <Icon size={36} style={{ color: 'var(--color-text-dim)' }} />}
      {title && <p style={{ color: 'var(--color-text-secondary)', fontWeight: 600 }}>{title}</p>}
      {description && <p style={{ color: 'var(--color-text-muted)', fontSize: '13px' }}>{description}</p>}
    </div>
  );
}

/**
 * Status badge component.
 * variant: 'online' | 'offline' | 'warning' | 'info' | 'violet'
 */
export function StatusBadge({ variant = 'info', label, showDot = true }) {
  return (
    <span className={`badge badge-${variant}`}>
      {showDot && <span className="badge-dot" />}
      {label}
    </span>
  );
}

/**
 * Metric summary card.
 */
export function MetricCard({ label, value, sub, colorClass = 'cyan', icon: Icon }) {
  return (
    <div className={`metric-card ${colorClass} animate-fade-in`}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span className="metric-label">{label}</span>
        {Icon && <Icon size={14} style={{ color: 'var(--color-text-muted)' }} />}
      </div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

/**
 * Section card with optional title and header actions.
 */
export function SectionCard({ title, titleIcon: TitleIcon, headerRight, children, style }) {
  return (
    <div className="card" style={style}>
      {title && (
        <div className="card-header">
          <div className="card-title-icon">
            {TitleIcon && <TitleIcon size={14} style={{ color: 'var(--color-text-muted)' }} />}
            <span className="card-title">{title}</span>
          </div>
          {headerRight}
        </div>
      )}
      {children}
    </div>
  );
}

/**
 * Horizontal progress bar.
 */
export function ProgressBar({ value, max = 1, colorClass = 'cyan', label, showPct = true }) {
  const pct = Math.min(1, Math.max(0, value / max)) * 100;
  return (
    <div>
      {(label || showPct) && (
        <div className="flex-between" style={{ marginBottom: 6 }}>
          {label && <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>{label}</span>}
          {showPct && <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--color-text-muted)' }}>{pct.toFixed(1)}%</span>}
        </div>
      )}
      <div className="progress-bar-wrap">
        <div className={`progress-bar progress-bar-${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/**
 * Demo data label chip.
 */
export function DemoLabel() {
  return <span className="demo-label">Demo Data</span>;
}

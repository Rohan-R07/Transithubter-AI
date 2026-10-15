import { useState, useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  LayoutDashboard, Telescope, Radio, Activity,
  Menu, X, ChevronRight, Satellite, Zap
} from 'lucide-react';

const NAV_ITEMS = [
  {
    section: 'Mission',
    items: [
      { path: '/', label: 'Mission Control', icon: LayoutDashboard },
      { path: '/activity', label: 'Mission Activity', icon: Activity },
    ]
  },
  {
    section: 'Systems',
    items: [
      { path: '/exoplanet', label: 'Exoplanet Hunter', icon: Telescope },
      { path: '/network', label: 'Comm Network', icon: Radio },
    ]
  }
];

export function Sidebar({ collapsed, onToggle }) {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Sidebar */}
      <aside className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}>
        {/* Logo */}
        <div className="sidebar-header">
          <div className="logo-mark">
            <div className="logo-icon">
              <Satellite size={18} style={{ color: 'white' }} />
            </div>
            <div>
              <div className="logo-text">TransitHunter</div>
              <div className="logo-sub">AI Mission Console</div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(section => (
            <div key={section.section}>
              <div className="nav-section-label">{section.section}</div>
              {section.items.map(item => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`nav-item ${isActive ? 'active' : ''}`}
                  >
                    <Icon className="nav-icon" size={16} />
                    {item.label}
                    {isActive && (
                      <ChevronRight size={12} style={{ marginLeft: 'auto', color: 'var(--color-cyan)' }} />
                    )}
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px' }}>
            <Zap size={12} style={{ color: 'var(--color-cyan)' }} />
            <span style={{ fontSize: 11, color: 'var(--color-text-dim)', fontFamily: 'var(--font-mono)' }}>
              v1.0.0 · Hackathon Build
            </span>
          </div>
        </div>
      </aside>

      {/* Mobile Toggle */}
      <button
        onClick={() => { setMobileOpen(open => !open); onToggle(); }}
        className="btn btn-secondary btn-sm"
        style={{
          position: 'fixed',
          top: 16,
          left: collapsed ? 16 : 'calc(var(--sidebar-width) + 16px)',
          zIndex: 200,
          width: 36,
          height: 36,
          padding: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'left var(--transition-base)',
        }}
        aria-label="Toggle sidebar"
      >
        {collapsed ? <Menu size={16} /> : <X size={16} />}
      </button>
    </>
  );
}

export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="page-header">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {actions && <div style={{ display: 'flex', gap: 8 }}>{actions}</div>}
    </div>
  );
}

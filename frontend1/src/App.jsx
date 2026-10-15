import { useState, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './App.css';
import { Sidebar } from './components/layout/Sidebar';
import { ActivityProvider } from './components/activity/ActivityLog';
import { LoadingState } from './components/ui/UIComponents';

// ── Lazy-load pages for code-splitting ───────────────────────────────────────
const DashboardPage  = lazy(() => import('./pages/DashboardPage'));
const ExoplanetPage  = lazy(() => import('./pages/ExoplanetPage'));
const NetworkPage    = lazy(() => import('./pages/NetworkPage'));
const ActivityPage   = lazy(() => import('./pages/ActivityPage'));

// ── Page suspense wrapper ─────────────────────────────────────────────────────
function PageSuspense({ children }) {
  return (
    <Suspense
      fallback={
        <div className="page-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 320 }}>
          <LoadingState message="Loading module…" />
        </div>
      }
    >
      {children}
    </Suspense>
  );
}

// ── App shell ─────────────────────────────────────────────────────────────────
function AppShell() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed-layout' : ''}`}>
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
      />

      <main className="main-content">
        <Routes>
          <Route path="/"          element={<PageSuspense><DashboardPage /></PageSuspense>} />
          <Route path="/exoplanet" element={<PageSuspense><ExoplanetPage /></PageSuspense>} />
          <Route path="/network"   element={<PageSuspense><NetworkPage /></PageSuspense>} />
          <Route path="/activity"  element={<PageSuspense><ActivityPage /></PageSuspense>} />
          {/* 404 fallback */}
          <Route path="*" element={
            <div className="page-container" style={{ textAlign: 'center', paddingTop: 80 }}>
              <div style={{ fontSize: 64, marginBottom: 16 }}>🛰️</div>
              <h1 style={{ color: 'var(--color-text-primary)', marginBottom: 8 }}>404 — Lost in Space</h1>
              <p style={{ color: 'var(--color-text-muted)' }}>This page doesn't exist in our star map.</p>
            </div>
          } />
        </Routes>
      </main>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <BrowserRouter>
      <ActivityProvider>
        <AppShell />
      </ActivityProvider>
    </BrowserRouter>
  );
}

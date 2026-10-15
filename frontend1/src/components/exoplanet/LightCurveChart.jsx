import { SectionCard } from '../ui/UIComponents';
import { TrendingUp } from 'lucide-react';

const FLUX_COLUMNS_REGEX = /^flux[._\s-]?\d+$|^f\d+$|^FLUX\.\d+$/i;
export function parseCsvForChart(csvText) {
  if (!csvText) return { data: [], hasFlux: false, columnCount: 0 };
  const lines = csvText.trim().split(/\r?\n/).filter(Boolean); if (lines.length < 2) return { data: [], hasFlux: false, columnCount: 0 };
  const headers = lines[0].split(',').map(v => v.trim().replace(/^"|"$/g, ''));
  const row = lines[1].split(',').map(v => v.trim().replace(/^"|"$/g, ''));
  let columns = headers.filter(h => FLUX_COLUMNS_REGEX.test(h));
  if (columns.length < 6) columns = headers.filter(h => /^-?\d+(\.\d+)?$/.test(h));
  const data = columns.map((col, index) => ({ index, flux: Number(row[headers.indexOf(col)]) })).filter(d => Number.isFinite(d.flux));
  return { data, hasFlux: data.length > 5, columnCount: headers.length };
}
export function LightCurveChart({ chartData, label = 'Light curve — flux vs observation index' }) {
  if (!chartData?.length) return null;
  const data = chartData.length > 500 ? chartData.filter((_, i) => i % Math.ceil(chartData.length / 500) === 0) : chartData;
  const width = 900, height = 240, pad = 22; const vals = data.map(d => d.flux); const min = Math.min(...vals), max = Math.max(...vals) || min + 1; const points = data.map((d, i) => `${pad + (i / Math.max(1, data.length - 1)) * (width - pad * 2)},${height - pad - ((d.flux - min) / (max - min || 1)) * (height - pad * 2)}`).join(' ');
  return <SectionCard title="Light Curve Visualization" titleIcon={TrendingUp}><p style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 12 }}>{label}. Values are displayed locally; no scientific inference is made.</p><div style={{ overflowX: 'auto' }}><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Light curve chart" style={{ width: '100%', minWidth: 480, height: 240, background: 'rgba(3,10,22,.5)', borderRadius: 8 }}><line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="var(--color-border)"/><line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="var(--color-border)"/><polyline points={points} fill="none" stroke="var(--color-cyan)" strokeWidth="2" vectorEffect="non-scaling-stroke"/></svg></div></SectionCard>;
}

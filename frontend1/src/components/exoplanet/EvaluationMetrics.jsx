import { SectionCard, ProgressBar } from '../ui/UIComponents';
import { BarChart2, Grid } from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell
} from 'recharts';

/**
 * EvaluationMetrics component.
 * 
 * Expected shape from backend:
 * {
 *   accuracy: 0.97,
 *   precision: 0.95,
 *   recall: 0.94,
 *   f1_score: 0.945,
 *   confusion_matrix: [[tp, fp], [fn, tn]],   // or [[...], [...]] for 2-class
 *   per_class: {
 *     "1": { precision, recall, f1, support },
 *     "2": { precision, recall, f1, support }
 *   }
 * }
 */
export function EvaluationMetrics({ metrics }) {
  if (!metrics) return null;

  const { accuracy, precision, recall, f1_score, confusion_matrix, per_class } = metrics;

  const summaryMetrics = [
    { label: 'Accuracy', value: accuracy, color: 'var(--color-cyan)' },
    { label: 'Precision', value: precision, color: 'var(--color-violet)' },
    { label: 'Recall', value: recall, color: 'var(--color-green)' },
    { label: 'F1 Score', value: f1_score, color: 'var(--color-amber)' },
  ].filter(m => m.value != null);

  const barData = summaryMetrics.map(m => ({
    name: m.label,
    value: +(m.value * 100).toFixed(2),
    color: m.color,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Summary metrics */}
      <SectionCard title="Model Evaluation Metrics" titleIcon={BarChart2}>
        <p style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 16 }}>
          Metrics computed on the test split by the backend. These describe overall model performance,
          not the individual prediction above.
        </p>

        {/* Metric bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 20 }}>
          {summaryMetrics.map(m => (
            <div key={m.label}>
              <div className="flex-between" style={{ marginBottom: 5 }}>
                <span style={{ fontSize: 13, color: 'var(--color-text-secondary)', fontWeight: 500 }}>{m.label}</span>
                <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: m.color, fontWeight: 600 }}>
                  {(m.value * 100).toFixed(2)}%
                </span>
              </div>
              <div className="progress-bar-wrap">
                <div className="progress-bar" style={{ width: `${m.value * 100}%`, background: m.color }} />
              </div>
            </div>
          ))}
        </div>

        {/* Bar chart */}
        {barData.length > 0 && (
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--color-border-muted)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => `${v}%`}
                width={42}
              />
              <Tooltip
                formatter={(v) => [`${v.toFixed(2)}%`]}
                contentStyle={{
                  background: 'var(--color-bg-elevated)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 8,
                  fontSize: 12,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-text-primary)',
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {barData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </SectionCard>

      {/* Confusion Matrix */}
      {confusion_matrix && (
        <ConfusionMatrix matrix={confusion_matrix} />
      )}

      {/* Per-class metrics */}
      {per_class && (
        <PerClassMetrics perClass={per_class} />
      )}
    </div>
  );
}

function ConfusionMatrix({ matrix }) {
  if (!matrix || matrix.length === 0) return null;

  const flat = matrix.flat();
  const maxVal = Math.max(...flat);

  const classLabels = matrix.map((_, i) => `Class ${i + 1}`);

  return (
    <SectionCard title="Confusion Matrix" titleIcon={Grid}>
      <p style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 16 }}>
        Rows: Actual class. Columns: Predicted class.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'separate', borderSpacing: 3 }}>
          <thead>
            <tr>
              <th style={{ padding: '4px 12px', fontSize: 11, color: 'var(--color-text-dim)', textAlign: 'right' }}>
                Actual ↓ / Pred →
              </th>
              {classLabels.map(label => (
                <th key={label} style={{ padding: '4px 12px', fontSize: 11, color: 'var(--color-text-muted)', textAlign: 'center' }}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIdx) => (
              <tr key={rowIdx}>
                <td style={{ padding: '4px 12px', fontSize: 11, color: 'var(--color-text-muted)', textAlign: 'right', whiteSpace: 'nowrap' }}>
                  {classLabels[rowIdx]}
                </td>
                {row.map((val, colIdx) => {
                  const isDiag = rowIdx === colIdx;
                  const intensity = maxVal > 0 ? val / maxVal : 0;
                  const bg = isDiag
                    ? `rgba(0, 212, 255, ${0.1 + intensity * 0.5})`
                    : `rgba(239, 68, 68, ${intensity * 0.4})`;
                  return (
                    <td key={colIdx} style={{ padding: 4 }}>
                      <div style={{
                        minWidth: 64,
                        minHeight: 56,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: bg,
                        border: `1px solid ${isDiag ? 'rgba(0,212,255,0.25)' : 'rgba(239,68,68,0.15)'}`,
                        borderRadius: 8,
                        fontFamily: 'var(--font-mono)',
                        fontSize: 15,
                        fontWeight: 700,
                        color: isDiag ? 'var(--color-cyan)' : val > 0 ? 'var(--color-red)' : 'var(--color-text-dim)',
                      }}>
                        {val}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}

function PerClassMetrics({ perClass }) {
  if (!perClass) return null;

  const entries = Object.entries(perClass);
  if (entries.length === 0) return null;

  return (
    <SectionCard title="Per-Class Breakdown">
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([cls, vals]) => (
              <tr key={cls}>
                <td style={{ color: 'var(--color-text-primary)', fontWeight: 600 }}>
                  {cls === '2' ? `${cls} — Exoplanet` : `${cls} — Non-Exoplanet`}
                </td>
                <td style={{ color: 'var(--color-violet)' }}>{vals.precision != null ? `${(vals.precision * 100).toFixed(1)}%` : '—'}</td>
                <td style={{ color: 'var(--color-green)' }}>{vals.recall != null ? `${(vals.recall * 100).toFixed(1)}%` : '—'}</td>
                <td style={{ color: 'var(--color-amber)' }}>{vals.f1 != null ? `${(vals.f1 * 100).toFixed(1)}%` : '—'}</td>
                <td style={{ color: 'var(--color-text-muted)' }}>{vals.support ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}

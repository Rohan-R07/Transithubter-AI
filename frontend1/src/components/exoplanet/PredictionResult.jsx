import { SectionCard, ProgressBar } from '../ui/UIComponents';
import { Star, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

/**
 * Renders the prediction result from the backend.
 * 
 * Expected shape:
 * {
 *   predicted_class: 1 | 2,
 *   label: "EXOPLANET CONFIRMED" | "NON-EXOPLANET",
 *   probabilities: { 1: 0.12, 2: 0.88 } | null,
 *   confidence: 0.88 | null,
 *   model_version: "v1.0" | null,
 *   note: string | null,
 * }
 */
export function PredictionResult({ result }) {
  if (!result) return null;

  const isExoplanet = result.predicted_class === 2 || result.label?.toLowerCase().includes('exoplanet');

  const probs = result.probabilities
    ? Object.entries(result.probabilities).sort(([a], [b]) => Number(a) - Number(b))
    : null;

  return (
    <SectionCard
      title="Prediction Result"
      titleIcon={Star}
      style={{
        borderColor: isExoplanet ? 'rgba(0, 212, 255, 0.3)' : 'var(--color-border)',
        boxShadow: isExoplanet ? 'var(--shadow-cyan)' : 'none',
      }}
    >
      <div className="animate-fade-in">
        {/* Main classification badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          padding: '16px 20px',
          background: isExoplanet ? 'var(--color-cyan-muted)' : 'var(--color-bg-elevated)',
          borderRadius: 10,
          marginBottom: 20,
          border: `1px solid ${isExoplanet ? 'rgba(0,212,255,0.2)' : 'var(--color-border)'}`,
        }}>
          {isExoplanet
            ? <CheckCircle size={32} style={{ color: 'var(--color-cyan)', flexShrink: 0 }} />
            : <XCircle size={32} style={{ color: 'var(--color-text-muted)', flexShrink: 0 }} />
          }
          <div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: 20,
              fontWeight: 700,
              color: isExoplanet ? 'var(--color-cyan)' : 'var(--color-text-primary)',
              letterSpacing: '-0.01em',
            }}>
              {result.label || (isExoplanet ? 'EXOPLANET CANDIDATE' : 'NON-EXOPLANET')}
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Class {result.predicted_class} · {isExoplanet
                ? 'Transit signal detected in light curve'
                : 'No significant transit signal detected'}
            </div>
          </div>
        </div>

        {/* Important distinction */}
        {isExoplanet && (
          <div style={{
            display: 'flex',
            gap: 8,
            alignItems: 'flex-start',
            padding: '10px 14px',
            background: 'var(--color-amber-muted)',
            border: '1px solid rgba(245,158,11,0.2)',
            borderRadius: 8,
            marginBottom: 20,
          }}>
            <AlertTriangle size={14} style={{ color: 'var(--color-amber)', marginTop: 1, flexShrink: 0 }} />
            <p style={{ fontSize: 12, color: 'var(--color-amber)', lineHeight: 1.5 }}>
              This is a model classification, not confirmed discovery. Independent photometric and spectroscopic
              follow-up is required to confirm planetary status.
            </p>
          </div>
        )}

        {/* Confidence */}
        {result.confidence != null && (
          <div style={{ marginBottom: 20 }}>
            <ProgressBar
              value={result.confidence}
              max={1}
              colorClass={isExoplanet ? 'cyan' : 'violet'}
              label={`Model Confidence: ${(result.confidence * 100).toFixed(1)}%`}
              showPct={false}
            />
          </div>
        )}

        {/* Class probabilities */}
        {probs && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
              Class Probabilities
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {probs.map(([cls, prob]) => {
                const isSelected = Number(cls) === result.predicted_class;
                const label = Number(cls) === 2 ? `Class ${cls} — Exoplanet Candidate` : `Class ${cls} — Non-Exoplanet`;
                return (
                  <div key={cls}>
                    <div className="flex-between" style={{ marginBottom: 4 }}>
                      <span style={{ fontSize: 12, color: isSelected ? 'var(--color-text-primary)' : 'var(--color-text-secondary)', fontWeight: isSelected ? 600 : 400 }}>
                        {label}
                      </span>
                      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: isSelected ? 'var(--color-cyan)' : 'var(--color-text-muted)' }}>
                        {(prob * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="progress-bar-wrap">
                      <div
                        className={`progress-bar ${isSelected ? 'progress-bar-cyan' : 'progress-bar-violet'}`}
                        style={{ width: `${prob * 100}%`, opacity: isSelected ? 1 : 0.5 }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Model version */}
        {result.model_version && (
          <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--color-border-muted)' }}>
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--color-text-dim)' }}>
              Model: {result.model_version}
            </span>
          </div>
        )}

        {/* Backend note */}
        {result.note && (
          <div style={{ marginTop: 8 }}>
            <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>{result.note}</span>
          </div>
        )}
      </div>
    </SectionCard>
  );
}

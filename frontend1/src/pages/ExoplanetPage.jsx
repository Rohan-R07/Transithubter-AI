import { useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  FlaskConical,
  Info,
  Play,
  RotateCcw,
  Sparkles,
  Telescope,
  Upload,
} from 'lucide-react';
import { PageHeader } from '../components/layout/Sidebar';
import { EmptyState, ErrorState, LoadingState, SectionCard } from '../components/ui/UIComponents';
import { LightCurveChart, parseCsvForChart } from '../components/exoplanet/LightCurveChart';
import { useActivity } from '../components/activity/ActivityLog';
import * as api from '../services/api';

export default function ExoplanetPage() {
  const inputRef = useRef(null);
  const { addEvent } = useActivity();

  const [file, setFile] = useState(null);
  const [chart, setChart] = useState([]);
  const [fluxValues, setFluxValues] = useState([]);
  const [mode, setMode] = useState('balanced');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [isSample, setIsSample] = useState(false);

  // ── CSV File Selection ──────────────────────────────────────────────────
  const selectFile = (f) => {
    setError('');
    setPrediction(null);
    setIsSample(false);
    setChart([]);
    setFluxValues([]);
    setFile(null);

    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.csv')) {
      return setError('Please choose a valid .csv file containing Kepler stellar flux readings.');
    }
    if (f.size > 50 * 1024 * 1024) {
      return setError('File is too large (>50MB). Please provide a single star light curve CSV.');
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = String(e.target.result);
        const parsed = parseCsvForChart(text);

        // Also extract all raw numerical values for model inference
        const lines = text.trim().split(/\r?\n/).filter(Boolean);
        let extractedFlux = [];

        if (lines.length >= 2) {
          // Check row 1 (first star sample)
          const headers = lines[0].split(',').map((v) => v.trim().replace(/^"|"$/g, ''));
          const row = lines[1].split(',').map((v) => v.trim().replace(/^"|"$/g, ''));

          // Drop LABEL column if present
          let pairs = headers.map((h, i) => ({ header: h, val: Number(row[i]) }));
          pairs = pairs.filter((p) => p.header.toUpperCase() !== 'LABEL');
          extractedFlux = pairs.map((p) => p.val).filter(Number.isFinite);
        } else if (lines.length === 1) {
          // Single row of numbers
          extractedFlux = lines[0]
            .split(',')
            .map((v) => Number(v.trim()))
            .filter(Number.isFinite);
        }

        setFile(f);
        setChart(parsed.data);
        setFluxValues(extractedFlux);

        if (extractedFlux.length !== 3197) {
          setError(
            `Notice: Found ${extractedFlux.length} flux values. The trained Kepler model expects exactly 3,197 timestamps. (For an immediate live test, click "Use Kepler Sample").`
          );
        }

        addEvent({
          type: 'info',
          message: `CSV loaded: ${f.name} (${extractedFlux.length} timestamps)`,
          detail: 'Ready for AI classification',
        });
      } catch (err) {
        setError(`Failed to parse CSV file: ${err.message}`);
      }
    };
    reader.onerror = () => setError('The CSV could not be read in this browser.');
    reader.readAsText(f);
  };

  // ── Use Real Backend Kepler Sample ──────────────────────────────────────
  const useSample = async () => {
    setLoading(true);
    setError('');
    setPrediction(null);
    setFile(null);

    try {
      const sampleData = await api.predictFromSample(0, mode);
      setFluxValues(sampleData.flux_values || []);
      setChart(sampleData.chart_data || []);
      setIsSample(true);
      setPrediction(sampleData.prediction || null);

      addEvent({
        type: 'info',
        message: 'Loaded verified Kepler sample (Row 0)',
        detail: '3,197 real photometric flux values',
      });
    } catch (err) {
      setError(`Failed to fetch backend sample: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ── Run Live Model Inference ───────────────────────────────────────────
  const analyze = async () => {
    if (!fluxValues.length) {
      setError('Please select a CSV file or load the Kepler sample first.');
      return;
    }

    if (fluxValues.length !== 3197) {
      setError(
        `Feature count mismatch: got ${fluxValues.length} values, model expects exactly 3,197. Use the "Kepler Sample" button for verified real data.`
      );
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const res = await api.predict(fluxValues, mode);
      setPrediction(res);

      addEvent({
        type: 'predict',
        message: `Inference: ${res.predicted_label.toUpperCase()} (P=${(res.probability_exoplanet * 100).toFixed(1)}%)`,
        detail: `Mode: ${mode} · Threshold: ${res.decision_threshold}`,
      });
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ── Reset ───────────────────────────────────────────────────────────────
  const reset = () => {
    setFile(null);
    setChart([]);
    setFluxValues([]);
    setIsSample(false);
    setError('');
    setPrediction(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="page-container animate-fade-in">
      <PageHeader
        title="Exoplanet Hunter"
        subtitle="Stellar photometric transit detection powered by TransitHunter AI Phase 1.5 classifier"
        actions={
          <button className="btn btn-secondary btn-sm" onClick={reset}>
            <RotateCcw size={14} /> Reset
          </button>
        }
      />

      <div style={{ padding: '24px 0', display: 'grid', gap: 20 }}>
        {/* Input & Parameters Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.2fr) minmax(280px,.8fr)', gap: 20 }}>
          {/* File Upload Zone */}
          <SectionCard title="Light-Curve Ingestion" titleIcon={Upload}>
            <input
              ref={inputRef}
              id="csv-upload"
              type="file"
              accept=".csv,text/csv"
              hidden
              onChange={(e) => selectFile(e.target.files?.[0])}
            />

            <div
              className="file-upload-zone"
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                selectFile(e.dataTransfer.files?.[0]);
              }}
            >
              <Upload size={30} style={{ color: 'var(--color-cyan)' }} />
              <div style={{ fontWeight: 650, marginTop: 8 }}>Choose or drop a Kepler CSV</div>
              <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 4 }}>
                Accepts continuous flux time series (FLUX.1 … FLUX.3197)
              </div>
            </div>

            {file && (
              <div style={{ marginTop: 12, color: 'var(--color-green)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                <FileText size={15} /> {file.name} · {(file.size / 1024).toFixed(1)} KB ({fluxValues.length} flux points)
              </div>
            )}
            {isSample && (
              <div style={{ marginTop: 12, color: 'var(--color-cyan)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                <CheckCircle2 size={15} /> Loaded NASA Kepler exoTest.csv sample (3,197 real flux timestamps)
              </div>
            )}

            {/* Actions */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 16 }}>
              <button className="btn btn-secondary btn-sm" onClick={useSample} disabled={loading}>
                <FlaskConical size={14} /> Use Kepler Sample
              </button>
              <button
                className="btn btn-primary btn-sm"
                disabled={!fluxValues.length || loading}
                onClick={analyze}
              >
                <Play size={14} /> {loading ? 'Running Inference…' : 'Classify with AI'}
              </button>
            </div>
          </SectionCard>

          {/* Model Decision Configuration */}
          <SectionCard title="Classifier Operating Mode" titleIcon={Telescope}>
            <p style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 14 }}>
              Select the calibrated decision threshold applied after uniform moving-average detrending (window=31) and robust 1.4826×MAD normalization:
            </p>

            <div style={{ display: 'grid', gap: 10 }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 10,
                  padding: '10px 12px',
                  borderRadius: 8,
                  border: mode === 'balanced' ? '1px solid var(--color-cyan)' : '1px solid var(--color-border-muted)',
                  background: mode === 'balanced' ? 'rgba(0,212,255,.07)' : 'transparent',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="radio"
                  name="threshold-mode"
                  value="balanced"
                  checked={mode === 'balanced'}
                  onChange={() => setMode('balanced')}
                  style={{ marginTop: 3 }}
                />
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)' }}>
                    Balanced Mode (T = 0.08) <span style={{ color: 'var(--color-green)', fontSize: 11 }}>Recommended</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 2 }}>
                    CV-selected operating point. 70.3% CV recall with low false positive rate (0.47%).
                  </div>
                </div>
              </label>

              <label
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 10,
                  padding: '10px 12px',
                  borderRadius: 8,
                  border: mode === 'high_recall' ? '1px solid var(--color-violet)' : '1px solid var(--color-border-muted)',
                  background: mode === 'high_recall' ? 'rgba(139,92,246,.07)' : 'transparent',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="radio"
                  name="threshold-mode"
                  value="high_recall"
                  checked={mode === 'high_recall'}
                  onChange={() => setMode('high_recall')}
                  style={{ marginTop: 3 }}
                />
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)' }}>
                    High-Recall Mode (T = 0.05) <span style={{ color: 'var(--color-amber)', fontSize: 11 }}>Sensitive</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginTop: 2 }}>
                    Catches subtle, faint transit dips at the expense of more false positives (~1.9%).
                  </div>
                </div>
              </label>
            </div>
          </SectionCard>
        </div>

        {/* Error notification */}
        {error && <ErrorState message={error} />}

        {/* Loading state */}
        {loading && <LoadingState message="Preprocessing flux & executing Random Forest inference…" />}

        {/* Light curve visualization */}
        {!loading && chart.length > 0 && (
          <LightCurveChart
            chartData={chart}
            label={isSample ? 'Kepler Real Photometric Flux Time Series (Test Star 0)' : 'Normalized Stellar Flux Profile'}
          />
        )}

        {/* Real Live Prediction Results Card */}
        {!loading && prediction && (
          <SectionCard
            title="Classification Result (FastAPI Engine)"
            titleIcon={Sparkles}
            headerRight={
              <span style={{ fontSize: 12, color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                Decision Threshold T={prediction.decision_threshold}
              </span>
            }
          >
            <div
              style={{
                padding: 18,
                borderRadius: 10,
                border: prediction.is_exoplanet_candidate
                  ? '1px solid rgba(16,185,129,.4)'
                  : '1px solid rgba(59,130,246,.3)',
                background: prediction.is_exoplanet_candidate
                  ? 'rgba(16,185,129,.07)'
                  : 'rgba(59,130,246,.05)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ fontSize: 28 }}>
                    {prediction.is_exoplanet_candidate ? '🪐' : '⭐'}
                  </div>
                  <div>
                    <div
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: prediction.is_exoplanet_candidate
                          ? 'var(--color-green)'
                          : 'var(--color-text-primary)',
                      }}
                    >
                      {prediction.is_exoplanet_candidate
                        ? 'EXOPLANET CANDIDATE DETECTED'
                        : 'NO TRANSIT SIGNAL DETECTED'}
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--color-text-muted)', marginTop: 2 }}>
                      Class {prediction.predicted_class} · {prediction.predicted_label.toUpperCase()} · Mode:{' '}
                      <code>{prediction.mode}</code>
                    </div>
                  </div>
                </div>

                {/* Probability meters */}
                <div style={{ minWidth: 220, textAlign: 'right' }}>
                  <div style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                    Transit Probability:
                  </div>
                  <div
                    style={{
                      fontSize: 22,
                      fontWeight: 700,
                      color: prediction.is_exoplanet_candidate ? 'var(--color-green)' : 'var(--color-cyan)',
                    }}
                  >
                    {(prediction.probability_exoplanet * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* Progress bar */}
              <div style={{ marginTop: 16 }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: 11,
                    color: 'var(--color-text-muted)',
                    marginBottom: 4,
                  }}
                >
                  <span>Non-Exoplanet: {(prediction.probability_non_exoplanet * 100).toFixed(1)}%</span>
                  <span>Exoplanet: {(prediction.probability_exoplanet * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: 8, background: 'rgba(255,255,255,.08)', borderRadius: 4, overflow: 'hidden', display: 'flex' }}>
                  <div
                    style={{
                      width: `${prediction.probability_non_exoplanet * 100}%`,
                      background: 'rgba(59,130,246,.7)',
                      height: '100%',
                    }}
                  />
                  <div
                    style={{
                      width: `${prediction.probability_exoplanet * 100}%`,
                      background: prediction.is_exoplanet_candidate ? 'var(--color-green)' : 'var(--color-cyan)',
                      height: '100%',
                    }}
                  />
                </div>
              </div>

              {/* Physical AI Explanation */}
              <div
                style={{
                  marginTop: 16,
                  padding: 12,
                  borderRadius: 8,
                  background: 'var(--color-bg-primary)',
                  border: '1px solid var(--color-border-muted)',
                  fontSize: 13,
                  color: 'var(--color-text-secondary)',
                  lineHeight: 1.6,
                }}
              >
                <strong>Scientific Analysis:</strong> {prediction.explanation}
              </div>

              {/* Disclaimer */}
              <div
                style={{
                  marginTop: 10,
                  fontSize: 11,
                  color: 'var(--color-text-dim)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <Info size={12} /> {prediction.disclaimer}
              </div>
            </div>
          </SectionCard>
        )}

        {/* Empty state when nothing selected */}
        {!file && !isSample && !error && !loading && (
          <EmptyState
            icon={Telescope}
            title="No light curve loaded"
            description="Upload a CSV with 3,197 Kepler flux points or click 'Use Kepler Sample' to test real space data."
          />
        )}
      </div>
    </div>
  );
}

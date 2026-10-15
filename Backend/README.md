# TransitHunter AI — Backend

## Overview
TransitHunter AI is an astronomical machine learning pipeline designed to detect exoplanetary transit signals from high-dimensional photometric flux time series recorded by NASA's Kepler mission.

This backend provides:
- **Phase 1 & 1.5**: ML training pipeline, diagnosis, and cross-validated model improvement
- **Phase 2**: FastAPI REST API for real-time exoplanet candidate classification

> [!IMPORTANT]
> 🛰️ **Dataset Source:** **[NASA Kepler Labelled Time Series Dataset (Kaggle)](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)** — 5,657 light curves, 3,197 flux readings per sample (`exoTrain.csv` and `exoTest.csv`).

---

## Directory Architecture

```
Backend/
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI entry point (lifespan, CORS, error handlers)
│   ├── schemas.py                     # Pydantic request/response models
│   ├── core/
│   │   ├── __init__.py
│   │   └── model_loader.py           # Singleton model loader (loads Phase 1.5 at startup)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py                  # GET /health
│   │   ├── model.py                   # GET /model/metadata
│   │   ├── network.py                 # Network topology, route, interference, restore
│   │   └── predict.py                 # POST /predict
│   └── services/
│       ├── __init__.py
│       ├── dataset_service.py         # Inspection, validation, detrending & MAD normalization
│       ├── network_service.py         # Space network topology, Dijkstra algorithm, interference
│       └── training_service.py        # RF training, threshold trade-offs, metrics, persistence
├── models/
│   ├── exoplanet_detector_phase1_5.joblib     # Phase 1.5 improved detector (used by API)
│   ├── feature_metadata_phase1_5.json         # Phase 1.5 metadata + calibrated threshold
│   ├── exoplanet_random_forest.joblib         # Phase 1 baseline RF model
│   ├── feature_metadata.json                  # Phase 1 baseline metadata
│   ├── baseline_exoplanet_random_forest.joblib
│   └── baseline_feature_metadata.json
├── reports/
│   ├── model_evaluation_phase1_5.json         # Phase 1.5 test evaluation & uncertainty
│   ├── cross_validation_comparison.json       # 5-fold CV comparison across candidates
│   ├── diagnosis_report.json                  # Baseline failure root-cause analysis
│   ├── model_evaluation.json                  # Phase 1 baseline test evaluation
│   ├── model_evaluation_baseline.json
│   └── dataset_inspection.json
├── scripts/
│   ├── __init__.py
│   ├── generate_predict_request.py   # Extracts 3,197 flux features from exoTest.csv
│   └── verify_live_api.py            # Automated live HTTP endpoint test suite (22 scenarios)
├── predict_request.json               # Sample prediction request from real dataset
├── tests/
│   ├── __init__.py
│   ├── test_api.py                    # 17 endpoint and unit tests
│   ├── test_generate_request.py       # 4 dataset extractor utility tests
│   └── test_network.py                # 16 network topology & Dijkstra routing tests
├── train_model.py                     # Phase 1 training pipeline
├── diagnose_and_train.py              # Phase 1.5 diagnosis + enhanced training
├── verify_model.py                    # Automated artifact verification
└── requirements.txt                   # Python dependencies
```

---

## Quick Start — API Server

### Prerequisites

- Python 3.12+
- Dataset files (`exoTrain.csv`, `exoTest.csv`) in the workspace root (only needed for training, **not** for inference)

### Install Dependencies

```bash
cd Backend
pip install -r requirements.txt
```

### Start the API

```bash
cd Backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The model loads automatically at startup. Visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

---

## API Endpoints

### `GET /health`

Returns API status and whether the model is loaded.

**Response** (200):
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "exoplanet_detector_phase1_5",
  "model_phase": "Phase 1.5 - Diagnosed and Improved Detector",
  "load_error": null
}
```

---

### `GET /model/metadata`

Returns model configuration, preprocessing pipeline, thresholds, and evaluation summary from stored artifacts. No metrics are fabricated.

**Response** (200):
```json
{
  "model_name": "exoplanet_detector_phase1_5",
  "phase": "Phase 1.5 - Diagnosed and Improved Detector",
  "model_type": "RandomForestClassifier",
  "library": "scikit-learn",
  "library_version": "1.9.0",
  "feature_count": 3197,
  "expected_input": "3197 numerical flux values (FLUX.1 … FLUX.3197)",
  "label_mapping": { "1": "non_exoplanet", "2": "exoplanet" },
  "classes": [1, 2],
  "default_threshold": 0.08,
  "threshold_options": {
    "balanced_recommended": 0.08,
    "high_recall_sensitive": 0.05
  },
  "preprocessing": {
    "pipeline": "detrend_and_normalize_lightcurves",
    "algorithm": "uniform_filter1d continuum subtraction + robust 1.4826*MAD standardization",
    "smooth_window": 31,
    "feature_dtype": "float64"
  },
  "created_at": "2026-10-15T07:19:47.267636+00:00",
  "evaluation_summary": { "..." }
}
```

---

### `POST /predict`

Classifies a single stellar light curve as an exoplanet candidate or non-exoplanet.

**Request**:
```json
{
  "flux_values": [12.3, -45.6, 78.9, "... (3,197 total)"],
  "mode": "balanced"
}
```

| Parameter | Type | Required | Description |
|:---|:---|:---:|:---|
| `flux_values` | `float[]` | Yes | Exactly 3,197 photometric flux values in column order (FLUX.1 … FLUX.3197) |
| `mode` | `string` | No | `"balanced"` (default, T=0.08) or `"high_recall"` (T=0.05, catches more candidates but higher false-positive rate) |

**Response** (200):
```json
{
  "predicted_class": 1,
  "predicted_label": "non_exoplanet",
  "is_exoplanet_candidate": false,
  "probability_exoplanet": 0.07,
  "probability_non_exoplanet": 0.93,
  "decision_threshold": 0.08,
  "mode": "balanced",
  "explanation": "No significant transit signal was detected. The model assigned P(exoplanet) = 0.0700, which is below the balanced (T=0.08) decision threshold of 0.08. The light curve is consistent with a non-transiting star.",
  "disclaimer": "A positive prediction reflects statistical similarity to confirmed Kepler transit flux patterns. It does not constitute physical proof that an exoplanet exists."
}
```

**Preprocessing applied**: The API applies the exact same preprocessing as the training pipeline — uniform moving-average detrending (window=31) + robust 1.4826×MAD normalization — before feeding the light curve to the model.

---

### Error Responses

| Status | Condition | Example |
|:---:|:---|:---|
| **422** | Wrong feature count | `"expected 3197 flux values, received 100"` |
| **422** | NaN or Inf in flux values | `"non-finite value(s) (NaN or Inf)"` |
| **422** | Non-numeric input | Pydantic validation error |
| **422** | Invalid mode | `"value is not a valid enumeration member"` |
| **503** | Model not loaded | `"Model is not available for inference"` |

No filesystem paths, stack traces, or internal details are exposed in error responses.

---

## Testing `POST /predict` with Real Kepler Light-Curve Data

A utility script `scripts/generate_predict_request.py` is provided to extract an authentic light curve from `exoTest.csv`, validate the 3,197 numeric flux features, and create a ready-to-use JSON request file at `predict_request.json`.

### 1. Generate the Request JSON

From the `backend/` directory:

```bash
# Generate request for row 0 (default: balanced mode, 3,197 flux values)
python scripts/generate_predict_request.py

# Optional: select a specific row by index
python scripts/generate_predict_request.py --row-index 1

# Optional: select an exoplanet candidate sample (LABEL=2)
python scripts/generate_predict_request.py --label 2

# Optional: set mode to 'high_recall'
python scripts/generate_predict_request.py --mode high_recall
```

The script will:
- Locate `exoTest.csv` in the workspace root or `backend/data/` without copying or modifying it.
- Strip the `LABEL` column.
- Validate that exactly **3,197 numeric, finite flux values** are present.
- Write the formatted payload to `predict_request.json`.

### 2. Test in Swagger UI (Interactive Docs)

1. Ensure the FastAPI server is running (`uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`).
2. Open your browser to **http://127.0.0.1:8000/docs**.
3. Locate and click to expand **`POST /predict`**.
4. Click the **Try it out** button in the top right of the endpoint box.
5. Open `predict_request.json` in your editor, copy the entire JSON content, and paste it into the **Request body** text area.
6. Click the large blue **Execute** button.
7. Scroll down to **Responses** to view the `200 OK` prediction result, probability scores, decision threshold, and human-readable explanation.

### 3. Test from the Terminal

```bash
# Using Python (cross-platform):
python -c "import urllib.request; req = urllib.request.Request('http://127.0.0.1:8000/predict', data=open('predict_request.json','rb').read(), headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req).read().decode())"
```

---

## Phase 3: Communication Network Simulation & Dynamic Dijkstra Routing

TransitHunter AI includes a space communication network simulation representing telemetry and data transmission links between Earth and deep-space observation platforms.

### 1. Simulated Space Network Topology

The network models 5 space communications nodes and 6 bidirectional links:

| Node | ID | Type | Description |
|:---|:---|:---|:---|
| **Earth Ground Station** | `earth_ground_station` | Ground Station | Canberra Deep Space Communication Complex |
| **Orbital Relay A** | `orbital_relay_a` | Orbital Relay | Primary GEO optical/RF transponder (~35,786 km) |
| **Orbital Relay B** | `orbital_relay_b` | Orbital Relay | Secondary GEO optical backup relay (~35,786 km) |
| **Deep Space Relay** | `deep_space_relay` | Deep Space Relay | Sun-Earth L1 communications bridge (~1.5M km) |
| **Exoplanet Observatory** | `exoplanet_observatory` | Observatory | Earth-trailing space telescope photometer |

**Links & Latencies (Edge Weights):**
- `link_earth_relay_a`: Earth Ground Station ↔ Orbital Relay A (**25.0 ms**, 2,000 km)
- `link_earth_relay_b`: Earth Ground Station ↔ Orbital Relay B (**35.0 ms**, 3,500 km)
- `link_relay_a_relay_b`: Orbital Relay A ↔ Orbital Relay B (**15.0 ms**, 5,000 km crosslink)
- `link_relay_a_deep_space`: Orbital Relay A ↔ Deep Space Relay (**120.0 ms**, 500,000 km)
- `link_relay_b_deep_space`: Orbital Relay B ↔ Deep Space Relay (**140.0 ms**, 600,000 km)
- `link_deep_space_observatory`: Deep Space Relay ↔ Exoplanet Observatory (**200.0 ms**, 1,500,000 km)

### 2. Network Endpoints

#### `GET /network/topology`
Returns the active state of all nodes and communication links.

```bash
curl http://127.0.0.1:8000/network/topology
```

#### `POST /network/route`
Calculates the optimal communication path using Dijkstra's shortest-path algorithm with one-way latency (ms) as the edge weight. Only operational, non-interfered links are traversed. Accepts node names or IDs.

```bash
curl -X POST http://127.0.0.1:8000/network/route \
  -H "Content-Type: application/json" \
  -d '{"source": "Earth Ground Station", "destination": "Exoplanet Observatory"}'
```

**Response (200 OK — Optimal Path):**
```json
{
  "reachable": true,
  "source": "Earth Ground Station",
  "destination": "Exoplanet Observatory",
  "source_id": "earth_ground_station",
  "destination_id": "exoplanet_observatory",
  "path": [
    "Earth Ground Station",
    "Orbital Relay A",
    "Deep Space Relay",
    "Exoplanet Observatory"
  ],
  "path_node_ids": [
    "earth_ground_station",
    "orbital_relay_a",
    "deep_space_relay",
    "exoplanet_observatory"
  ],
  "links": [...],
  "total_latency_ms": 345.0,
  "total_distance_km": 2002000.0,
  "hops": 3,
  "message": "Optimal route found (3 hops, 345.0 ms total latency): Earth Ground Station -> Orbital Relay A -> Deep Space Relay -> Exoplanet Observatory"
}
```

#### `POST /network/interference`
Simulates space signal disruption (e.g. solar flare, space debris) that takes a link offline (`available = false`). Subsequent route queries will dynamically route around this link.

```bash
# Disable by link ID
curl -X POST http://127.0.0.1:8000/network/interference \
  -H "Content-Type: application/json" \
  -d '{"link_id": "link_earth_relay_a", "reason": "Solar flare radiation burst"}'

# Or disable by endpoints
curl -X POST http://127.0.0.1:8000/network/interference \
  -H "Content-Type: application/json" \
  -d '{"source": "Earth Ground Station", "destination": "Orbital Relay A"}'
```

**Dynamic Rerouting Behavior:**
Once `link_earth_relay_a` is disabled, querying `/network/route` dynamically routes via `Orbital Relay B`:
`Earth Ground Station -> Orbital Relay B -> Orbital Relay A -> Deep Space Relay -> Exoplanet Observatory` (**370.0 ms**) without hardcoding.

#### `POST /network/restore`
Restores a previously disabled communication link back to operational status (`available = true`), or restores all links.

```bash
# Restore specific link
curl -X POST http://127.0.0.1:8000/network/restore \
  -H "Content-Type: application/json" \
  -d '{"link_id": "link_earth_relay_a"}'

# Restore all links across the entire network
curl -X POST http://127.0.0.1:8000/network/restore \
  -H "Content-Type: application/json" \
  -d '{"restore_all": true}'
```

#### `POST /network/reset`
Resets the network topology back to the default initial operational state.

```bash
curl -X POST http://127.0.0.1:8000/network/reset
```

#### Handling Unreachable Destinations
If all paths to a destination are offline (e.g. `link_deep_space_observatory` is disabled), Dijkstra determines that no path exists and returns:

```json
{
  "reachable": false,
  "source": "Earth Ground Station",
  "destination": "Exoplanet Observatory",
  "source_id": "earth_ground_station",
  "destination_id": "exoplanet_observatory",
  "path": [],
  "path_node_ids": [],
  "links": [],
  "total_latency_ms": null,
  "total_distance_km": null,
  "hops": 0,
  "message": "Destination 'Exoplanet Observatory' is unreachable from 'Earth Ground Station'. All available communication paths are disrupted or offline."
}
```

---

## Frontend Integration

From a React/Vite frontend (default `http://localhost:5173`), CORS is pre-configured. Example:

```javascript
// Health check
const health = await fetch('http://localhost:8000/health');

// Prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    flux_values: fluxArray,  // exactly 3197 numbers
    mode: 'balanced'         // or 'high_recall'
  })
});
const result = await response.json();
if (result.is_exoplanet_candidate) {
  // Candidate detected!
}
```

---

## Testing

Run all 37 automated tests across ML inference, request generation, and network routing:

```bash
cd Backend
python -m pytest tests/ -v
```

### Test Coverage (37/37 PASSED)

| Suite | # | Test | Validates |
|:---|:---:|:---|:---|
| **API & Inference** | 1 | `test_health_returns_ok` | Health endpoint returns 200 with correct schema |
| | 2 | `test_health_model_loaded` | Model loaded status and name |
| | 3 | `test_model_metadata_schema` | All required metadata keys present |
| | 4 | `test_model_metadata_threshold_options` | Threshold values match metadata |
| | 5 | `test_model_metadata_preprocessing` | Smooth window and algorithm description |
| | 6 | `test_predict_valid_balanced` | Full prediction response in balanced mode |
| | 7 | `test_predict_valid_high_recall` | High-recall mode uses T=0.05 |
| | 8 | `test_predict_too_few_features` | 422 on wrong count (100 features) |
| | 9 | `test_predict_too_many_features` | 422 on wrong count (3297 features) |
| | 10 | `test_predict_non_numeric_input` | 422 on string values in flux_values |
| | 11 | `test_predict_nan_value` | 422 on NaN in input |
| | 12 | `test_predict_inf_value` | 422 on Infinity in input |
| | 13 | `test_predict_model_not_loaded` | 503 when model unavailable (mocked) |
| | 14 | `test_preprocessing_shape_and_finite` | Detrending preserves shape, no NaN/Inf |
| | 15 | `test_preprocessing_reduces_baseline` | Drift removed, medians near zero |
| | 16 | `test_class_label_mapping` | Class 1→non_exoplanet, 2→exoplanet |
| | 17 | `test_predict_consistency_class_label` | is_exoplanet_candidate consistent with predicted_class |
| **Request Generator** | 18 | `test_find_dataset` | Automatic discovery of exoTest.csv in workspace root |
| | 19 | `test_extract_flux_sample` | Extract 3,197 finite floats, strips LABEL |
| | 20 | `test_build_predict_request` | Schema validation of generated request dict |
| | 21 | `test_generate_predict_request_file` | End-to-end file generation & JSON output |
| **Network & Dijkstra** | 22 | `test_topology_returns_all_nodes_and_links` | All 5 nodes & 6 links returned |
| | 23 | `test_initial_optimal_route` | Earth → Observatory optimal route (345 ms) |
| | 24 | `test_route_using_node_ids` | Route query accepts node IDs and names |
| | 25 | `test_same_source_and_destination` | Same node yields 0 hops, 0 ms latency |
| | 26 | `test_dynamic_rerouting_after_interference` | Dynamic Dijkstra reroute around disabled link |
| | 27 | `test_interference_by_endpoint_names` | Disable link using source & destination names |
| | 28 | `test_link_restoration_reverts_to_primary_path` | Restoring link immediately restores optimal path |
| | 29 | `test_restore_all_links` | `restore_all=true` restores entire topology |
| | 30 | `test_unreachable_destination` | Clear unreachable status when paths severed |
| | 31 | `test_unknown_source_node` | 404 with helpful error for unrecognized source |
| | 32 | `test_unknown_destination_node` | 404 with helpful error for unrecognized destination |
| | 33 | `test_empty_node_names` | 422 on empty string node parameters |
| | 34 | `test_unknown_link_id_interference` | 404 on non-existent link ID |
| | 35 | `test_interference_missing_identifiers` | 422 when no link identifiers provided |
| | 36 | `test_interference_no_direct_link` | 404 when no direct link exists between endpoints |
| | 37 | `test_reset_network_endpoint` | POST /network/reset restores factory topology |

---

### Live API Verification Suite (`scripts/verify_live_api.py`)

To verify the running FastAPI server directly over HTTP across all 8 endpoints and 22 live scenarios:

1. Ensure the API server is running:
   ```bash
   cd Backend
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

2. Run the live verification suite:
   ```bash
   cd Backend
   python scripts/verify_live_api.py
   ```

**Options & Parameters:**
```bash
# Custom base URL (e.g. running on another port or remote host)
python scripts/verify_live_api.py --base-url http://127.0.0.1:8000

# Custom request timeout in seconds (default: 10.0s)
python scripts/verify_live_api.py --timeout 15.0

# Explicit path to predict_request.json
python scripts/verify_live_api.py --request-json predict_request.json

# Verbose output
python scripts/verify_live_api.py --verbose
```

**What the script verifies live:**
- **System & Model**: `GET /health` (model loaded, status ok) and `GET /model/metadata` (feature count 3197, classes `[1, 2]`, calibrated thresholds, window 31).
- **ML Inference**: `POST /predict` using real Kepler data (`predict_request.json`) in both `balanced` and `high_recall` modes; verifies probability distributions and explanation fields.
- **Network Topology**: `GET /network/topology` for all 5 space stations and 6 links.
- **Dynamic Dijkstra Routing**: `POST /network/route` initial optimal path (345.0 ms), same-node zero-latency path, and lookup by node ID.
- **Interference & Rerouting**: `POST /network/interference` disabling `link_earth_relay_a` and confirming Dijkstra recalculates around the failure via Orbital Relay B (**370.0 ms**).
- **Link Restoration**: `POST /network/restore` confirming route immediately reverts to the primary optimal path (345.0 ms) and `restore_all=true` restores the full network.
- **Severed Destinations**: Confirms `POST /network/route` returns `reachable: false` with zero hops and diagnostic explanations when all paths are down.
- **Input Validation & 4xx Handling**: Rejects malformed JSON, NaN/Infinity, non-numeric strings, wrong feature counts, unknown node names, unknown link IDs, and disconnected node pairs with proper `404`/`422` statuses instead of `500` errors.
- **Fault-Tolerant Cleanup**: Guarantees `POST /network/reset` is invoked in a `finally` block to restore network state even if assertions fail.

---

## Phase 1.5: Diagnosis & Improved Exoplanet Detection

### 1. Diagnosis of Baseline Failure
In Phase 1, the baseline model achieved 99.12% accuracy on `exoTest.csv`, but **0% recall** on confirmed exoplanets:
- **Predicted probabilities for test exoplanets**: Max $P(\text{Class } 2)$ for the 5 test exoplanets was only **$0.0600$ (6%)**, with a median of **$0.0500$**.
- **Root Cause**:
  1. *Unconstrained Tree Memorization*: With 3,197 raw continuous flux columns and default tree depth, decision trees split on arbitrary timestamp columns ($FLUX.k$) to isolate the 37 positive training instances ($P=1.00$ on training set), rather than learning general transit morphology.
  2. *Lack of Continuum Normalization*: Spacecraft thermal drift, pointing jitter, and stellar rotation create baseline shifts that dwarf transit signals (~0.1% to 1% dips).
  3. *Ensemble Vote Dilution*: With only 0.73% positive examples in training, default 0.50 thresholding guarantees zero positive classifications.

---

### 2. Stratified Cross-Validation on `exoTrain.csv` ONLY
To prevent data leakage, candidate exploration and decision threshold calibration were performed strictly using **5-fold Stratified Cross-Validation on `exoTrain.csv`**:

| Candidate Strategy | Preprocessing | Best CV Threshold | CV Recall (Class 2) | CV Precision (Class 2) | CV F1-Score | CV False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1: Baseline RF** | Raw continuous flux | $0.10$ | $21.62\%$ ($8/37$) | $10.39\%$ | $0.1404$ | $69/5050$ |
| **Candidate 2: Normalized RF** | Per-star mean/std centering | $0.12$ | $21.62\%$ ($8/37$) | $14.04\%$ | $0.1702$ | $49/5050$ |
| **Candidate 3: Detrended Window 31 RF** | Uniform filter (w=31) + Robust MAD | **$0.12$**<br>*(or $0.08$)* | **$48.65\%$** ($18/37$)<br>*(or $70.3\%$)* | **$100.00\%$**<br>*(or $52.0\%$)* | **$0.6545$**<br>*(or $0.5977$)* | **$0/5050$**<br>*(or $24/5050$)* |
| **Candidate 4: Detrended Window 51 RF** | Uniform filter (w=51) + Robust MAD | $0.10$ | $51.35\%$ ($19/37$) | $70.37\%$ | $0.5938$ | $8/5050$ |

**Selection Decision**: Candidate 3 (Detrended Window 31 + Robust MAD normalization) was selected based on training-only CV evidence. It eliminates baseline drift and achieved an out-of-fold F1-score of **$0.6545$** with zero false positives at $T=0.12$, and high recall ($70.3\%$) with low false-positive rate ($0.47\%$) at $T=0.08$.

---

### 3. Final Test Set Evaluation (`exoTest.csv`)

The selected model was evaluated **once** on the untouched `exoTest.csv`:

#### Performance Comparison: Baseline vs. Phase 1.5 Improved Detector
| Metric | Baseline Model ($T=0.50$) | Phase 1.5 ($T=0.50$) | Phase 1.5 Selected ($T=0.08$) | Phase 1.5 Sensitive ($T=0.05$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | $99.12\%$ | $99.12\%$ | **$98.95\%$** | $97.72\%$ |
| **Exoplanets Detected (TP)** | **$0/5$** | **$0/5$** | **$1/5$** | **$3/5$** |
| **Exoplanet Recall** | **$0.00\%$** | **$0.00\%$** | **$20.00\%$** | **$60.00\%$** |
| **Exoplanet Precision** | $0.00\%$ | $0.00\%$ | **$33.33\%$** | $21.43\%$ |
| **Exoplanet F1-Score** | $0.0000$ | $0.0000$ | **$0.2500$** | **$0.3158$** |
| **False Positives (FP)** | **$0/565$** | **$0/565$** | **$2/565$** ($0.35\%$) | **$11/565$** ($1.9\%$) |
| **True Negatives (TN)** | $565/565$ | $565/565$ | $563/565$ | $554/565$ |

---

## Commands Reference

```bash
# Start the API server
cd Backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v

# Run Phase 1.5 training pipeline
python diagnose_and_train.py

# Run Phase 1 baseline training
python train_model.py

# Verify all model artifacts
python verify_model.py
```

---

## Domain Interpretation Disclaimer
TransitHunter AI classifies photometric flux patterns matching Kepler mission training labels. Predictions indicate statistical alignment with known planetary transit signatures and do not constitute physical astronomical proof of an exoplanet.

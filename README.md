# 🌌 TransitHunter AI

> **Exoplanet Transit Detection with Resilient Deep-Space Communication Routing**

TransitHunter AI is an end-to-end space exploration and mission operations platform. It combines a machine learning pipeline trained on NASA Kepler photometric time-series flux data with a high-performance FastAPI backend and a simulated deep-space communication network featuring dynamic Dijkstra routing.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Pytest-37%20Passed-brightgreen?logo=pytest)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🏛️ System Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │          NASA Kepler Light Curve Data         │
                    │   3,197 Photometric Timestamps (exoTest.csv)  │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      Signal Preprocessing & Detrending        │
                    │   - Moving-Average Filter (Window = 31)       │
                    │   - Stellar Baseline Drift Removal            │
                    │   - Robust Standardization (1.4826 x MAD)     │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │       Phase 1.5 ML Transit Classifier         │
                    │   Random Forest (100 Trees, Calibrated T=0.08)│
                    │   P(Exoplanet) vs P(Non-Exoplanet)            │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │           FastAPI Mission REST Engine         │
                    │   POST /predict  |  GET /model/metadata       │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │    Deep-Space Resilient Relay Network (Mesh)  │
                    │   Dijkstra Shortest-Path Latency Routing      │
                    │   Dynamic Rerouting on Signal Disruption      │
                    └───────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
TransitHunter-AI/
├── Backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                        # FastAPI entry point, lifespan, CORS & error handlers
│   │   ├── schemas.py                     # Strict Pydantic models for ML & Network simulation
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── model_loader.py           # Thread-safe singleton model loader
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── health.py                  # GET /health
│   │   │   ├── model.py                   # GET /model/metadata
│   │   │   ├── predict.py                 # POST /predict (balanced & high_recall modes)
│   │   │   └── network.py                 # Topology, Dijkstra routing, interference & restoration
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── dataset_service.py         # Detrending (w=31) & robust 1.4826*MAD normalization
│   │       ├── network_service.py         # Simulated space topology & Dijkstra routing engine
│   │       └── training_service.py        # RF training, threshold trade-offs & evaluation
│   ├── models/
│   │   ├── exoplanet_detector_phase1_5.joblib     # Calibrated production model (used by API)
│   │   ├── feature_metadata_phase1_5.json         # Phase 1.5 schema & decision thresholds
│   │   ├── exoplanet_random_forest.joblib         # Phase 1 baseline model
│   │   └── feature_metadata.json                  # Phase 1 baseline metadata
│   ├── reports/
│   │   ├── dataset_inspection.json                # Kepler dataset validation report
│   │   ├── model_evaluation_phase1_5.json         # Phase 1.5 test metrics & uncertainty
│   │   ├── cross_validation_comparison.json       # 5-fold CV benchmark across candidate models
│   │   ├── diagnosis_report.json                  # Root-cause analysis of baseline failure
│   │   └── model_evaluation.json                  # Phase 1 baseline evaluation
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── generate_predict_request.py   # Extracts 3,197 flux features from exoTest.csv
│   │   └── verify_live_api.py            # Automated live HTTP endpoint test suite (22 scenarios)
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_api.py                    # 17 ML & inference unit/integration tests
│   │   ├── test_generate_request.py       # 4 dataset extractor utility tests
│   │   └── test_network.py                # 16 network topology & Dijkstra routing tests
│   ├── predict_request.json               # Sample real Kepler test payload (3,197 flux values)
│   ├── train_model.py                     # Phase 1 baseline training pipeline
│   ├── diagnose_and_train.py              # Phase 1.5 diagnosis + enhanced training pipeline
│   ├── verify_model.py                    # Artifact integrity verification
│   ├── requirements.txt                   # Production backend dependencies
│   └── README.md                          # Backend detailed technical reference
├── frontend1/                             # Modern Mission Control Web Application
│   ├── src/
│   │   ├── components/                    # UI cards, status badges, SVG network graph, activity logs
│   │   ├── pages/                         # Mission Control Dashboard, Exoplanet Hunter, Space Mesh
│   │   ├── services/api.js                # Live REST integration with FastAPI backend (port 8000)
│   │   ├── styles/                        # Dark-mode space aesthetics & custom styles
│   │   ├── App.jsx                        # React Router routes and layout
│   │   └── main.jsx                       # React 19 application entry point
│   ├── package.json                       # Vite, React 19, Recharts, Lucide-React
│   ├── vite.config.js                     # Vite build configuration
│   └── .env                               # VITE_API_BASE_URL=http://127.0.0.1:8000
├── .gitignore                             # Git ignore rules (datasets, caches, binaries, node_modules)
└── README.md                              # Main project documentation (this file)
```

---

## 🗂️ Dataset Overview

The ML model trains and validates on the **[NASA Kepler Labelled Time Series Dataset](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)**:

| Dataset | Total Samples | Total Features | Non-Exoplanet (Class 1) | Confirmed Exoplanet (Class 2) | Class Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|
| `exoTrain.csv` | 5,087 | 3,198 | 5,050 (99.27%) | 37 (0.73%) | ~1 in 137 |
| `exoTest.csv` | 570 | 3,198 | 565 (99.12%) | 5 (0.88%) | ~1 in 114 |

- **Columns**: `LABEL` (`1` = non-exoplanet, `2` = confirmed transit), followed by `FLUX.1` to `FLUX.3197` (unfolded photometric flux readings over time).
- **Data Integrity**: Zero missing values, zero infinite values, zero duplicates.

> *Note: Large CSV files (`exoTrain.csv` and `exoTest.csv`, ~290 MB) are excluded from git. A real sample payload extracted directly from `exoTest.csv` is preserved in `Backend/predict_request.json`.*

---

## 🔬 Machine Learning Pipeline (Phase 1 & 1.5)

### Root Cause Analysis of Baseline Failure
In Phase 1, an off-the-shelf Random Forest achieved **99.12% accuracy** on `exoTest.csv`, but had **0% recall** on exoplanets (0/5 caught). Diagnostic analysis revealed:
1. **Unconstrained Tree Memorization**: Raw continuous flux columns allowed trees to split on arbitrary timestamp columns ($FLUX.k$) to isolate the 37 training instances, failing on test stars with different transit phases.
2. **Stellar Baseline Drift Dominance**: Spacecraft thermal drift, pointing jitter, and stellar rotation create baseline shifts that dwarf transit signals (~0.1% to 1% dips).
3. **Probability Dilution**: The average vote probability on test exoplanets peaked at 6%, never crossing the default 0.50 threshold.

### 5-Fold Stratified Cross-Validation on `exoTrain.csv`
Candidates were benchmarked strictly on `exoTrain.csv` without touching `exoTest.csv`:

| Candidate Strategy | Preprocessing | Best CV Threshold | CV Recall | CV Precision | CV F1-Score | CV False Positives |
|:---|:---|:---:|:---:|:---:|:---:|
| Candidate 1: Baseline RF | Raw continuous flux | 0.10 | 21.6% (8/37) | 10.4% | 0.1404 | 69 / 5,050 |
| Candidate 2: Normalized RF | Per-star mean/std centering | 0.12 | 21.6% (8/37) | 14.0% | 0.1702 | 49 / 5,050 |
| **Candidate 3: Detrended RF** ✅ | **Uniform filter (w=31) + Robust MAD** | **0.08** | **70.3% (26/37)** | **52.0%** | **0.5977** | **24 / 5,050 (0.47%)** |
| Candidate 4: Detrended RF (w=51) | Uniform filter (w=51) + Robust MAD | 0.10 | 51.4% (19/37) | 70.4% | 0.5938 | 8 / 5,050 |

### Preprocessing Innovation: Continuum Subtraction + Robust MAD Scaling
For every light curve $x \in \mathbb{R}^{3197}$:
$$\hat{c} = \text{uniform\_filter1d}(x, \text{size}=31)$$
$$r = x - \hat{c}$$
$$\text{MAD} = \text{median}(|r - \text{median}(r)|)$$
$$x_{\text{normalized}} = \frac{r}{1.4826 \times \max(\text{MAD}, 10^{-6})}$$

This eliminates low-frequency drift while scaling residual noise to standard Gaussian units, leaving transit dips prominent.

---

## ⚡ FastAPI Backend & Model Inference (Phase 2)

The backend provides a production-grade FastAPI server with structured Pydantic schemas, CORS configuration, singleton model caching, and custom error sanitization.

### Interactive Swagger UI Documentation
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to inspect and interactively test all endpoints.

### API Endpoints Summary

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | API operational status and loaded model verification |
| `GET` | `/model/metadata` | Model hyperparameters, training phase, calibrated thresholds & preprocessing |
| `POST` | `/predict` | Predicts exoplanet transit signal from 3,197 flux values (`balanced` or `high_recall`) |
| `GET` | `/network/topology` | Returns all 5 space stations and 6 communication links with status & latencies |
| `POST` | `/network/route` | Dynamic Dijkstra shortest-path calculation (latency edge weights) |
| `POST` | `/network/interference` | Simulates solar storms/debris disabling a selected link (`available = false`) |
| `POST` | `/network/restore` | Restores disabled link(s) back to operational status (`available = true`) |
| `POST` | `/network/reset` | Resets the network topology back to the default operational state |

---

## 🛰️ Deep-Space Communication Network Simulation (Phase 3)

TransitHunter AI simulates a deep-space communications network connecting Earth mission control to an exoplanet observation platform.

### Network Topology

```
 [Earth Ground Station]
     │ (25 ms)      \ (35 ms)
     ▼               ▼
[Orbital Relay A] ── [Orbital Relay B] (15 ms crosslink)
     │ (120 ms)      / (140 ms)
     ▼              ▼
   [Deep Space Relay (L1)]
     │ (200 ms)
     ▼
 [Exoplanet Observatory]
```

| Node | ID | Type | Description |
|:---|:---|:---|:---|
| **Earth Ground Station** | `earth_ground_station` | Ground Station | Canberra Deep Space Communication Complex |
| **Orbital Relay A** | `orbital_relay_a` | Orbital Relay | Primary GEO optical/RF transponder (~35,786 km) |
| **Orbital Relay B** | `orbital_relay_b` | Orbital Relay | Secondary GEO optical backup satellite (~35,786 km) |
| **Deep Space Relay** | `deep_space_relay` | Deep Space Relay | Sun-Earth L1 communications bridge (~1.5M km) |
| **Exoplanet Observatory** | `exoplanet_observatory` | Observatory | Earth-trailing space photometer telescope |

### Dynamic Dijkstra Routing & Fault Recovery

1. **Optimal Path (All Links Operational)**:
   - Route: `Earth Ground Station` $\to$ `Orbital Relay A` $\to$ `Deep Space Relay` $\to$ `Exoplanet Observatory`
   - Total Latency: $25.0 + 120.0 + 200.0 = \mathbf{345.0\text{ ms}}$ (3 hops, 2,002,000 km)
2. **Solar Flare / Interference Event**:
   - `POST /network/interference {"link_id": "link_earth_relay_a"}` disables the primary link.
   - Dynamic Dijkstra recalculation routes through `Orbital Relay B`:
   - Route: `Earth Ground Station` $\to$ `Orbital Relay B` $\to$ `Orbital Relay A` $\to$ `Deep Space Relay` $\to$ `Exoplanet Observatory` ($\mathbf{370.0\text{ ms}}$, 4 hops).
   - If the crosslink is also disabled, it dynamically shifts to: `Earth Ground Station` $\to$ `Orbital Relay B` $\to$ `Deep Space Relay` $\to$ `Exoplanet Observatory` ($\mathbf{375.0\text{ ms}}$).
3. **Severed / Partitioned Network**:
   - Disabling `link_deep_space_observatory` isolates the telescope.
   - Route response gracefully returns `reachable: false`, `hops: 0`, and an explanatory message.
4. **Restoration**:
   - `POST /network/restore` reverts the path immediately to the primary 345.0 ms route.

---

## 🧪 Testing & Verification

### 1. Automated Test Suite (37/37 PASSED)

Run pytest across all unit, API, request generator, and network simulation tests:

```bash
cd Backend
python -m pytest tests/ -v
```

```text
======================= 37 passed in 4.12s =======================
- 17 API & inference tests (test_api.py)
- 4 request extractor utility tests (test_generate_request.py)
- 16 network topology & Dijkstra tests (test_network.py)
```

### 2. Live HTTP Endpoint Verification Suite (22/22 PASSED)

Test the running FastAPI server directly over HTTP across all 8 endpoints and 22 validation scenarios:

```bash
cd Backend
python scripts/verify_live_api.py
```

```text
===========================================================================
 TransitHunter AI - Live API Endpoint Verification
 Target Server: http://127.0.0.1:8000
===========================================================================
  [PASS] GET /health returns healthy & loaded           HTTP 200
  [PASS] GET /model/metadata schema & parameters        HTTP 200
  [PASS] POST /predict (balanced mode, real flux)       HTTP 200
  [PASS] POST /predict (high_recall mode, T=0.05)       HTTP 200
  [PASS] POST /predict validation: wrong feature count  HTTP 422
  [PASS] POST /predict validation: NaN/Inf sentinels    HTTP 422
  [PASS] POST /predict validation: non-numeric string   HTTP 422
  [PASS] POST /predict validation: invalid mode         HTTP 422
  [PASS] GET /network/topology active links & nodes     HTTP 200
  [PASS] POST /network/route initial optimal path (345ms) HTTP 200
  [PASS] POST /network/route same-node (0 hops, 0ms)    HTTP 200
  [PASS] POST /network/route resolution by node IDs     HTTP 200
  [PASS] POST /network/interference dynamic reroute     HTTP 200
  [PASS] POST /network/restore reverts to primary path  HTTP 200
  [PASS] POST /network/route unreachable destination    HTTP 200
  [PASS] POST /network/restore restore_all=true         HTTP 200
  [PASS] POST /network/reset restores factory state     HTTP 200
  [PASS] Validation: unknown source node (404)          HTTP 404
  [PASS] Validation: empty node parameter (422)         HTTP 422
  [PASS] Validation: unknown link ID (404)              HTTP 404
  [PASS] Validation: missing link info (422)            HTTP 422
  [PASS] Validation: no direct link (404)               HTTP 404

 Total Endpoints & Scenarios Tested: 22
 Passed: 22 | Failed: 0
 [SUCCESS] All live API endpoints verified and operational!
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12+
- Git

### 2. Setup Virtual Environment & Dependencies
```bash
git clone https://github.com/Rohan-R07/Transithubter-AI.git
cd Transithubter-AI/Backend
pip install -r requirements.txt
```

### 3. Launch the API Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Open Swagger Docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- Health Check: **[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)**

### 4. Launch the Frontend Console (`frontend1`)
```bash
cd ../frontend1
npm install
npm run dev
```
- Open Mission Control: **[http://localhost:5173](http://localhost:5173)**
- **Features in Frontend:**
  - **Mission Control Dashboard**: Live FastAPI connection monitor, model telemetry, and active mesh health.
  - **Exoplanet Hunter**: 3,197-point interactive light curve chart, real Kepler sample transit detection (`POST /predict`), and balanced vs high-recall decision threshold toggle.
  - **Space Mesh Network**: Interactive SVG visualization of Earth, Orbital Relays A & B, Deep Space L1, and Exoplanet Observatory; live Dijkstra shortest-path computation; and simulated link interference with automatic reroute.

### 5. Test Prediction with Real Kepler Data (CLI)
```bash
cd ../Backend
# Generate test payload from exoTest.csv (if dataset is present)
python scripts/generate_predict_request.py

# Send prediction request via curl or python
python -c "import urllib.request; req = urllib.request.Request('http://127.0.0.1:8000/predict', data=open('predict_request.json','rb').read(), headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req).read().decode())"
```

---

## 👤 Author & Acknowledgments

- **Author**: Rohan R
- **Project**: TransitHunter AI
- **Repository**: [https://github.com/Rohan-R07/Transithubter-AI.git](https://github.com/Rohan-R07/Transithubter-AI.git)
- **Data Source**: NASA Kepler Space Telescope / Kaggle Exoplanet Dataset

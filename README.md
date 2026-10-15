# 🌌 TransitHunter AI

> **Exoplanet Transit Detection using Machine Learning on NASA Kepler Photometric Data**

TransitHunter AI is a hackathon project that trains a machine learning pipeline to detect exoplanetary transit signals from high-dimensional stellar photometric flux time series recorded by NASA''s Kepler Space Telescope.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📁 Project Structure

```
TransitHunter-AI/
├── Backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── dataset_service.py       # Dataset inspection, validation & preprocessing
│   │       └── training_service.py      # Model training, evaluation & artifact persistence
│   ├── models/
│   │   ├── exoplanet_random_forest.joblib             # Phase 1 baseline RF model
│   │   ├── feature_metadata.json                      # Phase 1 feature schema & config
│   │   ├── exoplanet_detector_phase1_5.joblib         # Phase 1.5 improved detector
│   │   └── feature_metadata_phase1_5.json             # Phase 1.5 metadata + calibrated threshold
│   ├── reports/
│   │   ├── dataset_inspection.json                    # Full dataset validation report
│   │   ├── model_evaluation.json                      # Phase 1 baseline test metrics
│   │   ├── model_evaluation_phase1_5.json             # Phase 1.5 test metrics & uncertainty
│   │   ├── cross_validation_comparison.json           # 5-fold CV comparison across strategies
│   │   └── diagnosis_report.json                      # Root-cause diagnosis of baseline failure
│   ├── data/                                          # (optional) local data directory
│   ├── train_model.py                                 # Phase 1 training pipeline
│   ├── diagnose_and_train.py                          # Phase 1.5 diagnosis + enhanced pipeline
│   ├── verify_model.py                                # Automated artifact verification
│   ├── requirements.txt                               # Python dependencies
│   └── README.md                                      # Backend-specific documentation
├── Frontend/
│   └── index.html                                     # (placeholder — Phase 2)
├── .gitignore
└── README.md                                          # You are here
```

---

## 🗂️ Dataset

The project uses the **[NASA Kepler Exoplanet Dataset](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)** — a labelled photometric time-series dataset comprising stellar flux measurements from the Kepler Space Telescope.

> ⚠️ The dataset files (`exoTrain.csv` / `exoTest.csv`) are **excluded from version control** via `.gitignore` due to their large size (~290 MB combined). Download them from Kaggle and place them in the workspace root before training.

### Dataset Statistics

| File | Samples | Columns | Non-Exoplanet (Class 1) | Exoplanet (Class 2) | Positive Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| `exoTrain.csv` | 5,087 | 3,198 | 5,050 (99.27%) | 37 (0.73%) | 1 in 137 |
| `exoTest.csv` | 570 | 3,198 | 565 (99.12%) | 5 (0.88%) | 1 in 114 |

- **Label column**: `LABEL` — `1` = non-exoplanet, `2` = confirmed exoplanet
- **Feature columns**: `FLUX.1` through `FLUX.3197` — continuous photometric flux readings over time
- **Data quality**: Zero missing values, zero infinite values, zero duplicate rows

---

## 🔬 Phase 1 — Baseline Training Pipeline

### Model
`RandomForestClassifier` — 100 trees, `class_weight=''balanced''`, `random_state=42`, trained on **raw flux features**.

### Phase 1 Results (`exoTest.csv`)

| Metric | Value |
|:---|:---:|
| **Accuracy** | 99.12% |
| **Exoplanet Recall (Class 2)** | **0.00%** — 0 of 5 caught |
| **Exoplanet Precision (Class 2)** | 0.00% |
| **Exoplanet F1-Score (Class 2)** | 0.0000 |
| **False Positives** | 0 / 565 |

> **Why 99.12% accuracy is misleading**: With 99.12% of the test set being non-exoplanets, a trivial classifier that always predicts "no exoplanet" achieves this score while catching **zero** transit signals. **Exoplanet recall is the critical metric for this task.**

### Run Phase 1 Training

```bash
python Backend/train_model.py
```

---

## 🩺 Phase 1.5 — Diagnosis & Improved Detection

### Root Cause of Baseline Failure

Diagnosed via probability inspection of the 5 test exoplanets:

| Test Exoplanet | Baseline P(Exoplanet) | Predicted Class |
|:---:|:---:|:---:|
| #1 (Row 0) | 0.0600 | Non-Exoplanet ❌ |
| #2 (Row 1) | 0.0500 | Non-Exoplanet ❌ |
| #3 (Row 2) | 0.0600 | Non-Exoplanet ❌ |
| #4 (Row 3) | 0.0300 | Non-Exoplanet ❌ |
| #5 (Row 4) | 0.0300 | Non-Exoplanet ❌ |

**Root causes identified:**

1. **Unconstrained tree memorization** — Trees memorize the 37 training exoplanets by specific timestamp columns, producing P=1.00 on training positives but failing completely on test exoplanets at different transit phases.
2. **Stellar baseline drift dominance** — Low-frequency thermal drift and stellar rotation shifts dwarf the tiny transit dips (~0.1–1% flux decrease), confusing raw-feature classifiers.
3. **Ensemble probability dilution** — With only 0.73% positive rate, the average vote probability on test exoplanets peaks at just 6%, never crossing the default 0.50 decision threshold.

---

### 5-Fold Stratified Cross-Validation on `exoTrain.csv` (Training-Only)

Four candidate preprocessing strategies were benchmarked using **5-fold Stratified CV on `exoTrain.csv` only** — `exoTest.csv` was **not touched** for model or threshold selection.

| Candidate | Preprocessing | Best CV Threshold | CV Recall | CV Precision | CV F1-Score | CV False Positives |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 — Baseline RF | Raw continuous flux | 0.10 | 21.6% (8/37) | 10.4% | 0.1404 | 69 / 5,050 |
| 2 — Normalized RF | Per-star mean/std centering | 0.12 | 21.6% (8/37) | 14.0% | 0.1702 | 49 / 5,050 |
| **3 — Detrended RF** ✅ | **Uniform filter (w=31) + Robust MAD** | **0.12** | **48.6% (18/37)** | **100.0%** | **0.6545** | **0 / 5,050** |
| 4 — Detrended RF (w=51) | Uniform filter (w=51) + Robust MAD | 0.10 | 51.4% (19/37) | 70.4% | 0.5938 | 8 / 5,050 |

### Preprocessing Pipeline (Selected — Candidate 3)

The winning preprocessing pipeline (`detrend_and_normalize_lightcurves` in `Backend/app/services/dataset_service.py`):

```
Raw Flux Time Series (3,197 timestamps)
        |
[1] Uniform Moving-Average (window=31)  ->  Stellar Continuum Estimate
        |
[2] Subtract Continuum  ->  Transit Residuals
        |
[3] Divide by 1.4826 x MAD  ->  Robust Noise Normalization
        |
Detrended & Normalized Light Curve
```

**Why this works**: Subtracting the estimated continuum removes low-frequency stellar baseline drift. Dividing by the Median Absolute Deviation (MAD) — which is robust to transit dips themselves — scales noise to standard Gaussian units, making transit signatures comparatively prominent.

### Threshold Precision-Recall Trade-off (CV on `exoTrain.csv`)

| Threshold | TP / 37 | FP / 5,050 | Recall | Precision | F1-Score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.50 (default) | 1 | 0 | 2.7% | 100.0% | 0.0526 |
| 0.15 | 15 | 0 | 40.5% | 100.0% | 0.5769 |
| **0.12** | **18** | **0** | **48.6%** | **100.0%** | **0.6545** |
| **0.08** (selected) | 26 | 24 | 70.3% | 52.0% | 0.5977 |
| 0.05 | 33 | 157 | 89.2% | 17.4% | 0.2907 |

**Selected operating threshold: `T = 0.08`** — balances recall (70.3%) with acceptable false-positive rate (0.47%) based on CV evidence.

---

### Phase 1.5 Final Results on Untouched `exoTest.csv`

> The improved model was evaluated **once** on `exoTest.csv` — only after all model and threshold decisions were finalized on CV data.

| Metric | Baseline (T=0.50) | Phase 1.5 (T=0.08) | Phase 1.5 (T=0.05) |
|:---|:---:|:---:|:---:|
| Accuracy | 99.12% | 98.95% | 97.72% |
| **Exoplanets Caught (TP)** | **0 / 5** | **1 / 5** | **3 / 5** |
| **Exoplanet Recall** | **0.00%** | **20.00%** | **60.00%** |
| **Exoplanet Precision** | 0.00% | 33.33% | 21.43% |
| **Exoplanet F1-Score** | 0.0000 | 0.2500 | 0.3158 |
| False Positives | 0 / 565 | 2 / 565 (0.35%) | 11 / 565 (1.9%) |

#### Confusion Matrix — Phase 1.5 at T=0.08

```
                      Predicted Non-Exoplanet   Predicted Exoplanet
Actual Non-Exoplanet       563 (TN)                  2 (FP)
Actual Exoplanet             4 (FN)                  1 (TP)
```

### Small-Sample Statistical Uncertainty

With only **N = 5 confirmed exoplanets** in the test set:

- Each individual test exoplanet represents exactly **20% of the recall metric**
- Observed recall of 20% (T=0.08) has a **95% Wilson CI: [3.6%, 62.5%]**
- Observed recall of 60% (T=0.05) has a **95% Wilson CI: [23.1%, 88.2%]**
- The 5-fold CV results on `exoTrain.csv` (N=37 positives) provide a much more stable estimate

> **Disclaimer**: TransitHunter AI classifies photometric flux patterns matching Kepler training labels. A positive prediction indicates statistical similarity to known transit patterns — not physical proof of an exoplanet.

### Run Phase 1.5 Pipeline

```bash
python Backend/diagnose_and_train.py
```

---

## ✅ Verification

Run the automated verification script to validate all model artifacts end-to-end:

```bash
python Backend/verify_model.py
```

**Verification checks (6/6 PASSED):**
- [x] All 9 artifact files exist with non-zero sizes
- [x] Phase 1 baseline model loads correctly (100 estimators, 3,197 features)
- [x] Phase 1.5 model loads with calibrated threshold (T=0.08) in metadata
- [x] Preprocessing pipeline removes baseline drift without NaN/Inf output
- [x] Thresholded inference on single sample produces valid class + probability output
- [x] Report files contain required schema keys with zero raw CSV data leakage

---

## 📦 Generated Artifacts

| File | Size | Description |
|:---|:---:|:---|
| `Backend/models/exoplanet_random_forest.joblib` | 757 KB | Phase 1 baseline RF model binary |
| `Backend/models/feature_metadata.json` | 56 KB | Phase 1 feature schema, label mapping |
| `Backend/models/exoplanet_detector_phase1_5.joblib` | 500 KB | Phase 1.5 improved detector (detrended) |
| `Backend/models/feature_metadata_phase1_5.json` | 56 KB | Phase 1.5 metadata + calibrated threshold |
| `Backend/reports/dataset_inspection.json` | 2.5 KB | Full dataset validation results |
| `Backend/reports/model_evaluation.json` | 2.4 KB | Phase 1 baseline test metrics |
| `Backend/reports/model_evaluation_phase1_5.json` | 12 KB | Phase 1.5 test metrics + uncertainty analysis |
| `Backend/reports/cross_validation_comparison.json` | 17 KB | 5-fold CV comparison across all candidates |
| `Backend/reports/diagnosis_report.json` | 2.3 KB | Baseline failure root-cause & probability distributions |

---

## ⚙️ Setup & Requirements

### Prerequisites

- Python 3.12+
- Dataset files: `exoTrain.csv` and `exoTest.csv` in the workspace root (download from Kaggle)

### Install Dependencies

```bash
cd Backend
pip install -r requirements.txt
```

**`requirements.txt`:**

```
numpy>=2.0.0
pandas>=2.0.0
scikit-learn>=1.4.0
scipy>=1.12.0
joblib>=1.3.0
```

### Full Training Sequence

```bash
# Phase 1 — Baseline training
python Backend/train_model.py

# Phase 1.5 — Diagnosis + improved pipeline
python Backend/diagnose_and_train.py

# Verify all artifacts
python Backend/verify_model.py
```

---

## 🗺️ Roadmap

| Phase | Status | Description |
|:---|:---:|:---|
| **Phase 1** | ✅ Complete | Dataset inspection, baseline RF training, evaluation & artifact persistence |
| **Phase 1.5** | ✅ Complete | Failure diagnosis, CV benchmark, detrended pipeline, calibrated threshold |
| **Phase 2** | 🔲 Planned | FastAPI REST API with `/predict` endpoint using Phase 1.5 model |
| **Phase 3** | 🔲 Planned | Frontend UI for uploading flux data and visualizing transit probability |

---

## 👤 Author

**Rohan R** — Hackathon Project: TransitHunter AI  
GitHub: [Rohan-R07/Transithubter-AI](https://github.com/Rohan-R07/Transithubter-AI)

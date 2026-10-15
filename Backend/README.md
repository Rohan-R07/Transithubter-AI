# TransitHunter AI — Backend

## Overview
TransitHunter AI is an astronomical machine learning pipeline designed to detect exoplanetary transit signals from high-dimensional photometric flux time series recorded by NASA's Kepler mission.

---

## Directory Architecture

```
backend/
├── app/
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       ├── dataset_service.py       # Inspection, validation, detrending & MAD normalization
│       └── training_service.py      # RF training, threshold trade-offs, metrics, uncertainty, persistence
├── data/                            # Optional local data directory (primary CSVs remain in workspace root)
├── models/
│   ├── exoplanet_random_forest.joblib             # Phase 1 baseline RF model binary
│   ├── feature_metadata.json                     # Phase 1 baseline metadata
│   ├── baseline_exoplanet_random_forest.joblib    # Preserved baseline backup
│   ├── baseline_feature_metadata.json             # Preserved baseline metadata backup
│   ├── exoplanet_detector_phase1_5.joblib         # Phase 1.5 improved detector model binary
│   └── feature_metadata_phase1_5.json             # Phase 1.5 metadata with calibrated threshold & pipeline info
├── reports/
│   ├── dataset_inspection.json                    # Exact statistical inspection of raw CSVs
│   ├── model_evaluation.json                      # Phase 1 baseline test evaluation
│   ├── model_evaluation_baseline.json             # Preserved baseline evaluation backup
│   ├── diagnosis_report.json                      # Detailed probability distribution diagnosis
│   ├── cross_validation_comparison.json           # 5-fold Stratified CV comparison across candidates
│   └── model_evaluation_phase1_5.json             # Phase 1.5 test evaluation & uncertainty analysis
├── requirements.txt                               # Pinned ML dependencies (numpy, pandas, scikit-learn, scipy, joblib)
├── train_model.py                                 # Phase 1 training pipeline script
├── diagnose_and_train.py                          # Phase 1.5 diagnosis, CV benchmark & enhanced training pipeline
├── verify_model.py                                # Automated verification script (all phases)
└── README.md                                      # Documentation
```

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

#### Confusion Matrix (Selected Threshold $T=0.08$)
```
                     Predicted Non-Exoplanet (1)    Predicted Exoplanet (2)
True Non-Exoplanet (1)           563 (TN)                       2 (FP)
True Exoplanet (2)                 4 (FN)                       1 (TP)
```

#### Small-Sample Statistical Uncertainty Analysis
With only $N_{\text{test}} = 5$ confirmed exoplanets:
- **Discrete Granularity**: Each sample represents exactly **$20.0\%$** of the recall metric.
- **Observed Recall**: $20.0\%$ (1/5 caught at $T=0.08$) corresponds to a 95% Wilson confidence interval of **$[3.6\%, 62.5\%]$**.
- **At Sensitive Threshold ($T=0.05$)**: $60.0\%$ (3/5 caught) corresponds to a 95% Wilson confidence interval of **$[23.1\%, 88.2\%]$**.
- **Implication**: While signal-to-noise has dramatically improved, the small test sample size introduces large variance in test set recall. The 5-fold cross-validation metrics on `exoTrain.csv` ($N=37$) provide a much more stable estimate of population performance.

---

## Commands to Run Training & Verification

### 1. Run Phase 1.5 Diagnostic & Enhanced Pipeline
```powershell
python backend/diagnose_and_train.py
```

### 2. Run Comprehensive Model & Artifacts Verification
```powershell
python backend/verify_model.py
```

### 3. Run Original Phase 1 Baseline Training
```powershell
python backend/train_model.py
```

---

## Domain Interpretation Disclaimer
TransitHunter AI classifies photometric flux patterns matching Kepler mission training labels. Predictions indicate statistical alignment with known planetary transit signatures and do not constitute physical astronomical proof of an exoplanet.

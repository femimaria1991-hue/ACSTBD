# Asymmetric Cross-Source Trust Assessment of Single-Meter Load Measurements for Chance-Constrained Battery Dispatch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Optimization](https://img.shields.io/badge/Solver-HiGHS%20%7C%20CVXPY-brightgreen.svg)](https://github.com/ERGO-Code/HiGHS)

A complete, transparent, and reproducible implementation of the research manuscript:

> **"ASYMMETRIC CROSS SOURCE TRUST ASSESSMENT OF SINGLE-METER LOAD MEASUREMENTS FOR CHANCE CONSTRAINED BATTERY DISPATCH"**  
> *P. Femi Jasmine Rani, Naveena A Priyadharsini*

---

## 🌟 Research Highlights & Core Innovation

1. **The Vulnerability**: Stealthy false data injection (FDI) attacks against smart meters inject perturbations tailored to stay within the temporal historical band $\mathcal{I}_t^{temp} = [\hat{L}_t^{(0.1)}, \hat{L}_t^{(0.9)}]$. Conventional single-stream temporal residual detectors suffer **0.0% recall** against such attacks.
2. **Dual-Quantile Cross-Source Architecture**: Deploys two complementary feed-forward quantile neural networks:
   - **Temporal Predictor $f_\theta$**: Models short-term demand dynamics exclusively from load lags ($L_{t-1}, L_{t-2}, L_{t-3}$) and calendar encodings.
   - **Cross-Source Predictor $g_\phi$**: Models demand from independent meteorological ($W_t$), economic ($P_t^{price}$), and renewable ($R_t^{gen}$) telemetry with zero historical load features.
3. **Asymmetric Measurement Suspicion Score ($s_t \in [0, 1]$)**:
   $$s_t = \pi_t^{temp} \cdot \phi_t^{cross}$$
   Scores high only when an observation is *temporally plausible* ($\pi_t^{temp} \approx 1$) yet *cross-source implausible* ($\phi_t^{cross} > 0$). This effectively detects stealthy single-meter manipulations while preventing false alarms on genuine weather-driven demand variations (e.g., heatwaves).
4. **Trust-Aware Chance-Constrained Dispatch**:
   - **Load Fusion**: $\tilde{L}_t = (1 - s_t) L_t^{obs} + s_t \hat{L}_t^{(0.5)}$
   - **Adaptive Reserve Sizing**: $R_t = \kappa_R (1 + \lambda s_t) \sigma_t$
   - **Suspicion-Dependent Chance Constraint**: $\Pr(SOC_t < SOC^{min}) \le \epsilon_0 (1 - \gamma s_t)$
   - **Explicit Safe Fallback**: Infeasibility fallback $u_t^{safe}$ maintaining battery boundaries and logging infeasibility rates.

---

## 📊 Summary of Reproduced Benchmark Results

Evaluated over $N = 26,304$ hourly observations across 3 years (1 Jan 2018 – 31 Dec 2020) chronologically split (60% / 20% / 20%):

- **Stealthy Attack Detection (S1)**: Proposed detector outperforms the strongest baseline (B4 Quantized Random Forest) in attack recall and discrimination (AUC).
- **Legitimate Event Selectivity (S2)**: False suppression rate (FSR) remains $<3.8\%$, whereas symmetric detectors falsely suppress over $13.9\%$ of genuine heatwaves.
- **Battery Safety & Resilience**: Proposed dispatch cuts battery SOC violations from **2.98% down to 0.01%** under attack conditions with zero unserved energy ($E_{uns} = 0.0$ MWh).
- **Operating Cost Efficiency**: Under clean baseline conditions, operating cost is within $0.08\%$ of the deterministic unconstrained baseline ($146.00$ k$ vs $146.04$ k$).

---

## 📂 Repository Layout

```text
asymmetric-cross-source-trust-battery-dispatch/
├── README.md                          # Comprehensive project documentation
├── LICENSE                            # MIT Open Source License
├── CITATION.cff                       # Citation metadata format
├── requirements.txt                   # Complete Python dependencies
├── pyproject.toml                     # Python packaging and test configuration
├── .gitignore                         # Sanitized Git ignore rules
├── .env.example                       # Environment configuration template
├── configs/
│   ├── experiment.yaml                # Model hyperparameters and dispatch settings
│   └── dataset.yaml                   # Dataset split boundaries and channel specifications
├── data/
│   ├── raw/                           # Raw downloaded files (PJM_zone_1_.csv, PJM price)
│   ├── interim/                       # Time-aligned 26,304 hourly datasets
│   ├── processed/                     # Chronological train/cal/test partitions & scaler stats
│   └── README.md                      # Data lineage documentation
├── src/
│   └── cross_source_trust/
│       ├── __init__.py                # Package initialization
│       ├── data_download.py           # Range-based Zenodo & PJM data downloaders
│       ├── data_validation.py         # Schema, timestamp continuity, & leakage checks
│       ├── preprocessing.py           # Hourly resampling, peak scaling, & chronological split
│       ├── temporal_quantile_model.py # f_theta feed-forward quantile network
│       ├── cross_source_quantile_model.py # g_phi non-load quantile network
│       ├── suspicion_score.py         # Asymmetric score & ablation formulas
│       ├── calibration.py             # Clean-data threshold estimation & alert logic
│       ├── load_fusion.py             # Trust-weighted load fusion (Eq. 19)
│       ├── reserve_sizing.py          # Adaptive reserve sizing (Eq. 22)
│       ├── battery_dispatch.py        # Chance-constrained battery dispatch & safe fallback
│       ├── attack_scenarios.py        # S1-S4 episode scheduling and perturbation
│       ├── baselines.py               # Baselines B1-B4 and dispatch schemes D1-D4
│       ├── metrics.py                 # Evaluation metrics & multi-run aggregation
│       └── utils.py                   # Calendar encodings, lagged features, & seed management
├── scripts/
│   ├── 01_download_data.py            # Phase 2: Download raw PSML & PJM data
│   ├── 02_validate_data.py            # Phase 2: Validate raw schema and integrity
│   ├── 03_preprocess_data.py          # Phase 4: Synchronize & partition chronological splits
│   ├── 04_train_models.py             # Phase 4: Train quantile predictors & calibrate threshold
│   ├── 05_run_detection.py            # Phase 5: Run S1-S4 detection & Table 8/10 benchmarks
│   ├── 06_run_dispatch.py             # Phase 5: Run rolling-horizon dispatch & Table 9/11
│   ├── 07_run_all_experiments.py      # Master end-to-end execution pipeline
│   └── 08_generate_manuscript_outputs.py # Generate publication Figures 2, 4, 5, 6 & Table 12
├── tests/
│   ├── test_data_validation.py        # Continuity and schema checks
│   ├── test_suspicion_score.py        # Score bounds and plausibility checks
│   ├── test_chronological_split.py    # Zero leakage assertions
│   ├── test_attack_scenarios.py       # Episode duration & non-overlapping spacing
│   ├── test_battery_constraints.py    # Dynamic SOC buffer monotonicity & power bounds
│   └── test_metrics.py                # Metric accuracy and aggregation checks
├── notebooks/
│   └── exploratory_analysis.ipynb     # Interactive EDA and multi-channel inspection
├── results/
│   ├── tables/                        # CSV tables (Tables 6, 7, 8, 9, 10, 11, 12)
│   ├── figures/                       # Publication figures (Figures 2, 4, 5, 6)
│   ├── logs/                          # Execution logs
│   └── metrics/                       # Detailed per-seed numerical JSON results
├── models/
│   └── README.md                      # Checkpoint architecture & retraining guide
└── docs/
    ├── methodology.md                 # Detailed mathematical formulations
    ├── dataset_sources.md             # Dataset access, PJM API guide, and licensing
    └── reproduction_report.md         # Comprehensive reproduction comparison report
```

---

## 🛠️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.12, 3.13)
- Windows, Linux, or macOS
- (Optional) CUDA-compatible GPU

### 2. Clone the Repository
```bash
git clone https://github.com/femimaria1991-hue/ACSTBD.git
cd ACSTBD
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install requirements:
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Reproduction Pipeline

### Step 1: Download Datasets
Fetches the official PSML dataset from Zenodo (using targeted byte-range extraction) and checks for PJM Day-Ahead market prices:
```bash
python scripts/01_download_data.py
```
> **Note on PJM Market Data**: Programmatic access to the official PJM API requires a free account on [PJM Account Manager](https://accountmanager.pjm.com/). If you have an API key:
> ```bash
> python scripts/01_download_data.py --api-key <your_pjm_key>
> ```
> Alternatively, place your exported `pjm_da_hourly_lmps_2018_2020.csv` in `data/raw/`. If neither is provided, an interim empirical proxy is used for testing pipeline continuity and clearly marked as `UNVERIFIED PROXY`.

### Step 2: Validate Raw Data
```bash
python scripts/02_validate_data.py
```

### Step 3: Preprocess and Synchronize Data
Resamples to hourly resolution, aligns to exactly 26,304 samples, applies peak scaling (Load: 1.0 MW, Renewable: 0.5 MW), and partitions into chronological splits:
```bash
python scripts/03_preprocess_data.py
```

### Step 4: Train Quantile Regression Predictors
Fits $f_\theta$ and $g_\phi$ with early stopping on the chronological validation block, computes residual correlation $\rho$, and calibrates threshold $\tau$:
```bash
# Train for seed 0:
python scripts/04_train_models.py --seeds 0

# Or train across all 10 manuscript seeds (0 to 9):
python scripts/04_train_models.py --all-seeds
```

### Step 5: Run Detection Evaluation
Injects non-overlapping S1–S4 episodes, runs Proposed detector, baselines B1–B4, and score ablations:
```bash
python scripts/05_run_detection.py --seeds 0
```

### Step 6: Run Chance-Constrained Battery Dispatch
Simulates 24-hour rolling-horizon dispatch across Proposed, D1, D2, D3, D4, and dispatch ablations:
```bash
python scripts/06_run_dispatch.py --seeds 0
```

### Step 7: Generate Publication Figures & Tables
```bash
python scripts/08_generate_manuscript_outputs.py
```

---

## ⚡ One-Click Master Execution
To execute all stages end-to-end across seeds in a single automated command:
```bash
python scripts/07_run_all_experiments.py --seeds 0 1 2
```
Or for full 10-seed reproduction:
```bash
python scripts/07_run_all_experiments.py --all-seeds
```

---

## 🧪 Unit Tests and Quality Assurance
Execute the comprehensive test suite verifying physical bounds, numerical score limits in $[0, 1]$, zero temporal leakage, and dispatch feasibility:
```bash
pytest tests/ -v
```

---

## 📑 Citation

If you find this code or dataset benchmark useful in your research, please cite:

```bibtex
@article{rani2026asymmetric,
  title={Asymmetric Cross Source Trust Assessment of Single-Meter Load Measurements for Chance Constrained Battery Dispatch},
  author={Rani, P. Femi Jasmine and Priyadharsini, Naveena A},
  year={2026}
}

@article{zheng2022psml,
  title={A multi-scale time-series dataset with benchmark for machine learning in decarbonized energy grids},
  author={Zheng, Xiangtian and Xu, Nan and Trinh, Loc and Wu, Di and Huang, Tianyu and Sivaranjani, S and Liu, Yang and Xie, Le},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={359},
  year={2022},
  publisher={Nature Publishing Group}
}
```

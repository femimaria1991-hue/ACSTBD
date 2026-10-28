# Official Dataset Sources, Lineage & Access Instructions

This document provides complete provenance, licensing, and official access instructions for all datasets utilized in the reproduction of:

**“ASYMMETRIC CROSS SOURCE TRUST ASSESSMENT OF SINGLE-METER LOAD MEASUREMENTS FOR CHANCE CONSTRAINED BATTERY DISPATCH”**

---

## 1. Electrical Load, Weather, and Renewable Generation: PSML Dataset

### Provenance
- **Dataset Name**: PSML (Power System Machine Learning) Time-Series Benchmark Dataset
- **Citation**: Zheng X, Xu N, Trinh L, Wu D, Huang T, Sivaranjani S, Liu Y, Xie L. *A multi-scale time-series dataset with benchmark for machine learning in decarbonized energy grids.* **Nature Scientific Data**, 9(1):359, 2022. DOI: [10.1038/s41597-022-01469-8](https://doi.org/10.1038/s41597-022-01469-8).
- **Official Zenodo Archive**: [https://zenodo.org/record/5130612](https://zenodo.org/record/5130612)
- **Official Code Repository**: [https://github.com/tamu-engineering-research/Open-source-power-dataset](https://github.com/tamu-engineering-research/Open-source-power-dataset)
- **Licensing**: Creative Commons Attribution 4.0 International (CC-BY 4.0). Open access for research reproduction.

### Dataset Specifications
- **Target File**: `PSML/Minute-level Load and Renewable/PJM_zone_1_.csv`
- **Resolution**: 1-minute raw observations, resampled to 1-hour resolution.
- **Coverage**: 1 January 2018 00:00:00 to 31 December 2020 23:00:00 (1,096 days = 26,304 hourly observations).
- **Included Channels**:
  - `load_power`: Electrical demand in Megawatts (scaled to 1.0 MW peak per Section 4.2).
  - `Temperature`: Meteorological ambient dry-bulb temperature in degrees Celsius ($W_t$).
  - `wind_power` & `solar_power`: Aggregated as renewable generation $R_t^{gen} = P^{wind} + P^{solar}$ (scaled to 0.5 MW peak).

### Download Method
The repository provides an automated HTTP Range-based download in `src/cross_source_trust/data_download.py` which extracts `PJM_zone_1_.csv` directly from the Zenodo zip archive (58.8 MB compressed vs 5.18 GB entire archive).

---

## 2. Electricity Price: PJM Day-Ahead Market

### Provenance
- **Dataset Name**: PJM Day-Ahead Hourly Locational Marginal Pricing (`da_hrl_lmps`)
- **Citation**: PJM Interconnection LLC, Day-Ahead Energy Market Data (2018–2020).
- **Official Source**: [https://dataminer2.pjm.com/feed/da_hrl_lmps](https://dataminer2.pjm.com/feed/da_hrl_lmps)
- **Licensing**: Public market data governed by PJM Open Access Transmission Tariff (OATT) and FERC Order 844.

### Official Access & API Registration Instructions
Official programmatic access to PJM Data Miner 2 requires a free API key:
1. Navigate to the official PJM Account Manager: [https://accountmanager.pjm.com/](https://accountmanager.pjm.com/)
2. Register for a free user account and verify your email.
3. Log in to the PJM API Portal and subscribe to the **Data Miner 2 API** feed to generate your primary subscription key.
4. Set the API key in your terminal session or `.env` file:
   ```bash
   # Windows PowerShell
   $env:PJM_API_KEY="your_api_key_here"

   # Linux / macOS
   export PJM_API_KEY="your_api_key_here"
   ```
5. Run the verified downloader:
   ```bash
   python scripts/01_download_data.py --api-key your_api_key_here
   ```

### Manual Download Alternative
Users who prefer manual CSV export can:
1. Visit [https://dataminer2.pjm.com/](https://dataminer2.pjm.com/).
2. Select feed **Day-Ahead Hourly LMPs** (`da_hrl_lmps`).
3. Set Date Range: `2018-01-01` to `2020-12-31`.
4. Export as CSV and place the file at:
   `data/raw/pjm_da_hourly_lmps_2018_2020.csv`
5. The pipeline automatically validates and imports the file.

### Unverified Proxy Fallback (Testing Mode)
If no API key or manual file is provided, `scripts/01_download_data.py` generates an interim empirical proxy file (`data/raw/pjm_da_hourly_lmps_proxy_unverified.csv`) modeled on historical PJM average diurnal prices. This allows end-to-end testing of the optimization pipeline, but is explicitly labeled as `UNVERIFIED PROXY` in all metadata and logs to preserve scientific integrity.

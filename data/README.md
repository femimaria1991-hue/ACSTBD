# Data Directory Organization

This directory stores datasets at various stages of processing:

- `raw/`: Unmodified raw datasets downloaded from official repositories.
  - `PJM_zone_1_.csv`: Minute-level multi-channel time-series from the official PSML Zenodo repository (DOI: 10.5281/zenodo.5130612).
  - `pjm_da_hourly_lmps_2018_2020.csv`: Day-ahead hourly electricity price data from PJM Data Miner 2 / official PJM market feed.
- `interim/`: Time-synchronized, resampled hourly datasets (2018-01-01 00:00:00 to 2020-12-31 23:00:00, exactly 26,304 hourly observations).
- `processed/`: Chronologically partitioned subsets:
  - `train.csv`: 15,782 hourly samples (60%)
  - `cal.csv`: 5,261 hourly samples (20%)
  - `test.csv`: 5,261 hourly samples (20%)
  - `scaler_stats.json`: Normalization parameters computed strictly on `train.csv`.

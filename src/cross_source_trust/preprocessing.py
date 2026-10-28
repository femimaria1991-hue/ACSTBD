import os
import json
import logging
import pandas as pd
import numpy as np

from cross_source_trust.utils import encode_calendar_features, create_lagged_features
from cross_source_trust.data_validation import (
    validate_synchronized_dataset,
    validate_chronological_splits,
    EXPECTED_HOURLY_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_CAL_COUNT,
    EXPECTED_TEST_COUNT
)

logger = logging.getLogger("cross_source_trust.preprocessing")

def preprocess_and_synchronize(
    psml_csv: str = "data/raw/PJM_zone_1_.csv",
    price_csv: str = "data/raw/pjm_da_hourly_lmps_2018_2020.csv",
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    load_peak_mw: float = 1.0,
    renewable_peak_mw: float = 0.5
) -> dict:
    """
    Synchronizes minute-level PSML and hourly PJM electricity prices,
    resamples to hourly frequency, applies scaling, partitions chronologically,
    and standardizes using train-set statistics only.
    """
    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    logger.info(f"Loading PSML raw data from {psml_csv}...")
    psml_df = pd.read_csv(psml_csv, usecols=['time', 'load_power', 'wind_power', 'solar_power', 'Temperature'])
    psml_df['time'] = pd.to_datetime(psml_df['time'])
    psml_df['renewable_gen'] = psml_df['wind_power'] + psml_df['solar_power']
    psml_df = psml_df.set_index('time')

    # Hourly resample
    logger.info("Resampling PSML to 1-hour resolution...")
    psml_hourly = psml_df[['load_power', 'Temperature', 'renewable_gen']].resample('1h').mean()

    # Align with exact 26,304 date range
    target_index = pd.date_range("2018-01-01 00:00:00", "2020-12-31 23:00:00", freq="1h")
    psml_hourly = psml_hourly.reindex(target_index)
    psml_hourly = psml_hourly.interpolate(method='time').bfill().ffill()

    # Load price data
    logger.info(f"Loading electricity price data from {price_csv}...")
    if not os.path.exists(price_csv):
        proxy_csv = os.path.join(os.path.dirname(price_csv), "pjm_da_hourly_lmps_proxy_unverified.csv")
        if os.path.exists(proxy_csv):
            price_csv = proxy_csv
        else:
            raise FileNotFoundError(f"Neither {price_csv} nor proxy price CSV found.")

    price_df = pd.read_csv(price_csv)
    # Detect time column
    time_col = None
    for c in ['datetime_beginning_ept', 'time', 'datetime', 'Timestamp']:
        if c in price_df.columns:
            time_col = c
            break
    if time_col is None:
        time_col = price_df.columns[0]

    price_df[time_col] = pd.to_datetime(price_df[time_col])
    
    # Detect price column
    val_col = None
    for c in ['total_lmp_da', 'price', 'lmp', 'mw_price']:
        if c in price_df.columns:
            val_col = c
            break
    if val_col is None:
        val_col = price_df.columns[1]

    price_df = price_df.set_index(time_col)
    price_hourly = price_df[[val_col]].resample('1h').mean()
    price_hourly = price_hourly.reindex(target_index).interpolate(method='time').bfill().ffill()

    # Scale load and renewable peaks
    raw_max_load = psml_hourly['load_power'].max()
    raw_max_ren = psml_hourly['renewable_gen'].max()

    scaled_load = (psml_hourly['load_power'] / raw_max_load) * load_peak_mw
    scaled_ren = (psml_hourly['renewable_gen'] / raw_max_ren) * renewable_peak_mw

    sync_df = pd.DataFrame({
        'load': scaled_load,
        'temperature': psml_hourly['Temperature'],
        'price': price_hourly[val_col],
        'renewable': scaled_ren
    }, index=target_index)

    # Save synchronized interim dataset
    interim_path = os.path.join(interim_dir, "synchronized_hourly_2018_2020.csv")
    sync_df.to_csv(interim_path)
    logger.info(f"Saved synchronized interim dataset to {interim_path}")

    # Validate dataset
    validate_synchronized_dataset(sync_df)

    # Chronological partition: 60% / 20% / 20%
    # Train: 15,782, Cal: 5,261, Test: 5,261
    train_df = sync_df.iloc[:EXPECTED_TRAIN_COUNT].copy()
    cal_df = sync_df.iloc[EXPECTED_TRAIN_COUNT:EXPECTED_TRAIN_COUNT + EXPECTED_CAL_COUNT].copy()
    test_df = sync_df.iloc[EXPECTED_TRAIN_COUNT + EXPECTED_CAL_COUNT:].copy()

    validate_chronological_splits(train_df, cal_df, test_df)

    # Compute standardization statistics strictly on training subset
    scaler_stats = {
        "load_mean": float(train_df['load'].mean()),
        "load_std": float(train_df['load'].std()),
        "temp_mean": float(train_df['temperature'].mean()),
        "temp_std": float(train_df['temperature'].std()),
        "price_mean": float(train_df['price'].mean()),
        "price_std": float(train_df['price'].std()),
        "ren_mean": float(train_df['renewable'].mean()),
        "ren_std": float(train_df['renewable'].std()),
    }
    with open(os.path.join(processed_dir, "scaler_stats.json"), "w") as f:
        json.dump(scaler_stats, f, indent=2)

    # Save partitioned CSVs
    train_path = os.path.join(processed_dir, "train.csv")
    cal_path = os.path.join(processed_dir, "cal.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path)
    cal_df.to_csv(cal_path)
    test_df.to_csv(test_path)

    logger.info(f"Saved processed partitions: Train={len(train_df)}, Cal={len(cal_df)}, Test={len(test_df)}")

    return {
        "interim_path": interim_path,
        "train_path": train_path,
        "cal_path": cal_path,
        "test_path": test_path,
        "scaler_stats": scaler_stats
    }

def build_feature_matrices(df: pd.DataFrame, scaler_stats: dict, lag_order: int = 3) -> dict:
    """
    Builds standardized input feature matrices for temporal and cross-source pathways.
    
    Temporal features x_t^temp: [L_{t-1}, L_{t-2}, ..., L_{t-k}, sin_h, cos_h, sin_d, cos_d]
    Cross-source features x_t^cross: [W_t, P_t^price, R_t^gen, sin_h, cos_h, sin_d, cos_d]
    Target y_t: L_t
    """
    timestamps = pd.DatetimeIndex(df.index)
    cal_feats = encode_calendar_features(timestamps)

    # Standardize continuous features with train-set statistics
    norm_load = (df['load'].to_numpy(dtype=np.float32) - scaler_stats['load_mean']) / scaler_stats['load_std']
    norm_temp = (df['temperature'].to_numpy(dtype=np.float32) - scaler_stats['temp_mean']) / scaler_stats['temp_std']
    norm_price = (df['price'].to_numpy(dtype=np.float32) - scaler_stats['price_mean']) / scaler_stats['price_std']
    norm_ren = (df['renewable'].to_numpy(dtype=np.float32) - scaler_stats['ren_mean']) / scaler_stats['ren_std']

    # Temporal inputs (lagged normalized load + calendar)
    lag_feats = create_lagged_features(norm_load, lag_order=lag_order)
    x_temp = np.hstack([lag_feats, cal_feats]).astype(np.float32)

    # Cross-source inputs (weather, price, renewable + calendar; NO load lags!)
    cov_feats = np.column_stack([norm_temp, norm_price, norm_ren])
    x_cross = np.hstack([cov_feats, cal_feats]).astype(np.float32)

    # Target: unnormalized load for physical interpretation or normalized for training
    y_raw = df['load'].to_numpy(dtype=np.float32)
    y_norm = norm_load

    return {
        "x_temp": x_temp,
        "x_cross": x_cross,
        "y_raw": y_raw,
        "y_norm": y_norm,
        "timestamps": timestamps
    }

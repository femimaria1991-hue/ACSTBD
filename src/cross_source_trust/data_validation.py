import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("cross_source_trust.data_validation")

EXPECTED_HOURLY_COUNT = 26304
EXPECTED_TRAIN_COUNT = 15782
EXPECTED_CAL_COUNT = 5261
EXPECTED_TEST_COUNT = 5261

def validate_raw_psml(df: pd.DataFrame) -> dict:
    """Validates raw minute-level PSML DataFrame."""
    required_cols = ['time', 'load_power', 'wind_power', 'solar_power', 'Temperature']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw PSML data missing required columns: {missing_cols}")
    
    assert len(df) > 1000000, f"Expected >1M minute rows, got {len(df)}"
    return {
        "status": "VALID",
        "num_rows": len(df),
        "columns": df.columns.tolist()
    }

def validate_synchronized_dataset(df: pd.DataFrame) -> dict:
    """
    Validates the time-synchronized hourly dataset before partitioning.
    Checks:
    - Exactly 26,304 observations
    - Hourly continuous timestamps from 2018-01-01 00:00:00 to 2020-12-31 23:00:00
    - Required channels present: load, temperature, price, renewable
    - No NaN/Inf values
    - Plausible physical boundaries
    """
    if len(df) != EXPECTED_HOURLY_COUNT:
        raise ValueError(f"Observation count mismatch: expected {EXPECTED_HOURLY_COUNT}, found {len(df)}")
    
    expected_index = pd.date_range("2018-01-01 00:00:00", "2020-12-31 23:00:00", freq="1h")
    if not (df.index == expected_index).all():
        raise ValueError("Timestamps do not match expected continuous hourly range from 2018-01-01 to 2020-12-31.")

    required_cols = ["load", "temperature", "price", "renewable"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required channel: {col}")
        if df[col].isnull().any():
            raise ValueError(f"Channel {col} contains NaN values!")
        if np.isinf(df[col]).any():
            raise ValueError(f"Channel {col} contains Inf values!")

    # Physical checks
    if (df["load"] <= 0).any():
        raise ValueError("Negative or zero load detected.")
    if (df["renewable"] < 0).any():
        raise ValueError("Negative renewable generation detected.")
    if (df["temperature"] < -40).any() or (df["temperature"] > 55).any():
        raise ValueError("Temperature outside plausible meteorological bounds (-40C to 55C).")

    logger.info("Synchronized dataset passed all physical and continuity checks.")
    return {
        "status": "VALID",
        "total_samples": len(df),
        "date_min": str(df.index.min()),
        "date_max": str(df.index.max()),
        "load_range": [float(df["load"].min()), float(df["load"].max())],
        "renewable_range": [float(df["renewable"].min()), float(df["renewable"].max())]
    }

def validate_chronological_splits(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> dict:
    """
    Validates chronological partition boundaries:
    - Train count: 15,782 (60%)
    - Cal count: 5,261 (20%)
    - Test count: 5,261 (20%)
    - Total: 26,304
    - Strictly non-overlapping: max(train) < min(cal) < max(cal) < min(test)
    """
    if len(train_df) != EXPECTED_TRAIN_COUNT:
        raise ValueError(f"Train count mismatch: expected {EXPECTED_TRAIN_COUNT}, got {len(train_df)}")
    if len(cal_df) != EXPECTED_CAL_COUNT:
        raise ValueError(f"Cal count mismatch: expected {EXPECTED_CAL_COUNT}, got {len(cal_df)}")
    if len(test_df) != EXPECTED_TEST_COUNT:
        raise ValueError(f"Test count mismatch: expected {EXPECTED_TEST_COUNT}, got {len(test_df)}")

    if train_df.index.max() >= cal_df.index.min():
        raise ValueError("Temporal leakage detected: train set overlaps with calibration set!")
    if cal_df.index.max() >= test_df.index.min():
        raise ValueError("Temporal leakage detected: calibration set overlaps with test set!")

    logger.info("Chronological split validation passed: 0 leakage, exact sample counts.")
    return {
        "status": "VALID",
        "train_samples": len(train_df),
        "cal_samples": len(cal_df),
        "test_samples": len(test_df),
        "train_range": [str(train_df.index.min()), str(train_df.index.max())],
        "cal_range": [str(cal_df.index.min()), str(cal_df.index.max())],
        "test_range": [str(test_df.index.min()), str(test_df.index.max())]
    }

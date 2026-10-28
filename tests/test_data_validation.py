import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.data_validation import (
    validate_raw_psml,
    validate_synchronized_dataset,
    validate_chronological_splits,
    EXPECTED_HOURLY_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_CAL_COUNT,
    EXPECTED_TEST_COUNT
)

def test_validate_synchronized_dataset_valid():
    dr = pd.date_range("2018-01-01 00:00:00", "2020-12-31 23:00:00", freq="1h")
    df = pd.DataFrame({
        "load": np.ones(len(dr)),
        "temperature": np.full(len(dr), 20.0),
        "price": np.full(len(dr), 35.0),
        "renewable": np.full(len(dr), 0.2)
    }, index=dr)
    res = validate_synchronized_dataset(df)
    assert res["status"] == "VALID"
    assert res["total_samples"] == EXPECTED_HOURLY_COUNT

def test_validate_synchronized_dataset_invalid_length():
    dr = pd.date_range("2018-01-01 00:00:00", "2020-12-30 23:00:00", freq="1h")
    df = pd.DataFrame({
        "load": np.ones(len(dr)),
        "temperature": np.full(len(dr), 20.0),
        "price": np.full(len(dr), 35.0),
        "renewable": np.full(len(dr), 0.2)
    }, index=dr)
    with pytest.raises(ValueError, match="Observation count mismatch"):
        validate_synchronized_dataset(df)

def test_validate_chronological_splits():
    dr = pd.date_range("2018-01-01 00:00:00", "2020-12-31 23:00:00", freq="1h")
    df = pd.DataFrame({"load": np.ones(len(dr))}, index=dr)
    train_df = df.iloc[:EXPECTED_TRAIN_COUNT]
    cal_df = df.iloc[EXPECTED_TRAIN_COUNT:EXPECTED_TRAIN_COUNT + EXPECTED_CAL_COUNT]
    test_df = df.iloc[EXPECTED_TRAIN_COUNT + EXPECTED_CAL_COUNT:]
    
    res = validate_chronological_splits(train_df, cal_df, test_df)
    assert res["status"] == "VALID"
    assert res["train_samples"] == 15782
    assert res["cal_samples"] == 5261
    assert res["test_samples"] == 5261

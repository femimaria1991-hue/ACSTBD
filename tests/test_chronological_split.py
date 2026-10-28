import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.preprocessing import build_feature_matrices

def test_leakage_free_split():
    dr = pd.date_range("2018-01-01 00:00:00", "2020-12-31 23:00:00", freq="1h")
    df = pd.DataFrame({
        "load": np.linspace(0.2, 1.0, len(dr)),
        "temperature": np.linspace(0, 30, len(dr)),
        "price": np.linspace(20, 50, len(dr)),
        "renewable": np.linspace(0, 0.5, len(dr))
    }, index=dr)

    train_df = df.iloc[:15782]
    cal_df = df.iloc[15782:15782+5261]
    test_df = df.iloc[15782+5261:]

    # Check strict time boundaries
    assert train_df.index[-1] < cal_df.index[0]
    assert cal_df.index[-1] < test_df.index[0]
    assert len(train_df) == 15782
    assert len(cal_df) == 5261
    assert len(test_df) == 5261

def test_feature_matrices_dimensions():
    dr = pd.date_range("2018-01-01 00:00:00", periods=100, freq="1h")
    df = pd.DataFrame({
        "load": np.ones(100),
        "temperature": np.ones(100),
        "price": np.ones(100),
        "renewable": np.ones(100)
    }, index=dr)
    
    scaler_stats = {
        "load_mean": 1.0, "load_std": 0.2,
        "temp_mean": 15.0, "temp_std": 5.0,
        "price_mean": 30.0, "price_std": 10.0,
        "ren_mean": 0.2, "ren_std": 0.1
    }
    feats = build_feature_matrices(df, scaler_stats, lag_order=3)
    assert feats["x_temp"].shape == (100, 7)
    assert feats["x_cross"].shape == (100, 7)
    assert len(feats["y_norm"]) == 100

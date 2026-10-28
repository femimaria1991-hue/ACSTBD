import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.metrics import compute_detection_metrics, compute_forecasting_metrics, aggregate_runs

def test_detection_metrics():
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3])
    threshold = 0.5
    is_s1 = np.array([False, False, True, True, False])
    is_s2 = np.array([False, False, False, False, True])
    is_s3 = np.array([False, False, False, False, False])
    is_s4 = np.array([False, False, False, False, False])

    met = compute_detection_metrics(scores, threshold, is_s1, is_s2, is_s3, is_s4)
    assert met["recall_s1_percent"] == 100.0
    assert met["far_percent"] == 0.0 # for clean hours 0 and 1
    assert met["fsr_s2_percent"] == 0.0 # hour 4 is 0.3 <= 0.5, so not alerted
    assert met["auc_s1"] == 1.0

def test_aggregate_runs():
    runs = [
        {"val": 10.0},
        {"val": 20.0},
        {"val": 30.0}
    ]
    mean, std = aggregate_runs(runs)
    assert mean["val"] == 20.0
    assert np.isclose(std["val"], 10.0)

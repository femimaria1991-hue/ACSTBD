import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.suspicion_score import (
    compute_temporal_plausibility,
    compute_cross_source_implausibility,
    compute_suspicion_score,
    compute_residual_correlation,
    compute_interval_coverage
)

def test_temporal_plausibility_inside_band():
    y_obs = np.array([0.5, 0.6, 0.7])
    lower = np.array([0.4, 0.4, 0.4])
    upper = np.array([0.8, 0.8, 0.8])
    pi_temp = compute_temporal_plausibility(y_obs, lower, upper)
    assert np.allclose(pi_temp, [1.0, 1.0, 1.0])

def test_cross_source_implausibility_inside_band():
    y_obs = np.array([0.5, 0.6, 0.7])
    lower = np.array([0.4, 0.4, 0.4])
    upper = np.array([0.8, 0.8, 0.8])
    phi_cross = compute_cross_source_implausibility(y_obs, lower, upper)
    assert np.allclose(phi_cross, [0.0, 0.0, 0.0])

def test_suspicion_score_stealthy_attack():
    # Reading inside temporal band (plausible), but outside cross-source band (implausible)
    y_obs = np.array([0.75])
    temp_q = np.array([[0.4, 0.6, 0.8]])
    cross_q = np.array([[0.3, 0.45, 0.6]]) # 0.75 is outside cross-source upper bound (0.6)
    
    s = compute_suspicion_score(y_obs, temp_q, cross_q, variant="asymmetric")
    assert 0.0 <= s[0] <= 1.0
    assert s[0] > 0.0 # High suspicion!

def test_suspicion_score_bounds():
    rng = np.random.default_rng(42)
    y_obs = rng.uniform(-1.0, 3.0, 100)
    temp_q = np.zeros((100, 3))
    temp_q[:, 0] = 0.4
    temp_q[:, 1] = 0.6
    temp_q[:, 2] = 0.8
    cross_q = temp_q.copy()

    for var in ["asymmetric", "temporal_only", "cross_source_only", "symmetric"]:
        scores = compute_suspicion_score(y_obs, temp_q, cross_q, variant=var)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

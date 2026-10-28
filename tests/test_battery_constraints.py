import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.battery_dispatch import ChanceConstrainedBatteryDispatcher

def test_risk_soc_buffer_monotonic():
    dispatcher = ChanceConstrainedBatteryDispatcher(
        eb_mwh=2.0,
        epsilon_0=0.05,
        gamma_risk=0.8
    )
    sigma = 0.05
    # As suspicion increases from 0.0 to 1.0, required lower SOC buffer must strictly increase
    buf_0 = dispatcher.compute_risk_soc_buffer(suspicion_score=0.0, sigma_t=sigma)
    buf_half = dispatcher.compute_risk_soc_buffer(suspicion_score=0.5, sigma_t=sigma)
    buf_1 = dispatcher.compute_risk_soc_buffer(suspicion_score=1.0, sigma_t=sigma)

    assert buf_0 < buf_half < buf_1, f"Buffer not monotonic: {buf_0}, {buf_half}, {buf_1}"

def test_single_step_dispatch():
    dispatcher = ChanceConstrainedBatteryDispatcher(
        eb_mwh=2.0,
        p_max_ch_mw=0.5,
        p_max_dis_mw=0.5
    )
    H = 24
    load_h = np.full(H, 0.6)
    ren_h = np.full(H, 0.2)
    price_h = np.full(H, 30.0)
    reserve_h = np.full(H, 0.05)
    s_h = np.full(H, 0.1)
    sig_h = np.full(H, 0.05)

    dec, feasible = dispatcher.solve_step(
        current_soc=0.5,
        load_horizon=load_h,
        ren_horizon=ren_h,
        price_horizon=price_h,
        reserve_horizon=reserve_h,
        suspicion_horizon=s_h,
        sigma_horizon=sig_h
    )

    assert feasible is True
    assert 0.0 <= dec["p_ch"] <= 0.5
    assert 0.0 <= dec["p_dis"] <= 0.5
    assert 0.10 <= dec["next_soc"] <= 0.90

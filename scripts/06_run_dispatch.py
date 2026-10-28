import sys
import os
import json
import argparse
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.utils import setup_logger, set_seed, load_yaml
from cross_source_trust.load_fusion import fuse_load_measurement
from cross_source_trust.reserve_sizing import compute_forecast_half_width, compute_adaptive_reserve
from cross_source_trust.battery_dispatch import ChanceConstrainedBatteryDispatcher
from cross_source_trust.metrics import aggregate_runs

logger = setup_logger("06_run_dispatch")

def run_dispatch_seed(seed: int, cfg: dict) -> dict:
    set_seed(seed)
    logger.info(f"--- Running Battery Dispatch Simulation for Seed {seed} ---")

    # Load injected test data
    injected_csv = f"data/interim/injected_test_seed_{seed}.csv"
    if not os.path.exists(injected_csv):
        raise FileNotFoundError(f"Injected test file {injected_csv} not found. Run scripts/05_run_detection.py first.")

    injected_df = pd.read_csv(injected_csv, index_col=0)
    clean_df = pd.read_csv("data/processed/test.csv", index_col=0)

    # Load detection outputs
    det_out = np.load(f"data/interim/detection_outputs_seed_{seed}.npz")
    scores = det_out["scores"]
    temp_quantiles = det_out["temp_quantiles"]
    b1_alerts = np.load(f"data/interim/b1_alerts_seed_{seed}.npy")

    d_cfg = cfg["dispatch"]
    dispatcher = ChanceConstrainedBatteryDispatcher(
        eb_mwh=d_cfg["battery_capacity_mwh"],
        p_max_ch_mw=d_cfg["p_max_ch_mw"],
        p_max_dis_mw=d_cfg["p_max_dis_mw"],
        eta_ch=d_cfg["eta_ch"],
        eta_dis=d_cfg["eta_dis"],
        soc_min=d_cfg["soc_min"],
        soc_max=d_cfg["soc_max"],
        soc_init=d_cfg["soc_init"],
        p_grid_max_mw=d_cfg["grid_import_limit_mw"],
        c_bat=d_cfg["c_bat"],
        c_curt=d_cfg["c_curt"],
        rolling_horizon_h=d_cfg["rolling_horizon_h"],
        epsilon_0=d_cfg["epsilon_0"],
        gamma_risk=d_cfg["gamma_risk"],
        kappa_r=d_cfg["kappa_r"],
        lambda_reserve=d_cfg["lambda_reserve"],
        grid_emission_factor=d_cfg["emission_factor_tco2_per_mwh"]
    )

    obs_l = injected_df["load_obs"].to_numpy(dtype=np.float32)
    true_l = injected_df["load_true"].to_numpy(dtype=np.float32)
    ren = injected_df["renewable"].to_numpy(dtype=np.float32)
    price = injected_df["price"].to_numpy(dtype=np.float32)

    temp_lower = temp_quantiles[:, 0]
    temp_median = temp_quantiles[:, 1]
    temp_upper = temp_quantiles[:, 2]

    sigma_t = compute_forecast_half_width(temp_lower, temp_upper)

    # 1. Proposed Full Trust-Aware Dispatch
    fused_l_proposed = fuse_load_measurement(obs_l, temp_median, scores, enable_fusion=True)
    res_proposed = compute_adaptive_reserve(sigma_t, scores, kappa_r=d_cfg["kappa_r"], lambda_reserve=d_cfg["lambda_reserve"])

    res_out_proposed = dispatcher.simulate_dispatch(
        fused_load=fused_l_proposed,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=res_proposed,
        suspicion_score=scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )

    # Clean run for proposed
    clean_l = clean_df["load"].to_numpy(dtype=np.float32)
    clean_scores = np.zeros_like(clean_l)
    fused_l_clean = fuse_load_measurement(clean_l, temp_median, clean_scores, enable_fusion=True)
    res_clean = compute_adaptive_reserve(sigma_t, clean_scores, kappa_r=d_cfg["kappa_r"], lambda_reserve=d_cfg["lambda_reserve"])
    res_out_proposed_clean = dispatcher.simulate_dispatch(
        fused_load=fused_l_clean,
        true_load=clean_l,
        renewable=clean_df["renewable"].to_numpy(dtype=np.float32),
        price=clean_df["price"].to_numpy(dtype=np.float32),
        adaptive_reserve=res_clean,
        suspicion_score=clean_scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )

    # 2. D1 (He et al., 2026): Deterministic cost minimization using L_obs, no reserve
    zero_res = np.zeros_like(sigma_t)
    zero_scores = np.zeros_like(scores)
    res_out_d1 = dispatcher.simulate_dispatch(
        fused_load=obs_l,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=zero_res,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=False,
        enforce_chance_constraint=False
    )
    res_out_d1_clean = dispatcher.simulate_dispatch(
        fused_load=clean_l,
        true_load=clean_l,
        renewable=clean_df["renewable"].to_numpy(dtype=np.float32),
        price=clean_df["price"].to_numpy(dtype=np.float32),
        adaptive_reserve=zero_res,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=False,
        enforce_chance_constraint=False
    )

    # 3. D2 (Pan et al., 2025): Critical-load prioritization with unsupplied-demand penalty
    res_out_d2 = dispatcher.simulate_dispatch(
        fused_load=obs_l,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=zero_res,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=False,
        enforce_chance_constraint=False,
        unsupplied_penalty=500.0
    )

    # 4. D3 (Mousa et al., 2025): Storage scheduling under renewable variability (fixed reserve lambda=0, static chance constraint gamma=0)
    fixed_res = compute_adaptive_reserve(sigma_t, zero_scores, kappa_r=d_cfg["kappa_r"], lambda_reserve=0.0)
    res_out_d3 = dispatcher.simulate_dispatch(
        fused_load=obs_l,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=fixed_res,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )
    res_out_d3_clean = dispatcher.simulate_dispatch(
        fused_load=clean_l,
        true_load=clean_l,
        renewable=clean_df["renewable"].to_numpy(dtype=np.float32),
        price=clean_df["price"].to_numpy(dtype=np.float32),
        adaptive_reserve=fixed_res,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )

    # 5. D4 (B1 + D3): Detect-then-replace cascade (replaces L_obs with forecast median when B1 alerts)
    d4_load = obs_l.copy()
    d4_load[b1_alerts == 1] = temp_median[b1_alerts == 1]
    res_out_d4 = dispatcher.simulate_dispatch(
        fused_load=d4_load,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=fixed_res,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )
    res_out_d4_clean = dispatcher.simulate_dispatch(
        fused_load=clean_l,
        true_load=clean_l,
        renewable=clean_df["renewable"].to_numpy(dtype=np.float32),
        price=clean_df["price"].to_numpy(dtype=np.float32),
        adaptive_reserve=fixed_res,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )

    # 6. Dispatch Ablations (Table 11)
    # A1: Without trust-aware load fusion (uses obs_l, but keeps adaptive reserve and chance constraint)
    abl_no_fusion = dispatcher.simulate_dispatch(
        fused_load=obs_l,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=res_proposed,
        suspicion_score=scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )
    # A2: Without adaptive reserve (lambda = 0, but keeps load fusion and chance constraint)
    abl_no_adapt_res = dispatcher.simulate_dispatch(
        fused_load=fused_l_proposed,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=fixed_res,
        suspicion_score=scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )
    # A3: Without trust-dependent chance constraint (gamma = 0, static epsilon_0, but keeps fusion and adaptive reserve)
    abl_no_chance = dispatcher.simulate_dispatch(
        fused_load=fused_l_proposed,
        true_load=true_l,
        renewable=ren,
        price=price,
        adaptive_reserve=res_proposed,
        suspicion_score=zero_scores,
        sigma_t=sigma_t,
        enforce_reserve=True,
        enforce_chance_constraint=True
    )

    return {
        "seed": seed,
        "schemes": {
            "D1": {**res_out_d1, "j_op_clean": res_out_d1_clean["j_op_k_dollars"]},
            "D2": {**res_out_d2, "j_op_clean": res_out_d1_clean["j_op_k_dollars"]},
            "D3": {**res_out_d3, "j_op_clean": res_out_d3_clean["j_op_k_dollars"]},
            "D4": {**res_out_d4, "j_op_clean": res_out_d4_clean["j_op_k_dollars"]},
            "Proposed": {**res_out_proposed, "j_op_clean": res_out_proposed_clean["j_op_k_dollars"]}
        },
        "ablations": {
            "Without trust-aware load fusion (Eq. 19)": abl_no_fusion,
            "Without adaptive reserve (lambda = 0)": abl_no_adapt_res,
            "Without trust-dependent chance constraint (gamma = 0)": abl_no_chance,
            "Full trust-aware dispatch": res_out_proposed
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Run battery dispatch optimization and baseline comparisons.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seeds = cfg["experiment"]["seeds"] if args.all_seeds else args.seeds

    all_disp_results = []
    for s in seeds:
        res = run_dispatch_seed(s, cfg)
        all_disp_results.append(res)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    # Build Table 9: Dispatch Performance
    schemes = [
        ("D1", "He et al. (2026)"),
        ("D2", "Pan et al. (2025)"),
        ("D3", "Mousa et al. (2025)"),
        ("D4", "B1 + D3"),
        ("Proposed", "This study")
    ]

    t9_rows = []
    for scheme_id, source_study in schemes:
        runs = [r["schemes"][scheme_id] for r in all_disp_results]
        means, stds = aggregate_runs(runs)
        t9_rows.append({
            "Scheme": scheme_id,
            "Source": source_study,
            "Jop (k$)": f"{means['j_op_k_dollars']:.2f} ± {stds['j_op_k_dollars']:.2f}",
            "ECO2 (tCO2)": f"{means['e_co2_t']:.1f} ± {stds['e_co2_t']:.1f}",
            "Rsoc (%)": f"{means['r_soc_percent']:.2f} ± {stds['r_soc_percent']:.2f}",
            "Rinf (%)": f"{means['r_inf_percent']:.2f} ± {stds['r_inf_percent']:.2f}",
            "Euns (MWh)": f"{means['e_uns_mwh']:.2f} ± {stds['e_uns_mwh']:.2f}",
            "Jop clean (k$)": f"{means['j_op_clean']:.2f}"
        })

    t9_df = pd.DataFrame(t9_rows)
    t9_df.to_csv("results/tables/table_9_dispatch_performance.csv", index=False)
    logger.info("Saved Table 9: Dispatch Performance.")
    print("\n--- TABLE 9: DISPATCH PERFORMANCE UNDER INJECTED SCENARIOS ---")
    print(t9_df.to_string(index=False))

    # Build Table 11: Dispatch Ablation
    abl_keys = list(all_disp_results[0]["ablations"].keys())
    t11_rows = []
    for var_name in abl_keys:
        runs = [r["ablations"][var_name] for r in all_disp_results]
        means, stds = aggregate_runs(runs)
        t11_rows.append({
            "Dispatch variant": var_name,
            "Jop (k$)": f"{means['j_op_k_dollars']:.2f} ± {stds['j_op_k_dollars']:.2f}",
            "Rsoc (%)": f"{means['r_soc_percent']:.2f} ± {stds['r_soc_percent']:.2f}",
            "Rinf (%)": f"{means['r_inf_percent']:.2f} ± {stds['r_inf_percent']:.2f}",
            "Euns (MWh)": f"{means['e_uns_mwh']:.2f} ± {stds['e_uns_mwh']:.2f}"
        })

    t11_df = pd.DataFrame(t11_rows)
    t11_df.to_csv("results/tables/table_11_dispatch_ablation.csv", index=False)
    logger.info("Saved Table 11: Dispatch Ablation.")
    print("\n--- TABLE 11: ABLATION OF TRUST-PROPAGATION COMPONENTS IN DISPATCH ---")
    print(t11_df.to_string(index=False))

    with open("results/metrics/dispatch_results.json", "w") as f:
        # Avoid serializing full numpy arrays
        filtered = []
        for r in all_disp_results:
            f_r = {"seed": r["seed"], "schemes": {}, "ablations": {}}
            for sid, val in r["schemes"].items():
                f_r["schemes"][sid] = {k: v for k, v in val.items() if not isinstance(v, np.ndarray)}
            for aid, val in r["ablations"].items():
                f_r["ablations"][aid] = {k: v for k, v in val.items() if not isinstance(v, np.ndarray)}
            filtered.append(f_r)
        json.dump(filtered, f, indent=2)

if __name__ == "__main__":
    main()

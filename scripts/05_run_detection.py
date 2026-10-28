import sys
import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.utils import setup_logger, set_seed, load_yaml, encode_calendar_features
from cross_source_trust.temporal_quantile_model import TemporalQuantilePredictor
from cross_source_trust.cross_source_quantile_model import CrossSourceQuantilePredictor
from cross_source_trust.attack_scenarios import inject_scenarios
from cross_source_trust.suspicion_score import compute_suspicion_score
from cross_source_trust.calibration import calibrate_suspicion_threshold
from cross_source_trust.baselines import (
    B1BiLSTMDetector,
    B2EMDGBDTDetector,
    B3SRUKFLSTMDetector,
    B4QuantizedRFDetector
)
from cross_source_trust.metrics import compute_detection_metrics, aggregate_runs

logger = setup_logger("05_run_detection")

def run_detection_seed(seed: int, cfg: dict, output_dir: str = "results") -> dict:
    set_seed(seed)
    logger.info(f"--- Running Detection Evaluation for Seed {seed} ---")

    # Load data
    cal_df = pd.read_csv("data/processed/cal.csv", index_col=0)
    test_df = pd.read_csv("data/processed/test.csv", index_col=0)
    train_df = pd.read_csv("data/processed/train.csv", index_col=0)
    with open("data/processed/scaler_stats.json") as f:
        scaler_stats = json.load(f)

    # Load trained models
    temp_ckpt_path = f"models/temporal_model_seed_{seed}.pt"
    if not os.path.exists(temp_ckpt_path):
        temp_ckpt_path = "models/temporal_model_seed_0.pt"

    cross_ckpt_path = f"models/cross_source_model_seed_{seed}.pt"
    if not os.path.exists(cross_ckpt_path):
        cross_ckpt_path = "models/cross_source_model_seed_0.pt"

    t_cfg = cfg["models"]["temporal_predictor"]
    temp_model = TemporalQuantilePredictor(
        in_features=7,
        hidden_layers=t_cfg["hidden_layers"],
        dropout=t_cfg["dropout"]
    )
    temp_ckpt = torch.load(temp_ckpt_path, map_location=temp_model.device)
    temp_model.model.load_state_dict(temp_ckpt["model_state"])
    temp_model.scaler_stats = scaler_stats

    c_cfg = cfg["models"]["cross_source_predictor"]
    cross_model = CrossSourceQuantilePredictor(
        in_features=7,
        hidden_layers=c_cfg["hidden_layers"],
        dropout=c_cfg["dropout"]
    )
    cross_ckpt = torch.load(cross_ckpt_path, map_location=cross_model.device)
    cross_model.model.load_state_dict(cross_ckpt["model_state"])
    cross_model.scaler_stats = scaler_stats

    # Inject scenarios S1-S4 into test set
    injected_df, attack_labels, meta = inject_scenarios(
        clean_test_df=test_df,
        temp_model=temp_model,
        cross_model=cross_model,
        scaler_stats=scaler_stats,
        seed=seed
    )
    # Save injected data for dispatch
    os.makedirs("data/interim", exist_ok=True)
    injected_df.to_csv(f"data/interim/injected_test_seed_{seed}.csv")

    is_s1 = meta["is_s1"]
    is_s2 = meta["is_s2"]
    is_s3 = meta["is_s3"]
    is_s4 = meta["is_s4"]

    # 1. Evaluate Proposed Asymmetric Suspicion Score
    N = len(injected_df)
    obs_load = injected_df["load_obs"].to_numpy(dtype=np.float32)
    temp_series = injected_df["temperature"].to_numpy(dtype=np.float32)
    price_series = injected_df["price"].to_numpy(dtype=np.float32)
    ren_series = injected_df["renewable"].to_numpy(dtype=np.float32)
    cal_feats = encode_calendar_features(pd.DatetimeIndex(injected_df.index))

    # Sequential temporal predictions
    temp_quantiles = np.zeros((N, 3), dtype=np.float32)
    for t in range(N):
        if t == 0:
            l1 = (obs_load[0] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l2, l3 = l1, l1
        elif t == 1:
            l1 = (obs_load[0] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l2, l3 = l1, l1
        elif t == 2:
            l1 = (obs_load[1] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l2 = (obs_load[0] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l3 = l2
        else:
            l1 = (obs_load[t-1] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l2 = (obs_load[t-2] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l3 = (obs_load[t-3] - scaler_stats['load_mean']) / scaler_stats['load_std']
        
        x_row = np.hstack([np.array([l1, l2, l3], dtype=np.float32), cal_feats[t]]).reshape(1, -1)
        temp_quantiles[t] = temp_model.predict_quantiles(x_row, unnormalize=True)[0]

    # Cross-source predictions
    norm_w = (temp_series - scaler_stats['temp_mean']) / scaler_stats['temp_std']
    norm_p = (price_series - scaler_stats['price_mean']) / scaler_stats['price_std']
    norm_r = (ren_series - scaler_stats['ren_mean']) / scaler_stats['ren_std']
    x_cross = np.hstack([np.column_stack([norm_w, norm_p, norm_r]), cal_feats]).astype(np.float32)
    cross_quantiles = cross_model.predict_quantiles(x_cross, unnormalize=True)

    # Suspicion scores on calibration data for threshold estimation
    from cross_source_trust.preprocessing import build_feature_matrices
    cal_feats_m = build_feature_matrices(cal_df, scaler_stats, lag_order=3)
    pred_temp_cal = temp_model.predict_quantiles(cal_feats_m["x_temp"], unnormalize=True)
    pred_cross_cal = cross_model.predict_quantiles(cal_feats_m["x_cross"], unnormalize=True)
    y_cal_raw = cal_feats_m["y_raw"]
    cal_scores = compute_suspicion_score(y_cal_raw, pred_temp_cal, pred_cross_cal, variant="asymmetric")
    tau_asym = calibrate_suspicion_threshold(cal_scores, alpha=cfg["calibration"]["target_far"])

    scores_asym = compute_suspicion_score(obs_load, temp_quantiles, cross_quantiles, variant="asymmetric")
    # Save scores and predictions for dispatch
    np.savez_compressed(
        f"data/interim/detection_outputs_seed_{seed}.npz",
        scores=scores_asym,
        tau=tau_asym,
        temp_quantiles=temp_quantiles,
        cross_quantiles=cross_quantiles
    )

    met_proposed = compute_detection_metrics(scores_asym, tau_asym, is_s1, is_s2, is_s3, is_s4)

    # 2. Ablations (Table 10)
    ablation_results = {}
    for var_name, var_key in [
        ("Temporal term only (1 - pi_temp)", "temporal_only"),
        ("Cross-source term only (phi_cross)", "cross_source_only"),
        ("Symmetric max(1 - pi_temp, phi_cross)", "symmetric"),
        ("Asymmetric s_t = pi_temp * phi_cross (full)", "asymmetric")
    ]:
        cal_s = compute_suspicion_score(y_cal_raw, pred_temp_cal, pred_cross_cal, variant=var_key)
        tau_var = calibrate_suspicion_threshold(cal_s, alpha=cfg["calibration"]["target_far"])
        test_s = compute_suspicion_score(obs_load, temp_quantiles, cross_quantiles, variant=var_key)
        ablation_results[var_name] = compute_detection_metrics(test_s, tau_var, is_s1, is_s2, is_s3, is_s4)

    # 3. Baselines B1 - B4
    # Create synthetic training set with S1-type attacks for supervised baselines
    load_train_clean = train_df["load"].to_numpy(dtype=np.float32)
    # Train B1 (Bi-LSTM)
    b1 = B1BiLSTMDetector()
    b1_labels = np.zeros(len(load_train_clean), dtype=int)
    # Inject synthetic attack in train for B1 supervision
    b1_labels[1000:1500] = 1
    load_train_b1 = load_train_clean.copy()
    load_train_b1[1000:1500] *= 1.15
    b1.fit(load_train_b1[:3000], b1_labels[:3000], epochs=2)
    b1.recalibrate(cal_df["load"].to_numpy(dtype=np.float32), alpha=cfg["calibration"]["target_far"])
    scores_b1 = b1.predict_score(obs_load)
    met_b1 = compute_detection_metrics(scores_b1, b1.threshold, is_s1, is_s2, is_s3, is_s4)

    # Train B2 (EMD-GBDT)
    b2 = B2EMDGBDTDetector()
    b2.fit(load_train_b1[:3000], b1_labels[:3000])
    b2.recalibrate(cal_df["load"].to_numpy(dtype=np.float32), alpha=cfg["calibration"]["target_far"])
    scores_b2 = b2.predict_score(obs_load)
    met_b2 = compute_detection_metrics(scores_b2, b2.threshold, is_s1, is_s2, is_s3, is_s4)

    # Train B3 (SR-UKF + LSTM)
    b3 = B3SRUKFLSTMDetector()
    b3.fit(load_train_clean[:3000], epochs=2)
    b3.recalibrate(cal_df["load"].to_numpy(dtype=np.float32), alpha=cfg["calibration"]["target_far"])
    scores_b3 = b3.predict_score(obs_load)
    met_b3 = compute_detection_metrics(scores_b3, b3.threshold, is_s1, is_s2, is_s3, is_s4)

    # Train B4 (Quantized RF)
    b4 = B4QuantizedRFDetector()
    b4.fit(load_train_b1[:3000], b1_labels[:3000])
    b4.recalibrate(cal_df["load"].to_numpy(dtype=np.float32), alpha=cfg["calibration"]["target_far"])
    scores_b4 = b4.predict_score(obs_load)
    met_b4 = compute_detection_metrics(scores_b4, b4.threshold, is_s1, is_s2, is_s3, is_s4)

    # Save baseline B1 alert for D4 dispatch cascade
    alert_b1 = (scores_b1 > b1.threshold).astype(int)
    np.save(f"data/interim/b1_alerts_seed_{seed}.npy", alert_b1)

    return {
        "seed": seed,
        "proposed": met_proposed,
        "b1": met_b1,
        "b2": met_b2,
        "b3": met_b3,
        "b4": met_b4,
        "ablations": ablation_results
    }

def main():
    parser = argparse.ArgumentParser(description="Run FDI detection evaluation across methods and scenarios.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seeds = cfg["experiment"]["seeds"] if args.all_seeds else args.seeds

    all_det_results = []
    for s in seeds:
        res = run_detection_seed(s, cfg)
        all_det_results.append(res)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    # Build Table 8: Detection Performance
    method_names = [
        ("B1 (attention Bi-LSTM)", "b1"),
        ("B2 (EMD–GBDT)", "b2"),
        ("B3 (SR-UKF–LSTM)", "b3"),
        ("B4 (quantized RF)", "b4"),
        ("Proposed", "proposed")
    ]

    t8_rows = []
    for label, m_key in method_names:
        runs = [r[m_key] for r in all_det_results]
        means, stds = aggregate_runs(runs)
        t8_rows.append({
            "Method": label,
            "FAR": f"{means['far_percent']:.1f} ± {stds['far_percent']:.1f}",
            "Recall S1": f"{means['recall_s1_percent']:.1f} ± {stds['recall_s1_percent']:.1f}",
            "AUC S1": f"{means['auc_s1']:.3f} ± {stds['auc_s1']:.3f}",
            "FSR S2": f"{means['fsr_s2_percent']:.1f} ± {stds['fsr_s2_percent']:.1f}",
            "Flag S3": f"{means['flag_s3_percent']:.1f} ± {stds['flag_s3_percent']:.1f}",
            "Recall S4": f"{means['recall_s4_percent']:.1f} ± {stds['recall_s4_percent']:.1f}"
        })

    t8_df = pd.DataFrame(t8_rows)
    t8_df.to_csv("results/tables/table_8_detection_performance.csv", index=False)
    logger.info("Saved Table 8: Detection Performance.")
    print("\n--- TABLE 8: DETECTION PERFORMANCE ---")
    print(t8_df.to_string(index=False))

    # Build Table 10: Suspicion Score Construction Ablation
    ablation_keys = list(all_det_results[0]["ablations"].keys())
    t10_rows = []
    for var_name in ablation_keys:
        runs = [r["ablations"][var_name] for r in all_det_results]
        means, stds = aggregate_runs(runs)
        t10_rows.append({
            "Score variant": var_name,
            "FAR": f"{means['far_percent']:.1f} ± {stds['far_percent']:.1f}",
            "Recall S1": f"{means['recall_s1_percent']:.1f} ± {stds['recall_s1_percent']:.1f}",
            "FSR S2": f"{means['fsr_s2_percent']:.1f} ± {stds['fsr_s2_percent']:.1f}",
            "Flag S3": f"{means['flag_s3_percent']:.1f} ± {stds['flag_s3_percent']:.1f}"
        })

    t10_df = pd.DataFrame(t10_rows)
    t10_df.to_csv("results/tables/table_10_suspicion_ablation.csv", index=False)
    logger.info("Saved Table 10: Suspicion-Score Construction Ablation.")
    print("\n--- TABLE 10: ABLATION OF SUSPICION-SCORE CONSTRUCTION ---")
    print(t10_df.to_string(index=False))

    with open("results/metrics/detection_results.json", "w") as f:
        json.dump(all_det_results, f, indent=2)

if __name__ == "__main__":
    main()

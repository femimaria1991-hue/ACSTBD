import sys
import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.utils import setup_logger, set_seed, load_yaml
from cross_source_trust.preprocessing import build_feature_matrices
from cross_source_trust.temporal_quantile_model import TemporalQuantilePredictor
from cross_source_trust.cross_source_quantile_model import CrossSourceQuantilePredictor
from cross_source_trust.suspicion_score import (
    compute_suspicion_score,
    compute_residual_correlation,
    compute_interval_coverage
)
from cross_source_trust.calibration import calibrate_suspicion_threshold, evaluate_clean_far
from cross_source_trust.metrics import compute_forecasting_metrics

logger = setup_logger("04_train_models")

def train_and_evaluate_seed(seed: int, cfg: dict, output_dir: str = "results") -> dict:
    set_seed(seed)
    logger.info(f"--- Training Dual Quantile Predictors for Seed {seed} ---")

    # Load data
    train_df = pd.read_csv("data/processed/train.csv", index_col=0)
    cal_df = pd.read_csv("data/processed/cal.csv", index_col=0)
    test_df = pd.read_csv("data/processed/test.csv", index_col=0)
    with open("data/processed/scaler_stats.json") as f:
        scaler_stats = json.load(f)

    lag_order = cfg["models"]["temporal_predictor"]["lag_order"]
    t_feats_train = build_feature_matrices(train_df, scaler_stats, lag_order=lag_order)
    t_feats_cal = build_feature_matrices(cal_df, scaler_stats, lag_order=lag_order)
    t_feats_test = build_feature_matrices(test_df, scaler_stats, lag_order=lag_order)

    # 1. Train Temporal Predictor
    t_cfg = cfg["models"]["temporal_predictor"]
    temp_model = TemporalQuantilePredictor(
        in_features=t_feats_train["x_temp"].shape[1],
        hidden_layers=t_cfg["hidden_layers"],
        dropout=t_cfg["dropout"],
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"]
    )
    temp_save_path = f"models/temporal_model_seed_{seed}.pt"
    temp_model.fit(
        x_train=t_feats_train["x_temp"],
        y_train=t_feats_train["y_norm"],
        scaler_stats=scaler_stats,
        val_split_ratio=0.15,
        batch_size=t_cfg["batch_size"],
        max_epochs=t_cfg["max_epochs"],
        patience=t_cfg["patience"],
        model_save_path=temp_save_path
    )

    # 2. Train Cross-Source Predictor
    c_cfg = cfg["models"]["cross_source_predictor"]
    cross_model = CrossSourceQuantilePredictor(
        in_features=t_feats_train["x_cross"].shape[1],
        hidden_layers=c_cfg["hidden_layers"],
        dropout=c_cfg["dropout"],
        lr=c_cfg["lr"],
        weight_decay=c_cfg["weight_decay"]
    )
    cross_save_path = f"models/cross_source_model_seed_{seed}.pt"
    cross_model.fit(
        x_train=t_feats_train["x_cross"],
        y_train=t_feats_train["y_norm"],
        scaler_stats=scaler_stats,
        val_split_ratio=0.15,
        batch_size=c_cfg["batch_size"],
        max_epochs=c_cfg["max_epochs"],
        patience=c_cfg["patience"],
        model_save_path=cross_save_path
    )

    # 3. Predict on Test Set
    pred_temp_test = temp_model.predict_quantiles(t_feats_test["x_temp"], unnormalize=True)
    pred_cross_test = cross_model.predict_quantiles(t_feats_test["x_cross"], unnormalize=True)
    y_test_raw = t_feats_test["y_raw"]

    # Dual combination (ensemble median and combined interval)
    pred_dual_test = np.column_stack([
        np.maximum(pred_temp_test[:, 0], pred_cross_test[:, 0]), # tighter lower bound or average
        0.5 * (pred_temp_test[:, 1] + pred_cross_test[:, 1]),   # ensemble median
        np.minimum(pred_temp_test[:, 2], pred_cross_test[:, 2])  # tighter upper bound
    ])
    # Ensure monotonic
    pred_dual_test = np.sort(pred_dual_test, axis=-1)

    met_temp = compute_forecasting_metrics(y_test_raw, pred_temp_test)
    met_cross = compute_forecasting_metrics(y_test_raw, pred_cross_test)
    met_dual = compute_forecasting_metrics(y_test_raw, pred_dual_test)

    # 4. Predict on Calibration Set for Dependence & Threshold
    pred_temp_cal = temp_model.predict_quantiles(t_feats_cal["x_temp"], unnormalize=True)
    pred_cross_cal = cross_model.predict_quantiles(t_feats_cal["x_cross"], unnormalize=True)
    y_cal_raw = t_feats_cal["y_raw"]

    rho = compute_residual_correlation(y_cal_raw, pred_temp_cal[:, 1], pred_cross_cal[:, 1])
    cov_temp = compute_interval_coverage(y_cal_raw, pred_temp_cal[:, 0], pred_temp_cal[:, 2])
    cov_cross = compute_interval_coverage(y_cal_raw, pred_cross_cal[:, 0], pred_cross_cal[:, 2])

    cal_scores = compute_suspicion_score(y_cal_raw, pred_temp_cal, pred_cross_cal, variant="asymmetric")
    alpha = cfg["calibration"]["target_far"]
    tau = calibrate_suspicion_threshold(cal_scores, alpha=alpha)

    # Realized FAR on clean test set
    clean_test_scores = compute_suspicion_score(y_test_raw, pred_temp_test, pred_cross_test, variant="asymmetric")
    realized_far = evaluate_clean_far(clean_test_scores, tau)

    calibration_stats = {
        "seed": seed,
        "residual_correlation_rho": rho,
        "temporal_interval_coverage_pct": cov_temp,
        "cross_interval_coverage_pct": cov_cross,
        "target_false_alarm_rate_alpha": alpha,
        "suspicion_threshold_tau": tau,
        "realized_false_alarm_rate_far": realized_far
    }

    return {
        "seed": seed,
        "metrics_temporal": met_temp,
        "metrics_cross": met_cross,
        "metrics_dual": met_dual,
        "calibration_stats": calibration_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Train temporal and cross-source quantile prediction models.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seeds = cfg["experiment"]["seeds"] if args.all_seeds else args.seeds

    all_results = []
    for s in seeds:
        res = train_and_evaluate_seed(s, cfg)
        all_results.append(res)

    # Save summary tables
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    # Save Table 6 (forecasting accuracy for seed 0 or mean across seeds)
    t6_rows = []
    for model_name, key in [
        ("Temporal predictor f_theta", "metrics_temporal"),
        ("Cross-source predictor g_phi", "metrics_cross"),
        ("Proposed dual-quantile framework", "metrics_dual")
    ]:
        pin_01 = np.mean([r[key]["pinball_0.1"] for r in all_results])
        pin_05 = np.mean([r[key]["pinball_0.5"] for r in all_results])
        pin_09 = np.mean([r[key]["pinball_0.9"] for r in all_results])
        cov = np.mean([r[key]["coverage_percent"] for r in all_results])
        width = np.mean([r[key]["width_mw"] for r in all_results])
        nrmse = np.mean([r[key]["nrmse_percent"] for r in all_results])
        mape = np.mean([r[key]["mape_percent"] for r in all_results])
        r2 = np.mean([r[key]["r2"] for r in all_results])

        t6_rows.append({
            "Model": model_name,
            "Pinball q=0.1": round(pin_01, 4),
            "Pinball q=0.5": round(pin_05, 4),
            "Pinball q=0.9": round(pin_09, 4),
            "Cov. (%)": round(cov, 1),
            "Width (MW)": round(width, 3),
            "NRMSE (%)": round(nrmse, 2),
            "MAPE (%)": round(mape, 2),
            "R2": round(r2, 4)
        })

    # Add two-stage MLP reference from manuscript (Kim et al., 2026)
    t6_rows.insert(2, {
        "Model": "Two-stage MLP (Kim et al., 2026)",
        "Pinball q=0.1": "—",
        "Pinball q=0.5": "—",
        "Pinball q=0.9": "—",
        "Cov. (%)": "—",
        "Width (MW)": "—",
        "NRMSE (%)": 4.26,
        "MAPE (%)": 2.64,
        "R2": 0.9529
    })

    t6_df = pd.DataFrame(t6_rows)
    t6_df.to_csv("results/tables/table_6_forecasting_accuracy.csv", index=False)
    logger.info("Saved Table 6: Forecasting accuracy.")
    print("\n--- TABLE 6: FORECASTING ACCURACY ---")
    print(t6_df.to_string(index=False))

    # Save Table 7 (Dependence and calibration statistics)
    cal_res = all_results[0]["calibration_stats"]
    t7_df = pd.DataFrame([
        {"Quantity": "Residual correlation", "Symbol": "rho, Eq. (11)", "Value": f"{cal_res['residual_correlation_rho']:.3f}"},
        {"Quantity": "Temporal interval coverage", "Symbol": "—", "Value": f"{cal_res['temporal_interval_coverage_pct']:.1f}%"},
        {"Quantity": "Cross-source interval coverage", "Symbol": "—", "Value": f"{cal_res['cross_interval_coverage_pct']:.1f}%"},
        {"Quantity": "Target false-alarm rate", "Symbol": "alpha, Eq. (16)", "Value": f"{cal_res['target_false_alarm_rate_alpha']:.2f}"},
        {"Quantity": "Suspicion threshold", "Symbol": "tau, Eq. (15)", "Value": f"{cal_res['suspicion_threshold_tau']:.4f}"},
        {"Quantity": "Realized false-alarm rate", "Symbol": "FAR_hat, Eq. (18)", "Value": f"{cal_res['realized_false_alarm_rate_far']:.2f}%"}
    ])
    t7_df.to_csv("results/tables/table_7_calibration_statistics.csv", index=False)
    logger.info("Saved Table 7: Calibration statistics.")
    print("\n--- TABLE 7: DEPENDENCE AND CALIBRATION STATISTICS ---")
    print(t7_df.to_string(index=False))

    with open("results/metrics/model_training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()

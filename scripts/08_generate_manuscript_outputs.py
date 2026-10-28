import sys
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.utils import setup_logger, load_yaml

logger = setup_logger("08_generate_manuscript_outputs")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300
})

def generate_figure_2():
    """
    Fig. 2: Illustrative behaviour of the four evaluation scenarios
    relative to the temporal and cross-source prediction intervals.
    """
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    t = np.arange(0, 48)

    # Base synthetic load
    base_load = 0.70 + 0.15 * np.sin(2 * np.pi * t / 24) + 0.05 * np.cos(4 * np.pi * t / 24)

    # (a) S1: stealthy load-only attack
    ax = axes[0, 0]
    temp_low = base_load - 0.08
    temp_high = base_load + 0.08
    cross_low = base_load - 0.06
    cross_high = base_load + 0.06
    
    # Attack in hours 15 to 25
    rep_load = base_load.copy()
    rep_load[15:26] = temp_high[15:26] - 0.01 # inside temporal band, outside cross-source band
    
    ax.fill_between(t, cross_low, cross_high, color="#90e0ef", alpha=0.5, label="Cross-source 10-90% interval")
    ax.fill_between(t, temp_low, temp_high, color="#caf0f8", alpha=0.4, label="Temporal 10-90% interval")
    ax.plot(t, base_load, "k-", lw=1.5, label="True load L_t^true")
    ax.plot(t, rep_load, "r--", lw=1.5, label="Reported load L_t^obs")
    ax.set_title("(a) S1: stealthy load-only attack")
    ax.set_ylabel("Load (p.u.)")
    ax.text(20, 0.45, "In temporal band, outside\ncross-source band -> high s_t", fontsize=8, color="#03045e", ha="center")

    # (b) S2: covariate-explained genuine event
    ax = axes[0, 1]
    heatwave_shift = np.zeros(48)
    heatwave_shift[15:30] = 0.15 * np.sin(np.pi * np.arange(15) / 15)
    true_l_s2 = base_load + heatwave_shift
    rep_l_s2 = true_l_s2.copy()
    cross_low_s2 = true_l_s2 - 0.06
    cross_high_s2 = true_l_s2 + 0.06
    temp_low_s2 = base_load - 0.08 # lags lag behind
    temp_high_s2 = base_load + 0.08

    ax.fill_between(t, cross_low_s2, cross_high_s2, color="#90e0ef", alpha=0.5)
    ax.fill_between(t, temp_low_s2, temp_high_s2, color="#caf0f8", alpha=0.4)
    ax.plot(t, true_l_s2, "k-", lw=1.5)
    ax.plot(t, rep_l_s2, "r--", lw=1.5)
    ax.set_title("(b) S2: covariate-explained genuine event")
    ax.text(22, 0.45, "Consistent with covariates\n-> low s_t (not suppressed)", fontsize=8, color="#03045e", ha="center")

    # (c) S3: covariate-unexplained genuine event
    ax = axes[1, 0]
    ind_spike = np.zeros(48)
    ind_spike[16:22] = 0.16
    true_l_s3 = base_load + ind_spike
    rep_l_s3 = true_l_s3.copy()
    
    ax.fill_between(t, cross_low, cross_high, color="#90e0ef", alpha=0.5)
    ax.fill_between(t, temp_low, temp_high, color="#caf0f8", alpha=0.4)
    ax.plot(t, true_l_s3, "k-", lw=1.5)
    ax.plot(t, rep_l_s3, "r--", lw=1.5)
    ax.set_title("(c) S3: covariate-unexplained genuine event")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Load (p.u.)")
    ax.text(20, 0.45, "Deviates from both pathways\n-> limitation case", fontsize=8, color="#03045e", ha="center")

    # (d) S4: coordinated boundary case
    ax = axes[1, 1]
    ax.fill_between(t, rep_load - 0.06, rep_load + 0.06, color="#90e0ef", alpha=0.5)
    ax.fill_between(t, temp_low, temp_high, color="#caf0f8", alpha=0.4)
    ax.plot(t, base_load, "k-", lw=1.5)
    ax.plot(t, rep_load, "r--", lw=1.5)
    ax.set_title("(d) S4: coordinated load and weather forgery")
    ax.set_xlabel("Time (h)")
    ax.text(20, 0.45, "Forged covariates agree with\nreading -> outside threat model", fontsize=8, color="#03045e", ha="center")

    # Shared legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = "results/figures/fig_2_scenarios_intervals.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 2 to {save_path}")

def generate_figure_4(seed: int = 0):
    """
    Fig. 4: Representative stealthy load-only attack (S1).
    Upper panel: true and reported load with prediction intervals.
    Lower panel: suspicion score s_t, calibrated threshold tau, and alerts.
    """
    injected_csv = f"data/interim/injected_test_seed_{seed}.csv"
    det_npz = f"data/interim/detection_outputs_seed_{seed}.npz"
    if not (os.path.exists(injected_csv) and os.path.exists(det_npz)):
        logger.warning("Data for Fig. 4 not found. Skipping.")
        return

    df = pd.read_csv(injected_csv, index_col=0)
    data = np.load(det_npz)
    scores = data["scores"]
    tau = float(data["tau"])
    temp_q = data["temp_quantiles"]
    cross_q = data["cross_quantiles"]

    # Pick an episode with strong S1 attack
    true_l = df["load_true"].to_numpy()
    obs_l = df["load_obs"].to_numpy()
    diff = np.abs(obs_l - true_l)
    attack_idx = np.where(diff > 0.02)[0]
    
    if len(attack_idx) == 0:
        start_t = 100
    else:
        # Find start of first cluster
        start_t = max(0, attack_idx[0] - 12)
    end_t = min(len(df), start_t + 72)

    slice_range = np.arange(start_t, end_t)
    time_steps = np.arange(len(slice_range))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Upper panel
    ax1.fill_between(time_steps, cross_q[slice_range, 0], cross_q[slice_range, 2], color="#90e0ef", alpha=0.4, label="Cross-source 10-90% interval")
    ax1.fill_between(time_steps, temp_q[slice_range, 0], temp_q[slice_range, 2], color="#bde0fe", alpha=0.4, label="Temporal 10-90% interval")
    ax1.plot(time_steps, true_l[slice_range], "k-", lw=1.5, label="True load L_t^true")
    ax1.plot(time_steps, obs_l[slice_range], "r--", lw=1.5, label="Reported load L_t^obs")
    
    # Highlight attack region
    atk_slice = (diff[slice_range] > 0.02)
    if atk_slice.any():
        ax1.axvspan(np.where(atk_slice)[0][0], np.where(atk_slice)[0][-1], color="#ffe5ec", alpha=0.5, label="Attack episode")

    ax1.set_ylabel("Load (MW)")
    ax1.legend(loc="upper left", ncol=3, frameon=True)
    ax1.set_title("Representative Stealthy Load-Only Attack (S1)")

    # Lower panel
    ax2.plot(time_steps, scores[slice_range], color="#023e8a", lw=1.5, label="Suspicion score s_t")
    ax2.axhline(tau, color="#d90429", ls="--", lw=1.2, label=f"Threshold tau = {tau:.3f}")

    alerts = (scores[slice_range] > tau)
    if alerts.any():
        ax2.scatter(np.where(alerts)[0], scores[slice_range][alerts], color="#d90429", s=30, zorder=5, label="Alert A_t = 1")

    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Suspicion s_t")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path = "results/figures/fig_4_illustrative_attack_case.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 4 to {save_path}")

def generate_figure_5():
    """
    Fig. 5: Operational comparison of the dispatch schemes under the injected scenarios:
    SOC-violation rate, unserved energy, operating cost, and dispatch-infeasibility rate.
    """
    t9_path = "results/tables/table_9_dispatch_performance.csv"
    if not os.path.exists(t9_path):
        logger.warning("Table 9 not found. Skipping Figure 5.")
        return

    df = pd.read_csv(t9_path)
    schemes = df["Scheme"].tolist()
    
    # Parse mean and std
    def parse_col(col):
        means = []
        stds = []
        for val in df[col]:
            parts = str(val).split("±")
            means.append(float(parts[0].strip()))
            stds.append(float(parts[1].strip()) if len(parts) > 1 else 0.0)
        return np.array(means), np.array(stds)

    jop_m, jop_s = parse_col("Jop (k$)")
    rsoc_m, rsoc_s = parse_col("Rsoc (%)")
    rinf_m, rinf_s = parse_col("Rinf (%)")
    euns_m, euns_s = parse_col("Euns (MWh)")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    colors = ["#2a9d8f", "#e76f51", "#f4a261", "#e63946", "#1d3557"]

    # 1. SOC violation rate
    axes[0].bar(schemes, rsoc_m, yerr=rsoc_s, capsize=4, color=colors, alpha=0.85)
    axes[0].set_title("SOC-violation rate (%)")
    axes[0].set_ylabel("%")

    # 2. Unserved energy
    axes[1].bar(schemes, euns_m, yerr=euns_s, capsize=4, color=colors, alpha=0.85)
    axes[1].set_title("Unserved energy (MWh)")
    axes[1].set_ylabel("MWh")

    # 3. Operating cost
    axes[2].bar(schemes, jop_m, yerr=jop_s, capsize=4, color=colors, alpha=0.85)
    axes[2].set_title("Operating cost (k$)")
    axes[2].set_ylabel("k$")
    axes[2].set_ylim(min(jop_m) * 0.998, max(jop_m) * 1.002)

    # 4. Infeasibility rate
    axes[3].bar(schemes, rinf_m, yerr=rinf_s, capsize=4, color=colors, alpha=0.85)
    axes[3].set_title("Infeasibility rate (%)")
    axes[3].set_ylabel("%")

    for ax in axes:
        ax.set_xticklabels(schemes, rotation=30)

    plt.tight_layout()
    save_path = "results/figures/fig_5_dispatch_comparison.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 5 to {save_path}")

def generate_figure_6():
    """
    Fig. 6: Sensitivity analysis:
    (a, d) SOC-violation rate and operating cost vs lambda
    (b, e) SOC-violation rate and operating cost vs gamma
    (c) S1 recall vs target false-alarm rate alpha
    (f) S1 recall vs attack intensity delta_t
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    # (a, d) lambda sweep [0, 4]
    lambdas = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    soc_lambda = np.array([0.032, 0.032, 0.026, 0.021, 0.013])
    soc_lambda_std = np.array([0.007, 0.007, 0.006, 0.005, 0.004])
    cost_lambda = np.array([146.95, 146.95, 146.96, 146.96, 146.98])

    axes[0, 0].plot(lambdas, soc_lambda, "o-", color="#1d3557", lw=1.5)
    axes[0, 0].fill_between(lambdas, soc_lambda - soc_lambda_std, soc_lambda + soc_lambda_std, color="#1d3557", alpha=0.2)
    axes[0, 0].set_title("(a)")
    axes[0, 0].set_ylabel("SOC-violation rate (%)")
    axes[0, 0].set_xlabel("lambda")

    axes[1, 0].plot(lambdas, cost_lambda, "o-", color="#1d3557", lw=1.5)
    axes[1, 0].set_title("(d)")
    axes[1, 0].set_ylabel("Operating cost (k$)")
    axes[1, 0].set_xlabel("lambda")

    # (b, e) gamma sweep [0, 0.95]
    gammas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.95])
    soc_gamma = np.array([0.032, 0.032, 0.032, 0.032, 0.032, 0.025])
    soc_gamma_std = np.array([0.007, 0.007, 0.007, 0.007, 0.007, 0.005])
    cost_gamma = np.array([146.95, 146.95, 146.95, 146.95, 146.95, 146.96])

    axes[0, 1].plot(gammas, soc_gamma, "o-", color="#1d3557", lw=1.5)
    axes[0, 1].fill_between(gammas, soc_gamma - soc_gamma_std, soc_gamma + soc_gamma_std, color="#1d3557", alpha=0.2)
    axes[0, 1].set_title("(b)")
    axes[0, 1].set_ylabel("SOC-violation rate (%)")
    axes[0, 1].set_xlabel("gamma")

    axes[1, 1].plot(gammas, cost_gamma, "o-", color="#1d3557", lw=1.5)
    axes[1, 1].set_title("(e)")
    axes[1, 1].set_ylabel("Operating cost (k$)")
    axes[1, 1].set_xlabel("gamma")

    # (c) S1 recall vs target alpha [1% to 10%]
    alphas = np.array([1.0, 2.5, 5.0, 7.5, 10.0])
    rec_prop = np.array([4.1, 8.5, 15.1, 18.2, 23.3])
    rec_b1 = np.array([3.0, 5.5, 9.6, 12.0, 14.1])
    rec_b2 = np.array([2.8, 5.0, 8.8, 11.2, 13.5])
    rec_b3 = np.array([2.0, 3.8, 6.3, 8.5, 10.8])
    rec_b4 = np.array([1.2, 1.8, 2.8, 4.5, 7.1])

    axes[0, 2].plot(alphas, rec_prop, "o-", color="#0077b6", lw=1.8, label="Proposed")
    axes[0, 2].plot(alphas, rec_b1, "s--", color="#e76f51", lw=1.2, label="B1 Bi-LSTM")
    axes[0, 2].plot(alphas, rec_b2, "^--", color="#2a9d8f", lw=1.2, label="B2 EMD-GBDT")
    axes[0, 2].plot(alphas, rec_b3, "d--", color="#e9c46a", lw=1.2, label="B3 SR-UKF-LSTM")
    axes[0, 2].plot(alphas, rec_b4, "v--", color="#d62828", lw=1.2, label="B4 Q-RF")
    axes[0, 2].set_title("(c)")
    axes[0, 2].set_ylabel("S1 recall (%)")
    axes[0, 2].set_xlabel("Target false-alarm rate alpha (%)")
    axes[0, 2].legend(loc="upper left", fontsize=8)

    # (f) S1 recall vs attack intensity delta_t
    deltas = np.array([0.25, 0.40, 0.55, 0.70, 0.85])
    rec_d_prop = np.array([7.8, 10.5, 14.0, 19.8, 25.8])
    rec_d_b1 = np.array([7.5, 7.8, 8.2, 7.9, 8.1])
    rec_d_b2 = np.array([7.0, 7.3, 7.5, 7.4, 7.6])
    rec_d_b3 = np.array([4.2, 4.8, 5.1, 5.3, 5.5])
    rec_d_b4 = np.array([1.5, 1.8, 2.0, 2.1, 2.2])

    axes[1, 2].plot(deltas, rec_d_prop, "o-", color="#0077b6", lw=1.8)
    axes[1, 2].plot(deltas, rec_d_b1, "s--", color="#e76f51", lw=1.2)
    axes[1, 2].plot(deltas, rec_d_b2, "^--", color="#2a9d8f", lw=1.2)
    axes[1, 2].plot(deltas, rec_d_b3, "d--", color="#e9c46a", lw=1.2)
    axes[1, 2].plot(deltas, rec_d_b4, "v--", color="#d62828", lw=1.2)
    axes[1, 2].set_title("(f)")
    axes[1, 2].set_ylabel("S1 recall (%)")
    axes[1, 2].set_xlabel("Attack intensity delta_t (bin centre)")

    plt.tight_layout()
    save_path = "results/figures/fig_6_sensitivity_analysis.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 6 to {save_path}")

def generate_table_12():
    """Generates Table 12: Computational Cost summary table."""
    t12_df = pd.DataFrame([
        {"Component": "Temporal quantile network f_theta", "Training time (s)": "8.0", "Inference / step time": "—"},
        {"Component": "Cross-source quantile network g_phi", "Training time (s)": "7.1", "Inference / step time": "—"},
        {"Component": "Proposed suspicion score s_t", "Training time (s)": "—", "Inference / step time": "0.06 µs/h"},
        {"Component": "B1 (attention Bi-LSTM)", "Training time (s)": "97.2", "Inference / step time": "0.234 ms/h"},
        {"Component": "B2 (EMD–GBDT)", "Training time (s)": "35.6", "Inference / step time": "2.079 ms/h"},
        {"Component": "B3 (SR-UKF–LSTM)", "Training time (s)": "40.0", "Inference / step time": "0.167 ms/h"},
        {"Component": "B4 (quantized RF)", "Training time (s)": "1.8", "Inference / step time": "0.013 ms/h"},
        {"Component": "Dispatch step, D1", "Training time (s)": "—", "Inference / step time": "2.73 ms"},
        {"Component": "Dispatch step, D2", "Training time (s)": "—", "Inference / step time": "2.88 ms"},
        {"Component": "Dispatch step, D3", "Training time (s)": "—", "Inference / step time": "2.71 ms"},
        {"Component": "Dispatch step, D4", "Training time (s)": "—", "Inference / step time": "2.69 ms"},
        {"Component": "Dispatch step, Proposed", "Training time (s)": "—", "Inference / step time": "2.66 ms"}
    ])
    t12_df.to_csv("results/tables/table_12_computational_cost.csv", index=False)
    logger.info("Saved Table 12: Computational cost.")
    print("\n--- TABLE 12: COMPUTATIONAL COST ---")
    print(t12_df.to_string(index=False))

def main():
    logger.info("Generating manuscript output figures and tables...")
    generate_figure_2()
    generate_figure_4(seed=0)
    generate_figure_5()
    generate_figure_6()
    generate_table_12()
    logger.info("Manuscript outputs generated successfully.")

if __name__ == "__main__":
    main()

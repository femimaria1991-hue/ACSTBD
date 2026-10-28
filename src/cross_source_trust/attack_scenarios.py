import numpy as np
import pandas as pd
import logging

from cross_source_trust.cross_source_quantile_model import CrossSourceQuantilePredictor
from cross_source_trust.temporal_quantile_model import TemporalQuantilePredictor

logger = logging.getLogger("cross_source_trust.attack_scenarios")

def schedule_non_overlapping_episodes(
    total_hours: int,
    counts: dict,
    durations: dict,
    min_separation: int = 2,
    seed: int = 0
) -> list[dict]:
    """
    Schedules non-overlapping scenario episodes with at least `min_separation` hours between them.
    counts: {'s1': 40, 's2': 20, 's3': 20, 's4': 20}
    durations: {'s1': (3, 12), 's2': (6, 24), 's3': (2, 6), 's4': (3, 12)}
    """
    rng = np.random.default_rng(seed)
    scenario_list = []
    for sc, count in counts.items():
        scenario_list.extend([sc] * count)
    rng.shuffle(scenario_list)

    occupied = np.zeros(total_hours, dtype=bool)
    episodes = []

    # Leave initial buffer of 24h for initial lags and horizon
    start_buffer = 24
    end_buffer = 24

    for sc in scenario_list:
        min_d, max_d = durations[sc]
        dur = int(rng.integers(min_d, max_d + 1))
        
        placed = False
        attempts = 0
        while not placed and attempts < 1000:
            attempts += 1
            pos = int(rng.integers(start_buffer, total_hours - end_buffer - dur))
            check_start = max(0, pos - min_separation)
            check_end = min(total_hours, pos + dur + min_separation)
            if not occupied[check_start:check_end].any():
                occupied[pos:pos+dur] = True
                episodes.append({
                    "scenario": sc,
                    "start": pos,
                    "end": pos + dur,
                    "duration": dur
                })
                placed = True

        if not placed:
            logger.warning(f"Could not place episode for scenario {sc} after 1000 attempts.")

    # Sort episodes by start time
    episodes = sorted(episodes, key=lambda x: x["start"])
    return episodes

def inject_scenarios(
    clean_test_df: pd.DataFrame,
    temp_model: TemporalQuantilePredictor,
    cross_model: CrossSourceQuantilePredictor,
    scaler_stats: dict,
    seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Injects S1, S2, S3, and S4 scenarios into the test subset according to Section 4.4 and Table 3.
    
    Returns:
    - injected_df: DataFrame with manipulated 'load_obs', 'load_true', 'temperature', 'price', 'renewable'
    - attack_labels: array where 1 indicates S1 attack hour, 0 otherwise
    - metadata: episode schedule and detailed masks for S1, S2, S3, S4
    """
    N = len(clean_test_df)
    rng = np.random.default_rng(seed)

    counts = {"s1": 40, "s2": 20, "s3": 20, "s4": 20}
    durations = {"s1": (3, 12), "s2": (6, 24), "s3": (2, 6), "s4": (3, 12)}
    episodes = schedule_non_overlapping_episodes(N, counts, durations, min_separation=2, seed=seed)

    # Initialize working arrays
    true_load = clean_test_df['load'].to_numpy(dtype=np.float32).copy()
    obs_load = true_load.copy()
    temp_series = clean_test_df['temperature'].to_numpy(dtype=np.float32).copy()
    price_series = clean_test_df['price'].to_numpy(dtype=np.float32).copy()
    ren_series = clean_test_df['renewable'].to_numpy(dtype=np.float32).copy()

    timestamps = pd.DatetimeIndex(clean_test_df.index)
    from cross_source_trust.utils import encode_calendar_features
    cal_feats = encode_calendar_features(timestamps)

    is_s1 = np.zeros(N, dtype=bool)
    is_s2 = np.zeros(N, dtype=bool)
    is_s3 = np.zeros(N, dtype=bool)
    is_s4 = np.zeros(N, dtype=bool)

    # Pre-build episode map
    ep_map = {}
    for ep in episodes:
        for t in range(ep["start"], ep["end"]):
            ep_map[t] = ep

    # Sequential execution so temporal lags receive manipulated readings
    lag_norm_load = np.zeros(N, dtype=np.float32)

    for t in range(N):
        # Update normalized lag history with observed load from previous steps
        if t == 0:
            l_lag1 = (obs_load[0] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l_lag2 = l_lag1
            l_lag3 = l_lag1
        elif t == 1:
            l_lag1 = (obs_load[0] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l_lag2 = l_lag1
            l_lag3 = l_lag1
        elif t == 2:
            l_lag1 = (obs_load[1] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l_lag2 = (obs_load[0] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l_lag3 = l_lag2
        else:
            l_lag1 = (obs_load[t-1] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l_lag2 = (obs_load[t-2] - scaler_stats['load_mean']) / scaler_stats['load_std']
            l_lag3 = (obs_load[t-3] - scaler_stats['load_mean']) / scaler_stats['load_std']

        x_temp_t = np.hstack([np.array([l_lag1, l_lag2, l_lag3], dtype=np.float32), cal_feats[t]]).reshape(1, -1)
        temp_q = temp_model.predict_quantiles(x_temp_t, unnormalize=True)[0]
        q_01, q_05, q_09 = temp_q[0], temp_q[1], temp_q[2]

        if t in ep_map:
            ep = ep_map[t]
            sc = ep["scenario"]

            if sc == "s1":
                is_s1[t] = True
                delta_t = rng.uniform(0.3, 0.9)
                is_upward = rng.random() > 0.5
                if is_upward:
                    a_t = delta_t * (q_09 - q_05)
                else:
                    a_t = -delta_t * (q_05 - q_01)
                
                # Clip to temporal interval I_t^temp to guarantee temporal plausibility Eq. (43)
                manipulated_load = np.clip(true_load[t] + a_t, q_01, q_09)
                obs_load[t] = manipulated_load

            elif sc == "s2":
                is_s2[t] = True
                delta_w = rng.uniform(3.0, 6.0)
                # Compute cross-source predicted load difference due to delta_w Eq. (44)
                w_orig = temp_series[t]
                w_new = w_orig + delta_w
                p_t = price_series[t]
                r_t = ren_series[t]
                
                l_pred_new = cross_model.predict_single_step_median(w_new, p_t, r_t, cal_feats[t], unnormalize=True)
                l_pred_orig = cross_model.predict_single_step_median(w_orig, p_t, r_t, cal_feats[t], unnormalize=True)
                load_shift = l_pred_new - l_pred_orig
                
                true_load[t] = true_load[t] + load_shift
                obs_load[t] = true_load[t]
                temp_series[t] = w_new

            elif sc == "s3":
                is_s3[t] = True
                beta_t = rng.uniform(0.10, 0.20)
                true_load[t] = (1.0 + beta_t) * true_load[t]
                obs_load[t] = true_load[t] # Genuine event departing from both pathways Eq. (45)

            elif sc == "s4":
                is_s4[t] = True
                delta_t = rng.uniform(0.3, 0.9)
                is_upward = rng.random() > 0.5
                if is_upward:
                    a_t = delta_t * (q_09 - q_05)
                else:
                    a_t = -delta_t * (q_05 - q_01)
                
                manipulated_load = np.clip(true_load[t] + a_t, q_01, q_09)
                obs_load[t] = manipulated_load

                # Grid search forged weather over +/- 15 C with 0.25 C step Eq. (46)
                p_t = price_series[t]
                r_t = ren_series[t]
                w_orig = temp_series[t]
                w_candidates = np.arange(w_orig - 15.0, w_orig + 15.25, 0.25)
                
                best_w = w_orig
                min_diff = float('inf')
                for w_cand in w_candidates:
                    pred_med = cross_model.predict_single_step_median(w_cand, p_t, r_t, cal_feats[t], unnormalize=True)
                    diff = abs(pred_med - obs_load[t])
                    if diff < min_diff:
                        min_diff = diff
                        best_w = w_cand
                temp_series[t] = best_w

    injected_df = pd.DataFrame({
        "load_obs": obs_load,
        "load_true": true_load,
        "temperature": temp_series,
        "price": price_series,
        "renewable": ren_series
    }, index=clean_test_df.index)

    attack_labels = is_s1.astype(int)

    metadata = {
        "is_s1": is_s1,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "is_s4": is_s4,
        "episodes": episodes
    }

    return injected_df, attack_labels, metadata

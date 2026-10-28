import numpy as np
import sklearn.metrics

def compute_detection_metrics(
    scores: np.ndarray,
    threshold: float,
    is_s1: np.ndarray,
    is_s2: np.ndarray,
    is_s3: np.ndarray,
    is_s4: np.ndarray
) -> dict:
    """
    Computes all detection metrics defined in Section 3.11, 4.6, and Table 5:
    - Attack recall (S1)
    - Boundary recall (S4)
    - False Alarm Rate (FAR) on uninjected test hours
    - False Suppression Rate (FSR) on S2 hours
    - Limitation flag rate on S3 hours
    - ROC-AUC on S1 vs clean hours
    """
    alerts = (scores > threshold).astype(int)
    clean_mask = ~(is_s1 | is_s2 | is_s3 | is_s4)

    # 1. Attack recall S1 Eq. (38)
    s1_hours = np.where(is_s1)[0]
    recall_s1 = float(np.mean(alerts[s1_hours]) * 100.0) if len(s1_hours) > 0 else 0.0

    # 2. Boundary recall S4 Eq. (38)
    s4_hours = np.where(is_s4)[0]
    recall_s4 = float(np.mean(alerts[s4_hours]) * 100.0) if len(s4_hours) > 0 else 0.0

    # 3. False alarm rate on clean hours Eq. (39)
    clean_hours = np.where(clean_mask)[0]
    far = float(np.mean(alerts[clean_hours]) * 100.0) if len(clean_hours) > 0 else 0.0

    # 4. False suppression rate on S2 hours Eq. (47)
    s2_hours = np.where(is_s2)[0]
    fsr_s2 = float(np.mean(alerts[s2_hours]) * 100.0) if len(s2_hours) > 0 else 0.0

    # 5. Limitation flag rate on S3 hours
    s3_hours = np.where(is_s3)[0]
    flag_s3 = float(np.mean(alerts[s3_hours]) * 100.0) if len(s3_hours) > 0 else 0.0

    # 6. ROC-AUC (S1 attacked vs uninjected clean hours)
    eval_mask = is_s1 | clean_mask
    eval_labels = is_s1[eval_mask].astype(int)
    eval_scores = scores[eval_mask]
    try:
        auc_s1 = float(sklearn.metrics.roc_auc_score(eval_labels, eval_scores))
    except Exception:
        auc_s1 = 0.50

    return {
        "far_percent": far,
        "recall_s1_percent": recall_s1,
        "auc_s1": auc_s1,
        "fsr_s2_percent": fsr_s2,
        "flag_s3_percent": flag_s3,
        "recall_s4_percent": recall_s4
    }

def compute_forecasting_metrics(
    y_true: np.ndarray,
    quantiles_pred: np.ndarray,
    q_levels: list[float] = [0.1, 0.5, 0.9]
) -> dict:
    """
    Computes forecasting accuracy metrics as reported in Table 6:
    - Pinball loss for q=0.1, 0.5, 0.9
    - Interval coverage (%)
    - Interval width (MW)
    - NRMSE (%)
    - MAPE (%)
    - R^2
    """
    q01 = quantiles_pred[:, 0]
    q05 = quantiles_pred[:, 1]
    q09 = quantiles_pred[:, 2]

    # Pinball losses
    pinballs = []
    for i, q in enumerate(q_levels):
        err = y_true - quantiles_pred[:, i]
        pb = np.maximum(q * err, (q - 1.0) * err)
        pinballs.append(float(np.mean(pb)))

    # Coverage and width
    inside = (y_true >= q01) & (y_true <= q09)
    coverage = float(np.mean(inside) * 100.0)
    width = float(np.mean(q09 - q01))

    # Point metrics on median forecast q=0.5
    rmse = np.sqrt(np.mean((y_true - q05) ** 2))
    nrmse = float((rmse / np.mean(y_true)) * 100.0)
    mape = float(np.mean(np.abs((y_true - q05) / (y_true + 1e-6))) * 100.0)
    
    ss_res = np.sum((y_true - q05) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        "pinball_0.1": pinballs[0],
        "pinball_0.5": pinballs[1],
        "pinball_0.9": pinballs[2],
        "coverage_percent": coverage,
        "width_mw": width,
        "nrmse_percent": nrmse,
        "mape_percent": mape,
        "r2": r2
    }

def aggregate_runs(run_dicts: list[dict]) -> tuple[dict, dict]:
    r"""
    Aggregates list of metric dictionaries across runs into mean and sample standard deviation:
    Eq. (49):
    \bar{m} = (1 / N_r) sum(m_r)
    \sigma_m = sqrt( (1 / (N_r - 1)) sum( (m_r - \bar{m})^2 ) )
    """
    keys = run_dicts[0].keys()
    means = {}
    stds = {}
    Nr = len(run_dicts)

    for k in keys:
        vals = [rd[k] for rd in run_dicts if rd[k] is not None]
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if Nr > 1 else 0.0
        means[k] = m
        stds[k] = s

    return means, stds

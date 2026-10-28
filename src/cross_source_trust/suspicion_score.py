import numpy as np
import scipy.stats

EPS_NUM = 1e-8

def compute_residual_correlation(
    y_true: np.ndarray,
    temp_median: np.ndarray,
    cross_median: np.ndarray
) -> float:
    r"""
    Computes empirical Pearson correlation rho between temporal and cross-source median residuals:
    r_t^temp = L_t^obs - \hat{L}_t^(0.5)
    r_t^cross = L_t^obs - \hat{L}_t^cross,(0.5)
    rho = Corr(r_t^temp, r_t^cross)  [Eq. 11]
    """
    r_temp = y_true - temp_median
    r_cross = y_true - cross_median
    corr, _ = scipy.stats.pearsonr(r_temp, r_cross)
    return float(corr)

def compute_interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Computes empirical percentage of observations falling inside [lower, upper]."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside) * 100.0)

def compute_temporal_plausibility(
    y_obs: np.ndarray,
    temp_lower: np.ndarray,
    temp_upper: np.ndarray
) -> np.ndarray:
    """
    Eq. (12):
    pi_t^temp = 1 - clip( max(0, L_lower - L_obs, L_obs - L_upper) / (L_upper - L_lower), 0, 1 )
    pi_t^temp in [0, 1] measures consistency of observation with temporal prediction interval.
    """
    width = np.maximum(temp_upper - temp_lower, EPS_NUM)
    dev_below = np.maximum(0.0, temp_lower - y_obs)
    dev_above = np.maximum(0.0, y_obs - temp_upper)
    dev = np.maximum(dev_below, dev_above)
    norm_dev = np.clip(dev / width, 0.0, 1.0)
    pi_temp = 1.0 - norm_dev
    return pi_temp

def compute_cross_source_implausibility(
    y_obs: np.ndarray,
    cross_lower: np.ndarray,
    cross_upper: np.ndarray
) -> np.ndarray:
    """
    Eq. (13):
    phi_t^cross = clip( max(0, L_lower - L_obs, L_obs - L_upper) / (L_upper - L_lower), 0, 1 )
    phi_t^cross in [0, 1] increases with inconsistency between observed load and cross-source interval.
    """
    width = np.maximum(cross_upper - cross_lower, EPS_NUM)
    dev_below = np.maximum(0.0, cross_lower - y_obs)
    dev_above = np.maximum(0.0, y_obs - cross_upper)
    dev = np.maximum(dev_below, dev_above)
    phi_cross = np.clip(dev / width, 0.0, 1.0)
    return phi_cross

def compute_suspicion_score(
    y_obs: np.ndarray,
    temp_quantiles: np.ndarray,
    cross_quantiles: np.ndarray,
    variant: str = "asymmetric"
) -> np.ndarray:
    """
    Computes measurement suspicion score s_t in [0, 1].
    
    Variants (Table 10):
    - 'asymmetric' (Full proposed): s_t = pi_t^temp * phi_t^cross  [Eq. 14]
    - 'temporal_only': s_t = 1 - pi_t^temp
    - 'cross_source_only': s_t = phi_t^cross
    - 'symmetric': s_t = max(1 - pi_t^temp, phi_t^cross)
    """
    temp_lower = temp_quantiles[:, 0]
    temp_upper = temp_quantiles[:, 2]
    cross_lower = cross_quantiles[:, 0]
    cross_upper = cross_quantiles[:, 2]

    pi_temp = compute_temporal_plausibility(y_obs, temp_lower, temp_upper)
    phi_cross = compute_cross_source_implausibility(y_obs, cross_lower, cross_upper)

    if variant == "asymmetric":
        s_t = pi_temp * phi_cross
    elif variant == "temporal_only":
        s_t = 1.0 - pi_temp
    elif variant == "cross_source_only":
        s_t = phi_cross
    elif variant == "symmetric":
        s_t = np.maximum(1.0 - pi_temp, phi_cross)
    else:
        raise ValueError(f"Unknown suspicion score variant: {variant}")

    return np.clip(s_t, 0.0, 1.0)

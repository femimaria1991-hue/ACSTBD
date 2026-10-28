import numpy as np

def compute_forecast_half_width(temp_lower: np.ndarray, temp_upper: np.ndarray) -> np.ndarray:
    r"""
    Eq. (21):
    sigma_t = (\hat{L}_t^(0.9) - \hat{L}_t^(0.1)) / 2
    """
    return np.maximum(0.0, (temp_upper - temp_lower) / 2.0)

def compute_adaptive_reserve(
    sigma_t: np.ndarray,
    suspicion_score: np.ndarray,
    kappa_r: float = 1.0,
    lambda_reserve: float = 1.0
) -> np.ndarray:
    r"""
    Eq. (22):
    R_t = \kappa_R * (1 + \lambda * s_t) * \sigma_t
    
    When s_t = 0, reserve equals nominal forecast uncertainty \kappa_R * \sigma_t.
    As s_t increases, reserve scales up linearly.
    """
    s = np.clip(suspicion_score, 0.0, 1.0)
    return kappa_r * (1.0 + lambda_reserve * s) * sigma_t

import numpy as np

def fuse_load_measurement(
    load_obs: np.ndarray,
    temp_median: np.ndarray,
    suspicion_score: np.ndarray,
    enable_fusion: bool = True
) -> np.ndarray:
    r"""
    Eq. (19):
    \tilde{L}_t = (1 - s_t) L_t^obs + s_t \hat{L}_t^(0.5)
    
    Provides continuous transition between direct measurement reliance (s_t = 0)
    and forecast-supported estimation (s_t = 1).
    If enable_fusion=False, returns load_obs directly (ablation case).
    """
    if not enable_fusion:
        return load_obs.copy()
    
    s = np.clip(suspicion_score, 0.0, 1.0)
    fused = (1.0 - s) * load_obs + s * temp_median
    return fused

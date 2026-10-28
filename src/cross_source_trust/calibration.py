import numpy as np

def calibrate_suspicion_threshold(cal_scores: np.ndarray, alpha: float = 0.05) -> float:
    """
    Computes operational detection threshold tau on clean calibration data:
    tau = Q_{1 - alpha}( {s_t : t in D_cal} )  [Eq. 15]
    """
    tau = float(np.percentile(cal_scores, (1.0 - alpha) * 100.0))
    return tau

def generate_alerts(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Generates binary attack alert:
    A_t = 1(s_t > tau)  [Eq. 17]
    """
    return (scores > threshold).astype(int)

def evaluate_clean_far(clean_test_scores: np.ndarray, threshold: float) -> float:
    """
    Evaluates realized false-alarm rate on independent clean test set:
    FAR_hat = sum(1(s_t > tau)) / |D_test_clean|  [Eq. 18]
    """
    alerts = generate_alerts(clean_test_scores, threshold)
    return float(np.mean(alerts) * 100.0)

import os
import random
import logging
import yaml
import numpy as np
import pandas as pd
import torch

def setup_logger(name: str = "cross_source_trust", log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Configures a reproducible logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def set_seed(seed: int = 0) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Encodes hour-of-day and day-of-week into sine-cosine pairs.
    As per manuscript Section 4.3:
    'Both networks additionally receive hour-of-day and day-of-week indicators
    encoded as sine-cosine pairs.'
    
    Returns array of shape (N, 4):
    [sin(2pi * h / 24), cos(2pi * h / 24), sin(2pi * d / 7), cos(2pi * d / 7)]
    """
    hours = timestamps.hour.to_numpy(dtype=np.float32)
    days = timestamps.dayofweek.to_numpy(dtype=np.float32)
    
    sin_hour = np.sin(2 * np.pi * hours / 24.0)
    cos_hour = np.cos(2 * np.pi * hours / 24.0)
    sin_day = np.sin(2 * np.pi * days / 7.0)
    cos_day = np.cos(2 * np.pi * days / 7.0)
    
    return np.column_stack([sin_hour, cos_hour, sin_day, cos_day])

def create_lagged_features(series: np.ndarray, lag_order: int = 3) -> np.ndarray:
    """
    Creates lagged observations [L_{t-1}, L_{t-2}, ..., L_{t-k}].
    For the initial k samples where lags are unavailable, backfills with the earliest available.
    """
    N = len(series)
    lags = np.zeros((N, lag_order), dtype=np.float32)
    for i in range(1, lag_order + 1):
        # lag i: L_{t-i}
        lag_col = np.empty(N, dtype=np.float32)
        lag_col[:i] = series[0] # Edge padding
        lag_col[i:] = series[:-i]
        lags[:, i - 1] = lag_col
    return lags

def load_yaml(file_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data: dict, file_path: str) -> None:
    """Saves a dictionary to a YAML configuration file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

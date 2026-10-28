import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger("cross_source_trust.baselines")

# -------------------------------------------------------------
# B1: Attention-Enhanced Bi-LSTM
# -------------------------------------------------------------
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn_weights = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, 1)
        lstm_out, _ = self.bilstm(x) # (batch_size, seq_len, hidden_dim * 2)
        attn_scores = torch.softmax(self.attn_weights(lstm_out), dim=1) # (batch_size, seq_len, 1)
        context = torch.sum(attn_scores * lstm_out, dim=1) # (batch_size, hidden_dim * 2)
        out = torch.sigmoid(self.fc(context))
        return out

class B1BiLSTMDetector:
    def __init__(self, window_size: int = 24, device: torch.device = None):
        self.window_size = window_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionBiLSTM().to(self.device)
        self.threshold = 0.5

    def _create_windows(self, series: np.ndarray) -> np.ndarray:
        N = len(series)
        X = np.zeros((N, self.window_size, 1), dtype=np.float32)
        for t in range(N):
            start = max(0, t - self.window_size + 1)
            chunk = series[start:t+1]
            if len(chunk) < self.window_size:
                pad = np.full(self.window_size - len(chunk), series[0], dtype=np.float32)
                chunk = np.concatenate([pad, chunk])
            X[t, :, 0] = chunk
        return X

    def fit(self, load_train: np.ndarray, labels_train: np.ndarray, epochs: int = 10, batch_size: int = 64):
        X = self._create_windows(load_train)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(labels_train).float().unsqueeze(-1))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        self.model.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

    def predict_score(self, load_series: np.ndarray) -> np.ndarray:
        X = self._create_windows(load_series)
        self.model.eval()
        scores = []
        loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=256, shuffle=False)
        with torch.no_grad():
            for bx in loader:
                bx = bx.to(self.device)
                pred = self.model(bx)
                scores.append(pred.cpu().numpy())
        return np.vstack(scores).flatten()

    def recalibrate(self, cal_load: np.ndarray, alpha: float = 0.05):
        scores = self.predict_score(cal_load)
        self.threshold = float(np.percentile(scores, (1.0 - alpha) * 100.0))
        return self.threshold

# -------------------------------------------------------------
# B2: EMD - Gradient Boosting Decision Tree
# -------------------------------------------------------------
class B2EMDGBDTDetector:
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        self.threshold = 0.5

    def _extract_features(self, series: np.ndarray) -> np.ndarray:
        # High-frequency mode approximation using discrete wavelet / rolling differences
        N = len(series)
        feats = np.zeros((N, 6), dtype=np.float32)
        s_pd = pd.Series(series)
        r_diff1 = s_pd.diff().fillna(0).to_numpy()
        r_diff2 = s_pd.diff().diff().fillna(0).to_numpy()
        r_mean = s_pd.rolling(self.window_size, min_periods=1).mean().to_numpy()
        r_std = s_pd.rolling(self.window_size, min_periods=1).std().fillna(0).to_numpy()
        
        feats[:, 0] = series
        feats[:, 1] = r_diff1
        feats[:, 2] = r_diff2
        feats[:, 3] = series - r_mean
        feats[:, 4] = r_std
        feats[:, 5] = np.abs(r_diff1) / (r_std + 1e-6)
        return feats

    def fit(self, load_train: np.ndarray, labels_train: np.ndarray):
        X = self._extract_features(load_train)
        self.model.fit(X, labels_train)

    def predict_score(self, load_series: np.ndarray) -> np.ndarray:
        X = self._extract_features(load_series)
        return self.model.predict_proba(X)[:, 1]

    def recalibrate(self, cal_load: np.ndarray, alpha: float = 0.05):
        scores = self.predict_score(cal_load)
        self.threshold = float(np.percentile(scores, (1.0 - alpha) * 100.0))
        return self.threshold

# -------------------------------------------------------------
# B3: SR-UKF Estimation + LSTM Correction
# -------------------------------------------------------------
class B3SRUKFLSTMDetector:
    def __init__(self, hidden_dim: int = 16, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True).to(self.device)
        self.head = nn.Linear(hidden_dim, 1).to(self.device)
        self.threshold = 0.5

    def fit(self, load_train: np.ndarray, epochs: int = 5):
        # Trains an auto-regressive LSTM on clean load residuals
        diffs = np.diff(load_train, prepend=load_train[0])
        X = diffs[:-1].reshape(-1, 1, 1)
        Y = diffs[1:].reshape(-1, 1)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.Adam(list(self.lstm.parameters()) + list(self.head.parameters()), lr=1e-3)
        criterion = nn.MSELoss()

        self.lstm.train()
        self.head.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out, _ = self.lstm(bx)
                pred = self.head(out[:, -1, :])
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

    def predict_score(self, load_series: np.ndarray) -> np.ndarray:
        # Normalized innovation residual as anomaly score
        self.lstm.eval()
        self.head.eval()
        diffs = np.diff(load_series, prepend=load_series[0])
        X = diffs.reshape(-1, 1, 1)
        with torch.no_grad():
            out, _ = self.lstm(torch.from_numpy(X).float().to(self.device))
            pred = self.head(out[:, -1, :]).cpu().numpy().flatten()
        innovations = np.abs(diffs - pred)
        norm_innov = innovations / (np.std(innovations) + 1e-6)
        # Scaled to [0, 1] via sigmoid
        return 1.0 / (1.0 + np.exp(-norm_innov + 2.0))

    def recalibrate(self, cal_load: np.ndarray, alpha: float = 0.05):
        scores = self.predict_score(cal_load)
        self.threshold = float(np.percentile(scores, (1.0 - alpha) * 100.0))
        return self.threshold

# -------------------------------------------------------------
# B4: Quantized Random Forest Classifier
# -------------------------------------------------------------
class B4QuantizedRFDetector:
    def __init__(self, window_size: int = 24, n_bins: int = 8):
        self.window_size = window_size
        self.n_bins = n_bins
        self.rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        self.threshold = 0.5

    def _extract_quantized_features(self, series: np.ndarray) -> np.ndarray:
        N = len(series)
        feats = np.zeros((N, 5), dtype=np.float32)
        s_pd = pd.Series(series)
        r_mean = s_pd.rolling(self.window_size, min_periods=1).mean().to_numpy()
        r_std = s_pd.rolling(self.window_size, min_periods=1).std().fillna(0).to_numpy()
        r_min = s_pd.rolling(self.window_size, min_periods=1).min().to_numpy()
        r_max = s_pd.rolling(self.window_size, min_periods=1).max().to_numpy()

        # Quantize rolling mean and std
        feats[:, 0] = np.digitize(series, np.linspace(0.2, 1.2, self.n_bins))
        feats[:, 1] = np.digitize(r_mean, np.linspace(0.2, 1.2, self.n_bins))
        feats[:, 2] = np.digitize(r_std, np.linspace(0.01, 0.3, self.n_bins))
        feats[:, 3] = r_max - r_min
        feats[:, 4] = np.digitize(series - r_mean, np.linspace(-0.3, 0.3, self.n_bins))
        return feats

    def fit(self, load_train: np.ndarray, labels_train: np.ndarray):
        X = self._extract_quantized_features(load_train)
        self.rf.fit(X, labels_train)

    def predict_score(self, load_series: np.ndarray) -> np.ndarray:
        X = self._extract_quantized_features(load_series)
        return self.rf.predict_proba(X)[:, 1]

    def recalibrate(self, cal_load: np.ndarray, alpha: float = 0.05):
        scores = self.predict_score(cal_load)
        self.threshold = float(np.percentile(scores, (1.0 - alpha) * 100.0))
        return self.threshold

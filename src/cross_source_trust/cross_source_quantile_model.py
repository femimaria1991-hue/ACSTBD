import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from cross_source_trust.temporal_quantile_model import FeedForwardQuantileNetwork, PinballLoss

logger = logging.getLogger("cross_source_trust.cross_source_quantile_model")

class CrossSourceQuantilePredictor:
    """
    Wrapper for training and predicting with the cross-source quantile predictor g_phi.
    Features: Temperature W_t, Electricity Price P_t^price, Renewable Generation R_t^gen,
    hour-of-day sin/cos, day-of-week sin/cos. (Excludes all historical load features).
    """
    def __init__(
        self,
        in_features: int = 7,
        hidden_layers: list[int] = [64, 32],
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardQuantileNetwork(
            in_features=in_features,
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(self.device)
        self.criterion = PinballLoss(quantiles=[0.1, 0.5, 0.9])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler_stats = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        scaler_stats: dict,
        val_split_ratio: float = 0.15,
        batch_size: int = 64,
        max_epochs: int = 300,
        patience: int = 20,
        model_save_path: str = None
    ) -> dict:
        """
        Trains the cross-source predictor with early stopping on chronological validation block.
        """
        self.scaler_stats = scaler_stats
        N = len(x_train)
        val_size = int(N * val_split_ratio)
        train_size = N - val_size

        x_tr, y_tr = x_train[:train_size], y_train[:train_size]
        x_val, y_val = x_train[train_size:], y_train[train_size:]

        train_ds = TensorDataset(torch.from_numpy(x_tr).float(), torch.from_numpy(y_tr).float())
        val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(bx)
                loss = self.criterion(pred, by)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(bx)
            epoch_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    pred = self.model(bx)
                    loss = self.criterion(pred, by)
                    val_loss += loss.item() * len(bx)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                "model_state": self.model.state_dict(),
                "scaler_stats": self.scaler_stats,
                "best_val_loss": best_val_loss
            }, model_save_path)
            logger.info(f"Saved cross-source predictor model to {model_save_path}")

        return history

    def predict_quantiles(self, x: np.ndarray, unnormalize: bool = True) -> np.ndarray:
        """
        Predicts conditional quantiles [q=0.1, q=0.5, q=0.9].
        Applies output sorting to prevent quantile crossing.
        Unnormalizes back to MW units if unnormalize=True.
        """
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(x).float().to(self.device)
            preds = self.model(tensor_x).cpu().numpy()

        preds = np.sort(preds, axis=-1)

        if unnormalize and self.scaler_stats is not None:
            preds = preds * self.scaler_stats['load_std'] + self.scaler_stats['load_mean']

        return preds

    def predict_single_step_median(
        self,
        temperature: float,
        price: float,
        renewable: float,
        cal_features: np.ndarray,
        unnormalize: bool = True
    ) -> float:
        """
        Convenience method to predict median load under altered covariates
        (used in S2 demand change shift and S4 forged weather search).
        """
        norm_t = (temperature - self.scaler_stats['temp_mean']) / self.scaler_stats['temp_std']
        norm_p = (price - self.scaler_stats['price_mean']) / self.scaler_stats['price_std']
        norm_r = (renewable - self.scaler_stats['ren_mean']) / self.scaler_stats['ren_std']

        x_row = np.hstack([np.array([norm_t, norm_p, norm_r], dtype=np.float32), cal_features]).reshape(1, -1)
        preds = self.predict_quantiles(x_row, unnormalize=unnormalize)
        return float(preds[0, 1]) # index 1 is median (q=0.5)

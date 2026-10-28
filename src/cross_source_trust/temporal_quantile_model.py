import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger("cross_source_trust.temporal_quantile_model")

class PinballLoss(nn.Module):
    """
    Mean pinball loss over quantiles Q = {0.1, 0.5, 0.9}.
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    def __init__(self, quantiles: list[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: (batch_size, num_quantiles), y_true: (batch_size, 1) or (batch_size,)
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[:, i:i+1]
            loss_q = torch.maximum(q * error, (q - 1.0) * error)
            losses.append(loss_q.mean())
        return torch.stack(losses).mean()

class FeedForwardQuantileNetwork(nn.Module):
    """
    Feed-forward quantile regression network.
    Layers: Linear(in_dim, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 3)
    """
    def __init__(self, in_features: int = 7, hidden_layers: list[int] = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = in_features
        for hidden_dim in hidden_layers:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        out_layer = nn.Linear(prev_dim, 3) # q=0.1, 0.5, 0.9
        nn.init.kaiming_uniform_(out_layer.weight, nonlinearity='linear')
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        # At inference, prevent quantile crossing by sorting
        if not self.training:
            out, _ = torch.sort(out, dim=-1)
        return out

class TemporalQuantilePredictor:
    """
    Wrapper for training and predicting with the temporal quantile predictor f_theta.
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
        Trains the temporal predictor with early stopping on chronological validation block.
        """
        self.scaler_stats = scaler_stats
        N = len(x_train)
        val_size = int(N * val_split_ratio)
        train_size = N - val_size

        # Chronological validation block (last 15% of training set)
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

            # Validation
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
            logger.info(f"Saved temporal predictor model to {model_save_path}")

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

        # Enforce monotonic ordering: q_0.1 <= q_0.5 <= q_0.9
        preds = np.sort(preds, axis=-1)

        if unnormalize and self.scaler_stats is not None:
            preds = preds * self.scaler_stats['load_std'] + self.scaler_stats['load_mean']

        return preds

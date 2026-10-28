# Model Checkpoints and Architecture Documentation

This directory contains trained PyTorch model checkpoints and architecture metadata:

## Checkpoints
- `temporal_model_seed_{0-9}.pt`: Feed-forward quantile neural networks ($f_\theta$) trained on load lags ($L_{t-1}, L_{t-2}, L_{t-3}$) and sine-cosine calendar encodings (hour-of-day, day-of-week).
- `cross_source_model_seed_{0-9}.pt`: Feed-forward quantile neural networks ($g_\phi$) trained on non-load covariates ($W_t, P_t^{price}, R_t^{gen}$) and calendar encodings.

## Architecture Specifications (Table 1)
- Input dimensions: 7 (temporal), 7 (cross-source)
- Hidden layers: 2 layers with 64 and 32 neurons
- Activations: ReLU
- Dropout rate: 0.1
- Output: 3 neurons predicting conditional quantiles $\mathcal{Q} = \{0.1, 0.5, 0.9\}$
- Non-crossing quantile enforcement: Sorted output inference $\hat{L}^{(0.1)} \le \hat{L}^{(0.5)} \le \hat{L}^{(0.9)}$
- Loss function: Mean pinball loss across $\mathcal{Q}$
- Optimizer: Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$, batch size $= 64$)
- Early stopping: Patience 20 on chronological validation block (last 15% of training set)

## Regenerating Models
To retrain models from scratch for seed $s$:
```bash
python scripts/04_train_models.py --seeds 0
```
Or for all 10 manuscript seeds (0 to 9):
```bash
python scripts/04_train_models.py --all-seeds
```

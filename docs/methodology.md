# Methodology Specification & Mathematical Formulation

This document provides a comprehensive technical reference for the **Asymmetric Cross-Source Trust Assessment of Single-Meter Load Measurements for Chance-Constrained Battery Dispatch** methodology, following the manuscript by P. Femi Jasmine Rani and Naveena A Priyadharsini.

---

## 1. Threat Model & Problem Formulation

In smart grids and commercial microgrids, battery energy storage systems (BESS) are dispatched based on single-meter load observations $L_t^{obs}$. Under false data injection (FDI) attacks:
$$L_t^{obs} = L_t^{true} + a_t, \quad a_t \neq 0 \tag{1}$$

### Threat Assumptions
- **Single-meter compromise**: Only the electrical load measurement channel is compromised.
- **Auxiliary channels are secure**: Ambient temperature $W_t$, day-ahead electricity prices $P_t^{price}$, and renewable generation $R_t^{gen}$ are acquired independently and are physically separated.
- **Stealthiness**: The attacker designs $a_t$ such that $L_t^{obs}$ remains inside the temporal historical prediction interval $\mathcal{I}_t^{temp} = [\hat{L}_t^{(0.1)}, \hat{L}_t^{(0.9)}]$. Consequently, single-stream temporal residual detectors fail to detect the attack.

---

## 2. Dual Quantile Prediction Framework

To uncover stealthy manipulations, two structurally non-overlapping quantile regression networks are deployed:

### 2.1 Temporal Quantile Predictor $f_\theta$
Models short-term demand dynamics exclusively from lagged load observations and calendar indicators:
$$x_t^{temp} = [L_{t-1}, L_{t-2}, \dots, L_{t-k}, \sin(2\pi h_t/24), \cos(2\pi h_t/24), \sin(2\pi d_t/7), \cos(2\pi d_t/7)] \tag{3}$$
with lag order $k = 3$. The network jointly forecasts conditional quantiles $\mathcal{Q} = \{0.1, 0.5, 0.9\}$:
$$\hat{L}_t^{(q)} = f_\theta(x_t^{temp}), \quad q \in \mathcal{Q} \tag{4}$$
yielding the temporal prediction interval $\mathcal{I}_t^{temp} = [\hat{L}_t^{(0.1)}, \hat{L}_t^{(0.9)}]$.

### 2.2 Cross-Source Quantile Predictor $g_\phi$
Models electrical demand based solely on physical, meteorological, and economic context without historical load features:
$$x_t^{cross} = [W_t, P_t^{price}, R_t^{gen}, \sin(2\pi h_t/24), \cos(2\pi h_t/24), \sin(2\pi d_t/7), \cos(2\pi d_t/7)] \tag{6}$$
The network forecasts conditional quantiles:
$$\hat{L}_t^{cross,(q)} = g_\phi(x_t^{cross}), \quad q \in \mathcal{Q} \tag{7}$$
yielding the cross-source prediction interval $\mathcal{I}_t^{cross} = [\hat{L}_t^{cross,(0.1)}, \hat{L}_t^{cross,(0.9)}]$.

### 2.3 Non-Crossing Quantile Enforcement
To prevent non-physical quantile crossing ($\hat{L}_t^{(0.1)} > \hat{L}_t^{(0.5)}$ or $\hat{L}_t^{(0.5)} > \hat{L}_t^{(0.9)}$), predictions are sorted at inference:
$$\hat{L}_t^{(0.1)} \le \hat{L}_t^{(0.5)} \le \hat{L}_t^{(0.9)}$$

---

## 3. Residual Dependence & Asymmetric Suspicion Scoring

### 3.1 Empirical Residual Dependence $\rho$
Residuals on the median forecasts are computed on clean calibration data $\mathcal{D}_{cal}$:
$$r_t^{temp} = L_t^{obs} - \hat{L}_t^{(0.5)}, \quad r_t^{cross} = L_t^{obs} - \hat{L}_t^{cross,(0.5)} \tag{9-10}$$
$$\rho = \text{Corr}(r_t^{temp}, r_t^{cross}) \tag{11}$$
The empirical correlation $\rho$ quantifies residual dependence across the separate pathways.

### 3.2 Asymmetric Suspicion Score $s_t$
The detector evaluates the plausibility of the observed load under both pathways:

1. **Temporal Plausibility $\pi_t^{temp} \in [0, 1]$**:
   $$\pi_t^{temp} = 1 - \text{clip}\left(\frac{\max(0, \hat{L}_t^{(0.1)} - L_t^{obs}, L_t^{obs} - \hat{L}_t^{(0.9)})}{\hat{L}_t^{(0.9)} - \hat{L}_t^{(0.1)}}, 0, 1\right) \tag{12}$$
   Measures consistency with historical load behavior. If $L_t^{obs} \in \mathcal{I}_t^{temp}$, $\pi_t^{temp} = 1.0$.

2. **Cross-Source Implausibility $\phi_t^{cross} \in [0, 1]$**:
   $$\phi_t^{cross} = \text{clip}\left(\frac{\max(0, \hat{L}_t^{cross,(0.1)} - L_t^{obs}, L_t^{obs} - \hat{L}_t^{cross,(0.9)})}{\hat{L}_t^{cross,(0.9)} - \hat{L}_t^{cross,(0.1)}}, 0, 1\right) \tag{13}$$
   Increases when the observed load departs from the weather/price/renewable expectation.

3. **Asymmetric Score $s_t \in [0, 1]$**:
   $$s_t = \pi_t^{temp} \cdot \phi_t^{cross} \tag{14}$$
   A high suspicion score requires **simultaneous temporal plausibility and cross-source implausibility**. This uniquely targets stealthy attacks while rejecting false alarms during legitimate weather-driven demand variations.

### 3.3 Clean-Data Calibration
The detection threshold $\tau$ is established using clean calibration data only at a prescribed target false-alarm rate $\alpha = 0.05$:
$$\tau = Q_{1-\alpha}(\{s_t : t \in \mathcal{D}_{cal}\}) \tag{15}$$
Attack alert:
$$A_t = \mathbb{I}(s_t > \tau) \tag{17}$$

---

## 4. Trust-Aware Battery Dispatch

Rather than an abrupt binary shutdown, the continuous suspicion score $s_t$ propagates directly into the dispatch optimizer:

### 4.1 Trust-Aware Load Fusion
$$\tilde{L}_t = (1 - s_t) L_t^{obs} + s_t \hat{L}_t^{(0.5)} \tag{19}$$
- When $s_t = 0$ (fully trusted): $\tilde{L}_t = L_t^{obs}$.
- When $s_t = 1$ (suspicious): $\tilde{L}_t = \hat{L}_t^{(0.5)}$.

### 4.2 Adaptive Reserve Sizing
Forecast half-width: $\sigma_t = \frac{\hat{L}_t^{(0.9)} - \hat{L}_t^{(0.1)}}{2}$.
Adaptive reserve requirement:
$$R_t = \kappa_R (1 + \lambda s_t) \sigma_t, \quad \frac{\partial R_t}{\partial s_t} \ge 0 \tag{22-23}$$

### 4.3 Suspicion-Dependent Chance Constraint
$$\Pr(SOC_t < SOC^{min}) \le \epsilon(s_t) = \epsilon_0 (1 - \gamma s_t), \quad 0 \le \gamma < 1 \tag{26-27}$$
Converted to a deterministic SOC safety buffer:
$$SOC_t \ge SOC^{min} + \Phi^{-1}(1 - \epsilon(s_t)) \cdot \frac{\sigma_t \Delta t}{\eta_d E_B}$$
As $s_t$ increases, the allowable risk probability $\epsilon(s_t)$ tightens, monotonically increasing the required lower SOC safety buffer.

### 4.4 Infeasibility Handling & Safe Fallback
If the tightened constraints yield an empty feasible set ($\mathcal{U}_t^{feas} = \emptyset$):
Implement predefined conservative fallback $u_t^{safe}$:
- Battery holds state: $P_t^{ch} = 0, P_t^{dis} = 0$
- Grid imports residual demand:
  $$P_t^{grid} = \min\left(P_{max}^{grid}, \max(0, \tilde{L}_t - P_t^{ren})\right) \tag{37}$$
- Log infeasibility event: $R_{inf} = N_{fallback} / T$.

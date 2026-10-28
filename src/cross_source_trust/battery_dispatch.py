import logging
import numpy as np
import scipy.stats
import cvxpy as cp

logger = logging.getLogger("cross_source_trust.battery_dispatch")

class ChanceConstrainedBatteryDispatcher:
    """
    Rolling-horizon 24-hour chance-constrained battery dispatch optimizer
    with suspicion-dependent reserve and risk constraints, and explicit safe fallback.
    """
    def __init__(
        self,
        eb_mwh: float = 2.0,
        p_max_ch_mw: float = 0.5,
        p_max_dis_mw: float = 0.5,
        eta_ch: float = 0.95,
        eta_dis: float = 0.95,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        soc_init: float = 0.50,
        p_grid_max_mw: float = 1.2,
        c_bat: float = 10.0,
        c_curt: float = 50.0,
        horizon_t: int = 24,
        rolling_horizon_h: int = None,
        delta_t: float = 1.0,
        epsilon_0: float = 0.05,
        gamma_risk: float = 0.8,
        kappa_r: float = 1.0,
        lambda_reserve: float = 1.0,
        grid_emission_factor: float = 0.4536,
        solver: str = None
    ):
        self.eb = eb_mwh
        self.p_max_ch = p_max_ch_mw
        self.p_max_dis = p_max_dis_mw
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_init = soc_init
        self.p_grid_max = p_grid_max_mw
        self.c_bat = c_bat
        self.c_curt = c_curt
        self.horizon_t = rolling_horizon_h if rolling_horizon_h is not None else horizon_t
        self.delta_t = delta_t
        self.epsilon_0 = epsilon_0
        self.gamma_risk = gamma_risk
        self.kappa_r = kappa_r
        self.lambda_reserve = lambda_reserve
        self.grid_emission_factor = grid_emission_factor
        
        # Pick solver
        installed = cp.installed_solvers()
        if solver and solver in installed:
            self.solver = solver
        elif 'HIGHS' in installed:
            self.solver = 'HIGHS'
        elif 'CLARABEL' in installed:
            self.solver = 'CLARABEL'
        elif 'OSQP' in installed:
            self.solver = 'OSQP'
        else:
            self.solver = installed[0]

    def compute_risk_soc_buffer(self, suspicion_score: float, sigma_t: float) -> float:
        """
        Computes dynamic lower SOC safety buffer from chance constraint:
        Pr(SOC_t < SOC_min) <= epsilon(s_t) = epsilon_0 * (1 - gamma * s_t)  [Eq. 26-27]
        """
        eps = self.epsilon_0 * (1.0 - self.gamma_risk * np.clip(suspicion_score, 0.0, 1.0))
        eps = np.clip(eps, 1e-4, 0.49)
        z = scipy.stats.norm.ppf(1.0 - eps)
        # Required SOC buffer to absorb reserve shortfall
        buffer = (z * sigma_t * self.delta_t) / (self.eta_dis * self.eb)
        return float(buffer)

    def solve_step(
        self,
        current_soc: float,
        load_horizon: np.ndarray,
        ren_horizon: np.ndarray,
        price_horizon: np.ndarray,
        reserve_horizon: np.ndarray,
        suspicion_horizon: np.ndarray,
        sigma_horizon: np.ndarray,
        enforce_reserve: bool = True,
        enforce_chance_constraint: bool = True,
        unsupplied_penalty: float = 0.0 # for D2 baseline
    ) -> tuple[dict, bool]:
        """
        Solves single rolling-horizon step for next H hours.
        Returns: (decisions_dict, is_feasible)
        """
        H = len(load_horizon)
        
        # High-performance HiGHS linear programming formulation
        eta_c_factor = self.eta_ch * self.delta_t / self.eb
        eta_d_factor = self.delta_t / (self.eta_dis * self.eb)

        # 5H variables: p_ch, p_dis, p_grid, p_ren, p_unsupplied
        c = np.concatenate([
            np.full(H, self.c_bat),
            np.full(H, self.c_bat),
            price_horizon,
            np.full(H, -self.c_curt),
            np.full(H, unsupplied_penalty if unsupplied_penalty > 0 else 1e5)
        ])

        A_eq = np.zeros((H, 5 * H))
        for k in range(H):
            A_eq[k, k] = -1.0 # ch
            A_eq[k, H + k] = 1.0 # dis
            A_eq[k, 2*H + k] = 1.0 # grid
            A_eq[k, 3*H + k] = 1.0 # ren
            A_eq[k, 4*H + k] = 1.0 # uns
        b_eq = load_horizon.copy()

        A_ub = np.zeros((2 * H, 5 * H))
        b_ub = np.zeros(2 * H)

        for k in range(H):
            for j in range(k + 1):
                A_ub[k, j] = eta_c_factor
                A_ub[k, H + j] = -eta_d_factor
            b_ub[k] = self.soc_max - current_soc

            if enforce_chance_constraint:
                soc_buf = self.compute_risk_soc_buffer(suspicion_horizon[k], sigma_horizon[k])
                soc_min_eff = min(self.soc_min + soc_buf, self.soc_max - 0.02)
            else:
                soc_min_eff = self.soc_min

            for j in range(k + 1):
                A_ub[H + k, j] = -eta_c_factor
                A_ub[H + k, H + j] = eta_d_factor
            b_ub[H + k] = current_soc - soc_min_eff

        bounds = []
        for k in range(H):
            bounds.append((0.0, self.p_max_ch))
        for k in range(H):
            max_dis = self.p_max_dis - reserve_horizon[k] if enforce_reserve else self.p_max_dis
            bounds.append((0.0, max(0.0, max_dis)))
        for k in range(H):
            bounds.append((0.0, self.p_grid_max))
        for k in range(H):
            bounds.append((0.0, max(0.0, ren_horizon[k])))
        for k in range(H):
            bounds.append((0.0, None if unsupplied_penalty > 0 else 0.0))

        is_feasible = False
        try:
            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                is_feasible = True
                p_ch_0 = float(res.x[0])
                p_dis_0 = float(res.x[H])
                p_grid_0 = float(res.x[2*H])
                p_ren_0 = float(res.x[3*H])
                p_uns_0 = float(res.x[4*H])
                next_soc = current_soc + eta_c_factor * p_ch_0 - eta_d_factor * p_dis_0
                decisions = {
                    "p_ch": p_ch_0,
                    "p_dis": p_dis_0,
                    "p_grid": p_grid_0,
                    "p_ren": p_ren_0,
                    "p_curt": float(ren_horizon[0] - p_ren_0),
                    "p_unsupplied": p_uns_0,
                    "next_soc": float(next_soc)
                }
                return decisions, True
        except Exception:
            pass

        # Section 3.10 Safe Fallback
        # Battery holds state: p_ch = 0, p_dis = 0
        # Grid import supplies residual load
        fused_load_0 = load_horizon[0]
        ren_0 = ren_horizon[0]
        p_ren_safe = min(ren_0, fused_load_0)
        p_curt_safe = ren_0 - p_ren_safe
        residual_demand = max(0.0, fused_load_0 - p_ren_safe)
        p_grid_safe = min(self.p_grid_max, residual_demand)
        unsupplied_safe = max(0.0, residual_demand - p_grid_safe)
        
        fallback = {
            "p_ch": 0.0,
            "p_dis": 0.0,
            "p_grid": p_grid_safe,
            "p_ren": p_ren_safe,
            "p_curt": p_curt_safe,
            "p_unsupplied": unsupplied_safe,
            "next_soc": current_soc
        }
        return fallback, False

    def simulate_dispatch(
        self,
        fused_load: np.ndarray,
        true_load: np.ndarray,
        renewable: np.ndarray,
        price: np.ndarray,
        adaptive_reserve: np.ndarray,
        suspicion_score: np.ndarray,
        sigma_t: np.ndarray,
        enforce_reserve: bool = True,
        enforce_chance_constraint: bool = True,
        unsupplied_penalty: float = 0.0
    ) -> dict:
        """
        Runs rolling-horizon dispatch over the entire test horizon.
        """
        N = len(fused_load)
        soc_series = np.zeros(N + 1)
        soc_series[0] = self.soc_init
        
        p_ch_series = np.zeros(N)
        p_dis_series = np.zeros(N)
        p_grid_series = np.zeros(N)
        p_ren_series = np.zeros(N)
        p_curt_series = np.zeros(N)
        e_uns_series = np.zeros(N)
        is_fallback_series = np.zeros(N, dtype=bool)

        for t in range(N):
            end_idx = min(t + self.horizon_t, N)
            H = end_idx - t
            
            l_h = fused_load[t:end_idx]
            r_h = renewable[t:end_idx]
            p_h = price[t:end_idx]
            res_h = adaptive_reserve[t:end_idx]
            s_h = suspicion_score[t:end_idx]
            sig_h = sigma_t[t:end_idx]

            dec, feasible = self.solve_step(
                current_soc=soc_series[t],
                load_horizon=l_h,
                ren_horizon=r_h,
                price_horizon=p_h,
                reserve_horizon=res_h,
                suspicion_horizon=s_h,
                sigma_horizon=sig_h,
                enforce_reserve=enforce_reserve,
                enforce_chance_constraint=enforce_chance_constraint,
                unsupplied_penalty=unsupplied_penalty
            )

            p_ch_series[t] = dec["p_ch"]
            p_dis_series[t] = dec["p_dis"]
            p_curt_series[t] = dec["p_curt"]
            is_fallback_series[t] = not feasible

            # Physical execution on actual load L_t^true
            actual_l = true_load[t]
            actual_ren = min(renewable[t], actual_l + dec["p_ch"] - dec["p_dis"])
            actual_ren = max(0.0, actual_ren)
            p_ren_series[t] = actual_ren
            
            net_demand = actual_l + dec["p_ch"] - dec["p_dis"] - actual_ren
            p_grid_series[t] = min(self.p_grid_max, max(0.0, net_demand))
            e_uns_series[t] = max(0.0, net_demand - p_grid_series[t])

            # Update true physical SOC
            delta_soc = (self.eta_ch * dec["p_ch"] * self.delta_t / self.eb) - (dec["p_dis"] * self.delta_t / (self.eta_dis * self.eb))
            soc_series[t + 1] = np.clip(soc_series[t] + delta_soc, 0.0, 1.0)

        # Operational metrics
        # J_op = sum(C_t^grid * P_t^grid + C^bat * (P^ch + P^dis) + C^curt * P^curt)  [Eq. 40]
        hourly_cost = price * p_grid_series + self.c_bat * (p_ch_series + p_dis_series) + self.c_curt * p_curt_series
        j_op = float(np.sum(hourly_cost)) / 1000.0 # in k$

        # E_CO2 = sum(e_grid * P_t^grid * Delta_t)  [Eq. 48]
        e_co2 = float(np.sum(self.grid_emission_factor * p_grid_series * self.delta_t)) # in tCO2

        # R_SOC = (1/T) sum(1(SOC_t < SOC_min))  [Eq. 41]
        soc_violations = (soc_series[:-1] < (self.soc_min - 1e-4)).astype(float)
        r_soc = float(np.mean(soc_violations) * 100.0)

        # R_inf = N_fallback / T  [Eq. 42]
        r_inf = float(np.mean(is_fallback_series) * 100.0)

        # Total unserved energy
        e_uns_total = float(np.sum(e_uns_series))

        return {
            "j_op_k_dollars": j_op,
            "e_co2_t": e_co2,
            "r_soc_percent": r_soc,
            "r_inf_percent": r_inf,
            "e_uns_mwh": e_uns_total,
            "soc_series": soc_series,
            "p_ch": p_ch_series,
            "p_dis": p_dis_series,
            "p_grid": p_grid_series,
            "p_ren": p_ren_series,
            "p_curt": p_curt_series,
            "is_fallback": is_fallback_series
        }

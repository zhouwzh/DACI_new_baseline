"""Window-level drift predictor (Sec.4.2 updated).

Thermal: Kalman filter with zeta = 1 - exp(-T_W/tau_bar). Process noise var = sigma_v^2.
  Thermal drive u computed from Eq.(12) using forecast q_mem + hypothesized residency.
Workload: two independent AR(1) channels per node (cmp, mem), RLS on sliding window.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from src.cluster import Cluster


@dataclass
class KalmanNode:
    theta_hat: float
    P: float
    zeta: float
    nu: float
    theta_amb: float
    sigma_v_sq: float


@dataclass
class AR1Channel:
    mu: float
    psi: float
    sigma_eta_sq: float
    history: List[float] = field(default_factory=list)
    last: float = 0.0


@dataclass
class Predictor:
    kalman: List[KalmanNode]
    ar_cmp: List[AR1Channel]
    ar_mem: List[AR1Channel]
    thermal_active: bool
    workload_active: bool

    @staticmethod
    def build(cluster: Cluster, cfg: dict,
              q_cmp_init: np.ndarray, q_mem_init: np.ndarray,
              theta_init: np.ndarray) -> "Predictor":
        therm = cfg["devices"]["thermal"]
        algo = cfg["algo"]["predictor"]
        regime_name = cfg["drift"]["regime"]["active"]
        regime = cfg["drift"]["regime"][regime_name]
        T_W = cluster.T_W_sec

        kalman = []
        ar_cmp = []
        ar_mem = []
        wsc = algo["ar1"]["warm_start_cmp"]
        wsm = algo["ar1"]["warm_start_mem"]

        for n_idx, node in enumerate(cluster.nodes):
            zeta = 1.0 - np.exp(-T_W / max(1e-6, node.tau_bar_s))
            kalman.append(KalmanNode(
                theta_hat=float(theta_init[n_idx]),
                P=algo["kalman"]["init_P_var"],
                zeta=zeta, nu=node.nu,
                theta_amb=cluster.theta_amb,
                sigma_v_sq=therm["sensor_noise_var"],
            ))
            # cmp: warm-start mu as fraction of mu_n (capacity)
            mu_cmp_ws = wsc["mu_frac_of_cap"] * node.mu_flops
            sigma_cmp_ws = wsc["sigma_eta_sq_rel"] * (node.mu_flops ** 2)
            ar_cmp.append(AR1Channel(
                mu=mu_cmp_ws, psi=wsc["psi"],
                sigma_eta_sq=sigma_cmp_ws,
                last=float(q_cmp_init[n_idx]),
                history=[float(q_cmp_init[n_idx])],
            ))
            # mem: warm-start in bytes
            mu_mem_ws = wsm["mu_gb"] * (1024 ** 3)
            sigma_mem_ws = wsm["sigma_eta_sq_gb_sq"] * ((1024 ** 3) ** 2)
            ar_mem.append(AR1Channel(
                mu=mu_mem_ws, psi=wsm["psi"],
                sigma_eta_sq=sigma_mem_ws,
                last=float(q_mem_init[n_idx]),
                history=[float(q_mem_init[n_idx])],
            ))
        return Predictor(
            kalman=kalman, ar_cmp=ar_cmp, ar_mem=ar_mem,
            thermal_active=regime.get("thermal_active", True),
            workload_active=regime.get("workload_active", True),
        )

    def kalman_update(self, n_idx: int, y: float, u_last: float) -> None:
        kn = self.kalman[n_idx]
        theta_pred = (1 - kn.zeta) * kn.theta_hat + kn.zeta * (kn.theta_amb + kn.nu * u_last)
        P_pred = (1 - kn.zeta) ** 2 * kn.P + kn.sigma_v_sq
        S = P_pred + kn.sigma_v_sq
        K = P_pred / S
        kn.theta_hat = theta_pred + K * (y - theta_pred)
        kn.P = (1 - K) * P_pred

    def _rls_channel(self, ar: AR1Channel, obs: float, rls_window: int) -> None:
        ar.history.append(obs)
        if len(ar.history) > rls_window:
            ar.history = ar.history[-rls_window:]
        ar.last = obs
        if len(ar.history) >= 5:
            xs = np.array(ar.history[:-1])
            ys = np.array(ar.history[1:])
            mu = float(np.mean(ar.history))
            xc = xs - mu
            yc = ys - mu
            denom = float(np.sum(xc ** 2))
            psi = float(np.sum(xc * yc) / denom) if denom > 1e-9 else ar.psi
            psi = max(-0.99, min(0.99, psi))
            preds = mu + psi * (xs - mu)
            resid = ys - preds
            sigma_eta_sq = float(np.var(resid)) if len(resid) > 1 else ar.sigma_eta_sq
            ar.mu = mu
            ar.psi = psi
            ar.sigma_eta_sq = max(1e-9, sigma_eta_sq)

    def ar_update(self, n_idx: int, q_cmp_obs: float, q_mem_obs: float,
                  rls_window: int) -> None:
        self._rls_channel(self.ar_cmp[n_idx], q_cmp_obs, rls_window)
        self._rls_channel(self.ar_mem[n_idx], q_mem_obs, rls_window)

    def forecast(self, cluster: Cluster, u_future: np.ndarray, K: int) -> Dict:
        """Return forecast dict.

        u_future: (N, K) hypothesized thermal drive (memory-occupancy ratio) per
          window. Typically computed by controller from current (a, b) + forecast q_mem.
        """
        N = cluster.N
        phi_hat = np.ones((N, K))
        phi_var = np.zeros((N, K))
        phi_inf = np.ones(N)
        q_cmp_hat = np.zeros((N, K))
        q_mem_hat = np.zeros((N, K))

        for n_idx, node in enumerate(cluster.nodes):
            kn = self.kalman[n_idx]
            ac = self.ar_cmp[n_idx]
            am = self.ar_mem[n_idx]

            theta_k = kn.theta_hat
            Pk = kn.P
            cmp_k = ac.last
            mem_k = am.last

            for k in range(K):
                u_k = float(u_future[n_idx, k]) if u_future.ndim == 2 else float(u_future[n_idx])
                if self.thermal_active:
                    theta_k = (1 - kn.zeta) * theta_k + kn.zeta * (kn.theta_amb + kn.nu * u_k)
                    Pk = (1 - kn.zeta) ** 2 * Pk + kn.sigma_v_sq
                else:
                    theta_k = cluster.theta_amb
                    Pk = 0.0
                phi_th = 1.0 + node.gamma * max(0.0, theta_k - node.theta_th)
                dphi_th = node.gamma if theta_k > node.theta_th else 0.0
                var_phi_th = (dphi_th ** 2) * Pk

                if self.workload_active:
                    cmp_fcst = ac.mu + (ac.psi ** (k + 1)) * (ac.last - ac.mu)
                    cmp_fcst = max(0.0, min(node.mu_flops, cmp_fcst))
                    cmp_var = ac.sigma_eta_sq * (1 - ac.psi ** (2 * (k + 1))) / max(1e-9, 1 - ac.psi ** 2)
                    mem_fcst = am.mu + (am.psi ** (k + 1)) * (am.last - am.mu)
                    mem_fcst = max(0.0, min(node.m_bytes, mem_fcst))
                else:
                    cmp_fcst = 0.0
                    cmp_var = 0.0
                    mem_fcst = 0.0

                phi_wk = 1.0 + node.rho * (cmp_fcst / node.mu_flops)
                var_phi_wk = (node.rho / node.mu_flops) ** 2 * cmp_var

                phi_hat[n_idx, k] = phi_th * phi_wk
                phi_var[n_idx, k] = (phi_wk ** 2) * var_phi_th + (phi_th ** 2) * var_phi_wk
                q_cmp_hat[n_idx, k] = cmp_fcst
                q_mem_hat[n_idx, k] = mem_fcst

            # Long-run tail
            u_ss = ac.mu / node.mu_flops if node.mu_flops > 0 else 0.0
            theta_ss = cluster.theta_amb + node.nu * u_ss
            phi_th_inf = 1.0 + node.gamma * max(0.0, theta_ss - node.theta_th)
            phi_inf[n_idx] = phi_th_inf * (1.0 + node.rho * u_ss)

        return {"phi_hat": phi_hat, "phi_var": phi_var, "phi_infinity": phi_inf,
                "q_cmp_hat": q_cmp_hat, "q_mem_hat": q_mem_hat}


def adaptive_horizon(phi_hat: np.ndarray, phi_var: np.ndarray, tau: float,
                     H_max: int, a_0: List[int], min_horizon: int = 1) -> int:
    K = phi_hat.shape[1]
    H_star = min_horizon
    for k in range(K):
        ok = True
        for n in a_0:
            denom = max(1e-6, phi_hat[n, k])
            cv = np.sqrt(max(0.0, phi_var[n, k])) / denom
            if cv > tau:
                ok = False
                break
        if ok:
            H_star = k + 1
        else:
            break
    return min(H_star, H_max)

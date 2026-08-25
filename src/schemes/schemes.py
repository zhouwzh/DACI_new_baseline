"""DACI controller + baseline schemes (updated per paper rewrite)."""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.cluster import Cluster
from src.model_spec import ModelSpec
from src.predictor import Predictor, adaptive_horizon
from src.dp import solve_initial_dp, solve_runtime_dp, eval_surrogate, _ordered_placements
from src.cost_model import T_decode_window, compute_u_thermal, stage_mem_bytes


@dataclass
class SchemeState:
    name: str
    S: int = 0
    a: List[int] = field(default_factory=list)
    b: List[int] = field(default_factory=list)
    predictor: Optional[Predictor] = None
    TPOT_nominal_s: float = 0.05
    W_sec: float = 1.0
    last_committed_window: int = 0
    n_reconfigs: int = 0
    total_overhead_s: float = 0.0


def _bootstrap_TPOT(cluster: Cluster, ms: ModelSpec, S: int, a: List[int], b: List[int],
                    P: int, link_obs: dict) -> float:
    phi_unit = np.ones(cluster.N)
    return T_decode_window(cluster, ms, b, a, phi_unit, link_obs, P, 0)


class DACIScheme:
    name = "DACI"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.algo = cfg["algo"]
        self.lambda_slack = self.algo["switching"]["lambda_slack"]
        self.ablation = cfg.get("ablation", {}).get("mode", "none")

    def decide_initial(self, cluster: Cluster, ms: ModelSpec,
                       obs: dict, P: int, G_hat: int) -> SchemeState:
        W = self.algo["window"]["W_tokens"]
        S_min = self.algo["stage_range"]["S_min"]
        S_max = self.algo["stage_range"]["S_max"]

        phi_0 = self._phi_from_obs(obs, cluster)
        phi_avg = phi_0.copy()
        K_0 = self.algo["window"]["H_max"]

        best = solve_initial_dp(
            cluster, ms, S_min, S_max, ms.L, P, G_hat,
            phi_0, phi_avg, obs["link_obs"], obs["q_mem_obs"],
            obs["theta_obs"], K_0,
        )
        if best["S"] is None:
            raise RuntimeError("No feasible initial deployment found")

        TPOT_nom = _bootstrap_TPOT(cluster, ms, best["S"], best["a"], best["b"], P, obs["link_obs"])
        W_sec = W * TPOT_nom

        pred = Predictor.build(cluster, self.cfg,
                               obs["q_cmp_obs"], obs["q_mem_obs"], obs["theta_obs"])
        return SchemeState(
            name=self.name, S=best["S"], a=best["a"], b=best["b"],
            predictor=pred, TPOT_nominal_s=TPOT_nom, W_sec=W_sec,
        )

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])

        H_max = self.algo["window"]["H_max"]
        tau_cv = self.algo["window"]["tau_variance"]
        u_future = np.tile(u_curr.reshape(-1, 1), (1, H_max))
        fcst = st.predictor.forecast(cluster, u_future, H_max)

        # --- Ablation: no_predictor (persistence across horizon AND tail) ---
        # True zero-knowledge baseline: hold current phi constant for both the
        # K_r-step rollout AND the long-run tail. Otherwise phi_infinity (which
        # uses AR(1) mu_cmp via predictor.forecast) leaks forecast information
        # into the surrogate's tail term, which dominates when G_rem >> K_r*W.
        if self.ablation == "no_predictor":
            phi0 = fcst["phi_hat"][:, 0:1]
            fcst["phi_hat"] = np.tile(phi0, (1, H_max))
            fcst["phi_var"] = np.zeros_like(fcst["phi_var"])
            fcst["phi_infinity"] = fcst["phi_hat"][:, 0].copy()
            qmem0 = fcst["q_mem_hat"][:, 0:1]
            fcst["q_mem_hat"] = np.tile(qmem0, (1, H_max))

        # --- Ablation: no_adaptive_H (fixed at H_max) ---
        if self.ablation == "no_adaptive_H":
            H_r = H_max
        else:
            ah = self.algo["predictor"]["adaptive_horizon"]
            # `enabled` used to be ignored: the adaptive horizon ran
            # unconditionally and setting the flag false did nothing. The
            # no_adaptive_H ablation was the only way to disable it, so the
            # config knob was a lie. Default is true, so honouring it changes
            # no existing result.
            H_r = (adaptive_horizon(fcst["phi_hat"], fcst["phi_var"], tau_cv,
                                    H_max, st.a, min_horizon=ah["min_horizon"])
                   if ah.get("enabled", True) else H_max)
        K_r = min(H_r, max(1, (G_rem + W - 1) // W))

        # --- Ablation: no_bottleneck (greedy, ignores H_s during search) ---
        if self.ablation == "no_bottleneck":
            from src.dp import solve_runtime_dp_greedy
            J_new, b_new = solve_runtime_dp_greedy(
                cluster, ms, st.S, st.a, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
                fcst["phi_hat"], fcst["phi_infinity"], fcst["q_mem_hat"], obs["link_obs"]
            )
        else:
            max_shift = self.algo.get("dp", {}).get("max_boundary_shift", None)
            J_new, b_new = solve_runtime_dp(
                cluster, ms, st.S, st.a, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
                fcst["phi_hat"], fcst["phi_infinity"], fcst["q_mem_hat"], obs["link_obs"],
                max_boundary_shift=max_shift,
            )

        J_incumbent = eval_surrogate(
            cluster, ms, st.S, st.a, st.b, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            fcst["phi_hat"], fcst["phi_infinity"], obs["link_obs"]
        )

        # --- Ablation: no_lazy (always accept if candidate exists) ---
        if self.ablation == "no_lazy":
            accepted = b_new is not None and b_new != st.b
        else:
            accepted = b_new is not None and J_new < J_incumbent - self.lambda_slack

        b_out = b_new if accepted else st.b

        TPOT_nom = _bootstrap_TPOT(cluster, ms, st.S, st.a, b_out, P, obs["link_obs"])
        st.TPOT_nominal_s = TPOT_nom
        st.W_sec = W * TPOT_nom

        meta = {
            "H_r_star": H_r, "K_r": K_r,
            "J_new": J_new, "J_incumbent": J_incumbent,
            "phi_hat_curr": fcst["phi_hat"][:, 0].tolist(),
            "phi_hat_horizon": fcst["phi_hat"].tolist(),
            "b_new_candidate": b_new,
            "u_thermal": u_curr.tolist(),
            "ablation": self.ablation,
        }
        return b_out, accepted, meta

    def _phi_from_obs(self, obs, cluster: Cluster) -> np.ndarray:
        """Instantaneous phi from current observations. phi_wk = 1 + rho * q_cmp/mu_n."""
        phi = np.ones(cluster.N)
        for n_idx, node in enumerate(cluster.nodes):
            phi_th = 1.0 + node.gamma * max(0.0, obs["theta_obs"][n_idx] - node.theta_th)
            phi_wk = 1.0 + node.rho * (obs["q_cmp_obs"][n_idx] / node.mu_flops)
            phi[n_idx] = phi_th * phi_wk
        return phi

    def _compute_u(self, cluster, ms, a, b, q_mem, P, t_r):
        """Thermal drive u per Eq.(12): memory-occupancy ratio."""
        return compute_u_thermal(cluster, ms, a, b, q_mem, P, t_r)


class SDAScheme(DACIScheme):
    name = "SDA"

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        meta = {"H_r_star": 0, "K_r": 0, "J_new": None, "J_incumbent": None,
                "phi_hat_curr": [], "b_new_candidate": None,
                "u_thermal": u_curr.tolist()}
        return st.b, False, meta



def _feasible_pool(cluster, ms, st, obs, P, t_end, q_mem_by_node):
    """Nodes that could host at least the SMALLEST stage, over ALL N nodes.

    Deliberately does not exclude nodes already hosting a stage. Reassigning
    which stage sits on an already-used node is still a placement change: it
    reloads weights, migrates KV and pays H_swap. The previous code skipped
    every node in `set(st.a)`, which restricted both RT and FM to swapping onto
    an IDLE node -- so at S=N the candidate pool was empty in every window, RT
    performed zero reconfigurations and was numerically identical to SDA, and
    FM collapsed onto DACI. That made them strawmen rather than baselines.

    The filter here is deliberately LOOSE. Screening on the largest stage
    instead looks safer and is wrong: with S stages and a tightened memory
    budget (the Orin Nano is 8 GB, not 12) the pool shrinks below S, no
    permutation exists at all, and the search silently degenerates to "never
    move" -- which is the same strawman in a new disguise. Per-stage
    feasibility is checked inside the enumeration, where it belongs, because
    whether a node fits depends on which stage it is asked to host.
    """
    smallest = min(st.b[i + 1] - st.b[i] for i in range(st.S))
    need = _stage_mem_need(ms, smallest, P, t_end)
    pool = []
    for n in range(cluster.N):
        m_eff = max(0.0, cluster.nodes[n].m_bytes - q_mem_by_node[n])
        if need <= m_eff:
            pool.append(n)
    return pool


def _best_placement(cluster, ms, st, pool, P, t_r, G_rem, W, K_r,
                    phi_hat_horizon, phi_infinity, link_obs,
                    q_mem_by_node, t_end, max_perms=20000):
    """Search ordered S-permutations of `pool`, scored by the surrogate.

    This is the joint placement search paper Sec. 5.1.5 describes for RT
    ("re-runs the joint placement-boundary DP"). Placement is chosen against
    the same surrogate objective the DP optimises -- including H_stage, so a
    candidate is charged for the weight reload, KV migration and H_swap that
    moving to it would cost -- and the boundary is then re-optimised under the
    winner by the caller's runtime DP.

    Scoring with the surrogate rather than a full DP per permutation keeps this
    affordable: P(8,4)=1680 candidates cost one O(S*L) evaluation each, versus
    1680 dynamic programs.
    """
    best_a, best_J = None, float("inf")
    n_eval = 0
    for a_cand in _ordered_placements(pool, st.S):
        # Per-stage memory feasibility: node a_cand[s-1] must hold stage s's
        # own blocks, which is what makes a loose pool filter safe.
        ok = True
        for s in range(1, st.S + 1):
            n_blocks = st.b[s] - st.b[s - 1]
            node = cluster.nodes[a_cand[s - 1]]
            m_eff = max(0.0, node.m_bytes - q_mem_by_node[a_cand[s - 1]])
            if _stage_mem_need(ms, n_blocks, P, t_end) > m_eff:
                ok = False
                break
        if not ok:
            continue
        n_eval += 1
        if n_eval > max_perms:
            break
        J = eval_surrogate(cluster, ms, st.S, a_cand, st.b, st.b, st.a,
                           ms.L, P, t_r, G_rem, W, K_r,
                           phi_hat_horizon, phi_infinity, link_obs)
        if J < best_J:
            best_J, best_a = J, list(a_cand)
    return best_a, best_J, n_eval


class RTScheme(DACIScheme):
    name = "RT"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.trigger_ratio = cfg["experiment"]["rt_baseline"]["trigger_ratio"]
        self.TPOT_baseline_nominal = None
        # AutoPipe-style trigger cool-down (default 50 windows; configurable)
        self.cool_down_windows = cfg["experiment"]["rt_baseline"].get("cool_down_windows", 50)
        self._last_trigger_r = -10**9

    def decide_initial(self, cluster, ms, obs, P, G_hat):
        st = super().decide_initial(cluster, ms, obs, P, G_hat)
        self.TPOT_baseline_nominal = st.TPOT_nominal_s
        self._last_trigger_r = -10**9
        return st

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        phi_now = self._phi_from_obs(obs, cluster)
        tpot_now = T_decode_window(cluster, ms, st.b, st.a, phi_now, obs["link_obs"], P, t_r)

        # Current window index (approximation: t_r / W)
        r_now = t_r // W
        on_cool_down = (r_now - self._last_trigger_r) < self.cool_down_windows

        triggered = (tpot_now > self.trigger_ratio * self.TPOT_baseline_nominal) and not on_cool_down
        if triggered:
            # Greedy placement swap: find most-degraded stage, swap to best cool alternative
            stage_phi = [phi_now[st.a[s - 1]] for s in range(1, st.S + 1)]
            hot_stage_idx = int(np.argmax(stage_phi))  # 0-indexed within stages
            hot_node = st.a[hot_stage_idx]

            # Joint placement search over ALL N nodes (Sec. 5.1.5), not just
            # idle ones -- see _feasible_pool for why the old restriction made
            # this a strawman.
            H_max = self.algo["window"]["H_max"]
            K_r = min(H_max, max(1, (G_rem + W - 1) // W))
            phi_hat_horizon = np.tile(phi_now.reshape(-1, 1), (1, H_max))
            phi_inf = phi_now.copy()
            q_mem_hat = np.tile(obs["q_mem_obs"].reshape(-1, 1), (1, H_max))
            pool = _feasible_pool(cluster, ms, st, obs, P, t_r + G_rem,
                                  obs["q_mem_obs"])
            a_new, _J_cand, _n_eval = _best_placement(
                cluster, ms, st, pool, P, t_r, G_rem, W, K_r,
                phi_hat_horizon, phi_inf, obs["link_obs"],
                obs["q_mem_obs"], t_r + G_rem)
            if a_new is not None:
                max_shift = self.algo.get("dp", {}).get("max_boundary_shift", None)
                J_new, b_new = solve_runtime_dp(
                    cluster, ms, st.S, a_new, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
                    phi_hat_horizon, phi_inf, q_mem_hat, obs["link_obs"],
                    max_boundary_shift=max_shift,
                )
                # Reactive: accept if J_new improves (no lazy slack; trigger has already fired)
                if b_new is not None and J_new < tpot_now * G_rem:
                    old_a = st.a
                    changed = list(a_new) != list(st.a)
                    st.a = a_new
                    self._last_trigger_r = r_now
                    meta = {"H_r_star": 0, "K_r": K_r, "J_new": J_new,
                            "J_incumbent": tpot_now * G_rem, "phi_hat_curr": phi_now.tolist(),
                            "b_new_candidate": b_new, "placement_changed": changed,
                            "a_prev_for_omega": old_a, "u_thermal": u_curr.tolist()}
                    return b_new, True, meta

        meta = {"H_r_star": 0, "K_r": 0, "J_new": None, "J_incumbent": None,
                "phi_hat_curr": phi_now.tolist(), "b_new_candidate": None,
                "u_thermal": u_curr.tolist()}
        return st.b, False, meta


class FMScheme(DACIScheme):
    name = "FM"

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        H_max = self.algo["window"]["H_max"]
        tau_cv = self.algo["window"]["tau_variance"]
        u_future = np.tile(u_curr.reshape(-1, 1), (1, H_max))
        fcst = st.predictor.forecast(cluster, u_future, H_max)
        ah = self.algo["predictor"]["adaptive_horizon"]
        H_r = (adaptive_horizon(fcst["phi_hat"], fcst["phi_var"], tau_cv,
                                H_max, st.a, min_horizon=ah["min_horizon"])
               if ah.get("enabled", True) else H_max)
        K_r = min(H_r, max(1, (G_rem + W - 1) // W))

        # Joint placement search over ALL N nodes, on the forecast horizon.
        phi_hat_curr = fcst["phi_hat"][:, 0]
        # Memory feasibility uses the horizon max, consistent with DP Eq.(26).
        q_mem_max_h = np.max(fcst["q_mem_hat"][:, :K_r], axis=1)
        pool = _feasible_pool(cluster, ms, st, obs, P, t_r + G_rem, q_mem_max_h)
        a_best, _J_cand, _n_eval = _best_placement(
            cluster, ms, st, pool, P, t_r, G_rem, W, K_r,
            fcst["phi_hat"], fcst["phi_infinity"], obs["link_obs"],
            q_mem_max_h, t_r + G_rem)

        a_new = list(a_best) if a_best is not None else list(st.a)
        swapped = a_new != list(st.a)

        # Re-optimize b under a_new (or a unchanged if no swap)
        max_shift = self.algo.get("dp", {}).get("max_boundary_shift", None)
        J_new, b_new = solve_runtime_dp(
            cluster, ms, st.S, a_new, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            fcst["phi_hat"], fcst["phi_infinity"], fcst["q_mem_hat"], obs["link_obs"],
            max_boundary_shift=max_shift if not swapped else None,
        )
        J_inc = eval_surrogate(
            cluster, ms, st.S, st.a, st.b, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            fcst["phi_hat"], fcst["phi_infinity"], obs["link_obs"]
        )
        accepted = b_new is not None and J_new < J_inc - self.lambda_slack
        if accepted:
            old_a = st.a
            if swapped:
                st.a = a_new
            b_out = b_new
        else:
            b_out = st.b
            old_a = st.a

        TPOT_nom = _bootstrap_TPOT(cluster, ms, st.S, st.a, b_out, P, obs["link_obs"])
        st.TPOT_nominal_s = TPOT_nom
        st.W_sec = W * TPOT_nom

        meta = {"H_r_star": H_r, "K_r": K_r, "J_new": J_new, "J_incumbent": J_inc,
                "phi_hat_curr": phi_hat_curr.tolist(),
                "b_new_candidate": b_new, "placement_changed": swapped and accepted,
                "a_prev_for_omega": old_a,
                "u_thermal": u_curr.tolist()}
        return b_out, accepted, meta


def _stage_mem_need(ms, n_blocks, P, t_end):
    """Conservative memory need for a stage with n_blocks up to decode step t_end."""
    from src.cost_model import stage_mem_bytes
    return stage_mem_bytes(ms, n_blocks, P, t_end)


class OracleScheme(DACIScheme):
    name = "OR"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._gt_phi_future = None
        self._gt_q_mem_future = None

    def set_ground_truth(self, phi_future: np.ndarray, q_mem_future: np.ndarray):
        self._gt_phi_future = phi_future
        self._gt_q_mem_future = q_mem_future

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        H_max = self.algo["window"]["H_max"]
        if self._gt_phi_future is None:
            phi_hat = np.ones((cluster.N, H_max))
            q_mem_hat = np.zeros((cluster.N, H_max))
        else:
            phi_hat = self._gt_phi_future
            q_mem_hat = self._gt_q_mem_future
        phi_inf = phi_hat[:, -1]
        K_r = min(H_max, max(1, (G_rem + W - 1) // W))
        J_new, b_new = solve_runtime_dp(
            cluster, ms, st.S, st.a, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            phi_hat, phi_inf, q_mem_hat, obs["link_obs"]
        )
        J_inc = eval_surrogate(
            cluster, ms, st.S, st.a, st.b, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            phi_hat, phi_inf, obs["link_obs"]
        )
        accepted = b_new is not None and J_new < J_inc
        b_out = b_new if accepted else st.b
        meta = {"H_r_star": H_max, "K_r": K_r, "J_new": J_new, "J_incumbent": J_inc,
                "phi_hat_curr": phi_hat[:, 0].tolist(), "b_new_candidate": b_new,
                "u_thermal": u_curr.tolist()}
        return b_out, accepted, meta


SCHEMES = {"DACI": DACIScheme, "SDA": SDAScheme, "RT": RTScheme,
           "FM": FMScheme, "OR": OracleScheme}


def build_scheme(name: str, cfg: dict):
    if name not in SCHEMES:
        raise KeyError(f"Unknown scheme: {name}")
    return SCHEMES[name](cfg)
"""Bottleneck-augmented DP (Proposition 1) and both instantiations."""
from typing import List, Tuple, Dict, Optional
import numpy as np
from src.cluster import Cluster
from src.model_spec import ModelSpec
from src.cost_model import stage_mem_bytes, H_stage


def placement_score(cluster: Cluster, phi_avg: np.ndarray) -> np.ndarray:
    """Score for top-K pruning Eq.(20). Uses baseline-conditioned workload drift:
      phi_baseline_n = 1 + rho_n * q_cmp_base_n / mu_n
    plus the passive thermal drift estimate in phi_avg (from current theta obs).
    Thermal-foreground term is stage-size-dependent so it's enforced later in DP;
    here we only need a per-node coarse ranking.
    """
    scores = np.zeros(cluster.N)
    for n_idx, node in enumerate(cluster.nodes):
        phi_wk_base = 1.0 + node.rho * (node.q_cmp_base_flops / node.mu_flops)
        # Combine with passive thermal estimate from phi_avg (may be 1 at t=0)
        phi_combined = max(phi_avg[n_idx], 1.0) * phi_wk_base
        scores[n_idx] = node.mu_flops / max(1e-6, phi_combined)
    return scores


def prune_placements(cluster: Cluster, ms: ModelSpec, phi_avg: np.ndarray,
                     S_max: int, P: int, G_hat: int, q_mem: np.ndarray) -> List[int]:
    K = min(cluster.N, 2 * S_max)
    feasible_nodes = []
    for n_idx, node in enumerate(cluster.nodes):
        m_eff = max(0.0, node.m_bytes - q_mem[n_idx])
        min_mem = ms.omega_block_bytes + (P + G_hat) * ms.chi_unit_block_bytes_per_token
        if min_mem <= m_eff:
            feasible_nodes.append(n_idx)
    scores = placement_score(cluster, phi_avg)
    feasible_nodes.sort(key=lambda n: -scores[n])
    return feasible_nodes[:K]


def _ordered_placements(pool: List[int], S: int):
    n = len(pool)
    if S > n:
        return
    def perm(chosen, remaining):
        if len(chosen) == S:
            yield [pool[i] for i in chosen]
            return
        for i, r in enumerate(remaining):
            yield from perm(chosen + [r], remaining[:i] + remaining[i + 1:])
    yield from perm([], list(range(n)))


def _phi_tilde_conditional(cluster: Cluster, ms: ModelSpec,
                            n: int, n_blocks: int, P: int, G_hat: int,
                            theta_obs: np.ndarray, K_0: int) -> float:
    """Conditional drift tilde_phi_n(L_s) per Eq.(u_fg, theta_blend, phi_cond).

    Foreground-induced steady-state utilization:
        u_fg = min(1, (omega(L_s) + chi_bar(P, G_hat/2) + q_mem_base) / m)
    Kalman steady-state mean:
        theta_inf = theta_amb + nu * u_fg
    Blend with current observation to guard against transient arrival anomalies:
        alpha = 1 - exp(-K_0 / tau_bar)
        theta_tilde = alpha * theta_inf + (1 - alpha) * theta_obs
    Drift:
        phi_th = 1 + gamma * (theta_tilde - theta_th)_+
        phi_wk = 1 + rho * q_cmp_base / mu    (Table 1 baseline, r=0 has no AR(1) samples)
        phi_tilde = phi_th * phi_wk

    K_0 is unitless (pass H_max numeric value); tau_bar is in seconds. The ratio
    K_0/tau_bar is a shortcut that gives alpha ~= 0.9 for typical configs.
    """
    node = cluster.nodes[n]
    omega_s = ms.omega_stage_bytes(n_blocks)
    chi_bar = n_blocks * (P + G_hat // 2) * ms.chi_unit_block_bytes_per_token
    q_mem_bg = node.q_mem_base_bytes
    u_fg = min(1.0, (omega_s + chi_bar + q_mem_bg) / node.m_bytes)
    theta_inf = cluster.theta_amb + node.nu * u_fg
    alpha = 1.0 - np.exp(-K_0 / node.tau_bar_s)
    theta_tilde = alpha * theta_inf + (1.0 - alpha) * theta_obs[n]
    phi_th = 1.0 + node.gamma * max(0.0, theta_tilde - node.theta_th)
    phi_wk = 1.0 + node.rho * (node.q_cmp_base_flops / node.mu_flops)
    return phi_th * phi_wk


def init_dp_per_placement(cluster: Cluster, ms: ModelSpec,
                          a: List[int], S: int, L: int,
                          P: int, G_hat: int,
                          phi_hat_0: np.ndarray, phi_avg: np.ndarray,
                          link_state_0: Dict[Tuple[int, int], Tuple[float, float]],
                          q_mem: np.ndarray,
                          theta_obs: np.ndarray, K_0: int
                          ) -> Tuple[Optional[float], Optional[List[int]]]:
    """Initial DP with conditional drift (Eq. phi_cond) using blended theta.

    phi_avg argument kept for backward compatibility (unused in V).
    """
    D_s0 = [0.0] * (S + 1)
    for s in range(1, S + 1):
        if s < S:
            a_s = a[s - 1]
            a_next = a[s]
            alpha, beta = link_state_0[(a_s, a_next)]
            D_s0[s] = (alpha + beta * ms.z_pf_bytes_with_P(P)) + G_hat * (alpha + beta * ms.z_dec_bytes())

    INF = float("inf")
    dp: Dict[Tuple[int, int], Dict[float, Tuple[float, Tuple[int, float]]]] = {}
    dp[(0, 0)] = {0.0: (0.0, (-1, 0.0))}

    for s in range(1, S + 1):
        a_s = a[s - 1]
        node = cluster.nodes[a_s]
        m_eff = max(0.0, node.m_bytes - q_mem[a_s])
        for j in range(s, L - (S - s) + 1):
            for i in range(s - 1, j):
                n_blocks = j - i
                if stage_mem_bytes(ms, n_blocks, P, P) > m_eff:
                    continue
                if (i, s - 1) not in dp:
                    continue
                L_load = ms.omega_stage_bytes(n_blocks) / node.B_load_bps
                kappa_pf = ms.kappa_pf_stage(n_blocks, P)
                C_pf = (kappa_pf / node.mu_flops) * phi_hat_0[a_s]
                t_mid = G_hat // 2
                kappa_dec = ms.kappa_dec_stage(n_blocks, P, t_mid)
                # CONDITIONAL drift with blended theta: phi_tilde depends on (n, n_blocks)
                phi_tilde = _phi_tilde_conditional(cluster, ms, a_s, n_blocks, P, G_hat,
                                                    theta_obs, K_0)
                C_dec_cond = (kappa_dec / node.mu_flops) * phi_tilde
                V = C_pf + G_hat * C_dec_cond + D_s0[s]

                for u_prev, (f_prev, _) in dp[(i, s - 1)].items():
                    u_new = max(u_prev, L_load)
                    f_new = f_prev + V
                    key = (j, s)
                    if key not in dp:
                        dp[key] = {}
                    if u_new not in dp[key] or f_new < dp[key][u_new][0]:
                        dp[key][u_new] = (f_new, (i, u_prev))

    if (L, S) not in dp:
        return None, None

    best_J = INF
    best_u = None
    for u, (f, _) in dp[(L, S)].items():
        total = f + u
        if total < best_J:
            best_J = total
            best_u = u
    if best_u is None:
        return None, None

    b = [0] * (S + 1)
    b[S] = L
    cur_j, cur_s, cur_u = L, S, best_u
    while cur_s > 0:
        _, (prev_i, prev_u) = dp[(cur_j, cur_s)][cur_u]
        b[cur_s - 1] = prev_i
        cur_j, cur_s, cur_u = prev_i, cur_s - 1, prev_u
    return best_J, b


def solve_initial_dp(cluster: Cluster, ms: ModelSpec,
                     S_min: int, S_max: int, L: int,
                     P: int, G_hat: int,
                     phi_hat_0: np.ndarray, phi_avg: np.ndarray,
                     link_state_0: Dict[Tuple[int, int], Tuple[float, float]],
                     q_mem: np.ndarray,
                     theta_obs: np.ndarray, K_0: int) -> Dict:
    pool = prune_placements(cluster, ms, phi_avg, S_max, P, G_hat, q_mem)
    best = {"J": float("inf"), "S": None, "a": None, "b": None}
    for S in range(S_min, S_max + 1):
        if S > len(pool):
            break
        for a in _ordered_placements(pool, S):
            J, b = init_dp_per_placement(cluster, ms, a, S, L, P, G_hat,
                                          phi_hat_0, phi_avg, link_state_0, q_mem,
                                          theta_obs, K_0)
            if J is not None and J < best["J"]:
                best = {"J": J, "S": S, "a": a, "b": b}
    return best


def solve_runtime_dp(cluster: Cluster, ms: ModelSpec,
                     S: int, a: List[int], b_prev: List[int], a_prev: List[int],
                     L: int, P: int, t_r: int, G_rem: int, W: int, K_r: int,
                     phi_hat_horizon: np.ndarray,
                     phi_infinity: np.ndarray,
                     q_mem_horizon: np.ndarray,
                     link_obs: Dict[Tuple[int, int], Tuple[float, float]],
                     max_boundary_shift: Optional[int] = None,
                     ) -> Tuple[Optional[float], Optional[List[int]]]:
    """Runtime DP Eq.(19)-(21). q_mem_horizon: (N, K_r) forecast.
    If max_boundary_shift is set, enforces |b_s^r - b_s^{r-1}| <= max_boundary_shift
    for all internal boundaries s=1..S-1 (Eq.(27)).
    """
    INF = float("inf")
    D_s_per_tok = [0.0] * (S + 1)
    for s in range(1, S + 1):
        if s < S:
            alpha, beta = link_obs[(a[s - 1], a[s])]
            D_s_per_tok[s] = alpha + beta * ms.z_dec_bytes()

    tail = max(0, G_rem - K_r * W)
    horizon_phi_sum = np.zeros(cluster.N)
    for n in range(cluster.N):
        horizon_phi_sum[n] = W * float(np.sum(phi_hat_horizon[n, :K_r])) + tail * phi_infinity[n]

    t_mid = t_r + min(G_rem, K_r * W) // 2
    q_mem_max = np.max(q_mem_horizon[:, :K_r], axis=1)

    dp: Dict[Tuple[int, int], Dict[float, Tuple[float, Tuple[int, float]]]] = {}
    dp[(0, 0)] = {0.0: (0.0, (-1, 0.0))}

    for s in range(1, S + 1):
        a_s = a[s - 1]
        node = cluster.nodes[a_s]
        m_eff_min = max(0.0, node.m_bytes - q_mem_max[a_s])
        b_prev_left = b_prev[s - 1] if len(b_prev) > s - 1 else None
        b_prev_right = b_prev[s] if len(b_prev) > s else None
        for j in range(s, L - (S - s) + 1):
            if (max_boundary_shift is not None and s < S
                    and b_prev_right is not None
                    and abs(j - b_prev_right) > max_boundary_shift):
                continue
            for i in range(s - 1, j):
                if (max_boundary_shift is not None and s > 1
                        and b_prev_left is not None
                        and abs(i - b_prev_left) > max_boundary_shift):
                    continue
                n_blocks = j - i
                if stage_mem_bytes(ms, n_blocks, P, t_r + K_r * W) > m_eff_min:
                    continue
                if (i, s - 1) not in dp:
                    continue
                kappa_dec_mid = ms.kappa_dec_stage(n_blocks, P, t_mid)
                xi = kappa_dec_mid / node.mu_flops
                V = xi * horizon_phi_sum[a_s] + G_rem * D_s_per_tok[s]
                h_val = H_stage(cluster, ms, s, i, j, a, b_prev, a_prev, P, t_r, link_obs)
                for u_prev, (f_prev, _) in dp[(i, s - 1)].items():
                    u_new = max(u_prev, h_val)
                    f_new = f_prev + V
                    key = (j, s)
                    if key not in dp:
                        dp[key] = {}
                    if u_new not in dp[key] or f_new < dp[key][u_new][0]:
                        dp[key][u_new] = (f_new, (i, u_prev))

    if (L, S) not in dp:
        return None, None
    best_J = INF
    best_u = None
    for u, (f, _) in dp[(L, S)].items():
        total = f + u
        if total < best_J:
            best_J = total
            best_u = u
    if best_u is None:
        return None, None

    b = [0] * (S + 1)
    b[S] = L
    cur_j, cur_s, cur_u = L, S, best_u
    while cur_s > 0:
        _, (prev_i, prev_u) = dp[(cur_j, cur_s)][cur_u]
        b[cur_s - 1] = prev_i
        cur_j, cur_s, cur_u = prev_i, cur_s - 1, prev_u
    return best_J, b


def solve_runtime_dp_greedy(cluster: Cluster, ms: ModelSpec,
                             S: int, a: List[int], b_prev: List[int], a_prev: List[int],
                             L: int, P: int, t_r: int, G_rem: int, W: int, K_r: int,
                             phi_hat_horizon: np.ndarray,
                             phi_infinity: np.ndarray,
                             q_mem_horizon: np.ndarray,
                             link_obs: Dict[Tuple[int, int], Tuple[float, float]]
                             ) -> Tuple[Optional[float], Optional[List[int]]]:
    """Ablation: greedy additive-only DP. Picks b minimizing sum V_r, checks Omega_b
    only at the end (no bottleneck u-state). Tests whether handoff-bottleneck
    awareness during the search matters (Sec. 5.3 w/o Bottleneck-DP).
    """
    INF = float("inf")
    D_s_per_tok = [0.0] * (S + 1)
    for s in range(1, S + 1):
        if s < S:
            alpha, beta = link_obs[(a[s - 1], a[s])]
            D_s_per_tok[s] = alpha + beta * ms.z_dec_bytes()

    tail = max(0, G_rem - K_r * W)
    horizon_phi_sum = np.zeros(cluster.N)
    for n in range(cluster.N):
        horizon_phi_sum[n] = W * float(np.sum(phi_hat_horizon[n, :K_r])) + tail * phi_infinity[n]
    t_mid = t_r + min(G_rem, K_r * W) // 2
    q_mem_max = np.max(q_mem_horizon[:, :K_r], axis=1)

    # Additive DP without bottleneck state
    dp: Dict[Tuple[int, int], Tuple[float, int]] = {(0, 0): (0.0, -1)}

    for s in range(1, S + 1):
        a_s = a[s - 1]
        node = cluster.nodes[a_s]
        m_eff_min = max(0.0, node.m_bytes - q_mem_max[a_s])
        for j in range(s, L - (S - s) + 1):
            for i in range(s - 1, j):
                n_blocks = j - i
                if stage_mem_bytes(ms, n_blocks, P, t_r + K_r * W) > m_eff_min:
                    continue
                if (i, s - 1) not in dp:
                    continue
                kappa_dec_mid = ms.kappa_dec_stage(n_blocks, P, t_mid)
                xi = kappa_dec_mid / node.mu_flops
                V = xi * horizon_phi_sum[a_s] + G_rem * D_s_per_tok[s]
                f_prev, _ = dp[(i, s - 1)]
                f_new = f_prev + V
                key = (j, s)
                if key not in dp or f_new < dp[key][0]:
                    dp[key] = (f_new, i)

    if (L, S) not in dp:
        return None, None

    # Back-track
    b = [0] * (S + 1)
    b[S] = L
    cur_j, cur_s = L, S
    while cur_s > 0:
        _, prev_i = dp[(cur_j, cur_s)]
        b[cur_s - 1] = prev_i
        cur_j, cur_s = prev_i, cur_s - 1

    # Compute Omega_b post-hoc
    omega = 0.0
    for s in range(1, S + 1):
        h = H_stage(cluster, ms, s, b[s - 1], b[s], a, b_prev, a_prev, P, t_r, link_obs)
        omega = max(omega, h)

    best_J = dp[(L, S)][0] + omega
    return best_J, b


def eval_surrogate(cluster: Cluster, ms: ModelSpec,
                   S: int, a: List[int], b: List[int], b_prev: List[int], a_prev: List[int],
                   L: int, P: int, t_r: int, G_rem: int, W: int, K_r: int,
                   phi_hat_horizon: np.ndarray,
                   phi_infinity: np.ndarray,
                   link_obs: Dict[Tuple[int, int], Tuple[float, float]]) -> float:
    tail = max(0, G_rem - K_r * W)
    D_s_per_tok = [0.0] * (S + 1)
    for s in range(1, S + 1):
        if s < S:
            alpha, beta = link_obs[(a[s - 1], a[s])]
            D_s_per_tok[s] = alpha + beta * ms.z_dec_bytes()
    t_mid = t_r + min(G_rem, K_r * W) // 2
    f_add = 0.0
    h_max = 0.0
    for s in range(1, S + 1):
        a_s = a[s - 1]
        node = cluster.nodes[a_s]
        n_blocks = b[s] - b[s - 1]
        kappa_dec_mid = ms.kappa_dec_stage(n_blocks, P, t_mid)
        xi = kappa_dec_mid / node.mu_flops
        h_sum = W * float(np.sum(phi_hat_horizon[a_s, :K_r])) + tail * phi_infinity[a_s]
        V = xi * h_sum + G_rem * D_s_per_tok[s]
        f_add += V
        h_val = H_stage(cluster, ms, s, b[s - 1], b[s], a, b_prev, a_prev, P, t_r, link_obs)
        h_max = max(h_max, h_val)
    return f_add + h_max
"""Ground-truth drift simulator (updated per Sec.5.2.3 rewrite).

Workload GT: M/G/inf superposition. Each task has:
  w_cmp ~ Uniform(0.15, 0.35) * mu_n  (FLOPs/sec)
  w_mem ~ Uniform(0.5, 2.0) GB        (independent of w_cmp)
Aggregated:
  q_cmp = sum(w_cmp over active), capped at mu_n
  q_mem = sum(w_mem over active), capped at m_n - omega(one block)

Thermal GT: first-order RC with asymmetric tau_up / tau_down, physics ticks at
  fixed T_W_sec (1Hz typical), DECOUPLED from control window W_sec. Thermal drive:
    u_n^r = min(1, (omega(L_n) + chi(L_n, P, t) + q_mem) / m_n)   -- Eq.(12)
  where L_n is the current block set resident on node n (union over stages a_s=n).

Network GT: 2-state Markov + AR(1) within state. Unchanged from v1.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from src.cluster import Cluster, Node


@dataclass
class DriftState:
    theta: np.ndarray  # (N,) in °C
    # List per node of active tasks: (start_t, end_t, w_cmp_flops_per_sec, w_mem_bytes)
    active_tasks: List[List[Tuple[float, float, float, float]]]
    link_state: Dict[Tuple[int, int], str]
    link_inflation: Dict[Tuple[int, int], Tuple[float, float]]
    rng: np.random.Generator
    t_sec: float = 0.0


def _compute_q(state: DriftState, cluster: Cluster, one_block_bytes: float,
               t_now: float) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate active tasks + tier-level baseline into (q_cmp, q_mem) per node with caps."""
    N = cluster.N
    q_cmp = np.zeros(N)
    q_mem = np.zeros(N)
    for n in range(N):
        node = cluster.nodes[n]
        cmp_sum = node.q_cmp_base_flops    # always-on baseline load
        mem_sum = node.q_mem_base_bytes
        for (s_t, e_t, w_c, w_m) in state.active_tasks[n]:
            if s_t <= t_now <= e_t:
                cmp_sum += w_c
                mem_sum += w_m
        q_cmp[n] = min(node.mu_flops, cmp_sum)
        q_mem[n] = min(node.m_bytes - one_block_bytes, mem_sum)
        q_mem[n] = max(0.0, q_mem[n])
    return q_cmp, q_mem


def _init_network_state(cluster: Cluster, rng: np.random.Generator,
                        regime: dict, drift_cfg: dict) -> Tuple[Dict, Dict]:
    N = cluster.N
    state: Dict[Tuple[int, int], str] = {}
    infl: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            state[(i, j)] = "normal"
            infl[(i, j)] = (1.0, 1.0)
    return state, infl


def init_drift(cluster: Cluster, cfg: dict, seed: int,
               one_block_bytes: float = 0.0) -> DriftState:
    """Init drift state at t=0. Warm-starts workload by pre-populating active tasks
    sampled from stationary distribution; initial temperature = theta_amb + nu*u + noise.
    """
    rng = np.random.default_rng(seed)
    drift_cfg = cfg["drift"]
    regime_name = drift_cfg["regime"]["active"]
    regime = drift_cfg["regime"][regime_name]
    wl_cfg = drift_cfg["workload"]
    N = cluster.N

    active_tasks: List[List[Tuple[float, float, float, float]]] = [[] for _ in range(N)]

    if regime.get("workload_active", True):
        for n in range(N):
            node = cluster.nodes[n]
            tier_wl = wl_cfg["per_tier"][node.tier]
            lam = rng.uniform(*tier_wl["lambda_job_per_sec_range"])
            mean_dur = np.exp(wl_cfg["duration_lognormal_mu_log"]
                              + 0.5 * wl_cfg["duration_lognormal_sigma"] ** 2)
            n_active = max(1, rng.poisson(lam * mean_dur))
            for _ in range(n_active):
                dur = rng.lognormal(wl_cfg["duration_lognormal_mu_log"],
                                    wl_cfg["duration_lognormal_sigma"])
                start_t = -rng.uniform(0, dur)
                end_t = start_t + dur
                if end_t <= 0:
                    continue
                cmp_frac = rng.uniform(*tier_wl["cmp_intensity_frac_of_mu_range"])
                w_cmp = cmp_frac * node.mu_flops
                w_mem = rng.uniform(*tier_wl["mem_footprint_gb_range"]) * (1024 ** 3)
                active_tasks[n].append((start_t, end_t, w_cmp, w_mem))

    ds = DriftState(
        theta=np.zeros(N),
        active_tasks=active_tasks,
        link_state={}, link_inflation={},
        rng=rng, t_sec=0.0,
    )

    # Initial temperature based on u_mem at t=0
    q_cmp_0, q_mem_0 = _compute_q(ds, cluster, one_block_bytes, 0.0)
    for n_idx, node in enumerate(cluster.nodes):
        if regime.get("thermal_active", True):
            # At t=0 no foreground request yet, so u_mem = q_mem / m
            u0 = min(1.0, q_mem_0[n_idx] / node.m_bytes)
            theta_ss = cluster.theta_amb + node.nu * u0
            ds.theta[n_idx] = theta_ss + rng.normal(0, drift_cfg["thermal"]["initial_noise_std_c"])
        else:
            ds.theta[n_idx] = cluster.theta_amb

    ds.link_state, ds.link_inflation = _init_network_state(cluster, rng, regime, drift_cfg)

    if regime_name == "R4_network" and regime.get("beta_doubling", False):
        n_deg = regime.get("degraded_links", 2)
        all_links = [(i, j) for i in range(N) for j in range(N) if i != j]
        chosen = rng.choice(len(all_links), size=min(n_deg, len(all_links)), replace=False)
        for k in chosen:
            ds.link_state[all_links[k]] = "degraded"
            ds.link_inflation[all_links[k]] = (2.0, 2.0)

    return ds


def advance_workload(state: DriftState, cluster: Cluster, wl_cfg: dict,
                     t_start: float, t_end: float,
                     workload_active: bool, one_block_bytes: float) -> Tuple[np.ndarray, np.ndarray]:
    """Advance M/G/inf over [t_start, t_end]; return (q_cmp, q_mem) averaged (midpoint).

    Tier-aware: cmp intensity range and mem footprint range are per-tier
    (higher tier -> larger jobs, reflecting kubelet-style scheduling).
    Arrival rate lambda is global (sampled once per node per interval).
    """
    N = cluster.N
    if not workload_active:
        return np.zeros(N), np.zeros(N)

    dt = t_end - t_start
    if dt > 0:
        for n in range(N):
            node = cluster.nodes[n]
            tier_wl = wl_cfg["per_tier"][node.tier]
            lam_n = state.rng.uniform(*tier_wl["lambda_job_per_sec_range"])
            n_new = state.rng.poisson(lam_n * dt)
            for _ in range(n_new):
                arrive = t_start + state.rng.uniform(0, dt)
                dur = state.rng.lognormal(wl_cfg["duration_lognormal_mu_log"],
                                          wl_cfg["duration_lognormal_sigma"])
                cmp_frac = state.rng.uniform(*tier_wl["cmp_intensity_frac_of_mu_range"])
                w_cmp = cmp_frac * node.mu_flops
                w_mem = state.rng.uniform(*tier_wl["mem_footprint_gb_range"]) * (1024 ** 3)
                state.active_tasks[n].append((arrive, arrive + dur, w_cmp, w_mem))
            state.active_tasks[n] = [t for t in state.active_tasks[n] if t[1] > t_start]

    t_mid = 0.5 * (t_start + t_end)
    return _compute_q(state, cluster, one_block_bytes, t_mid)

    t_mid = 0.5 * (t_start + t_end)
    return _compute_q(state, cluster, one_block_bytes, t_mid)


def advance_thermal_fixed_dt(state: DriftState, cluster: Cluster,
                              u_total: np.ndarray,
                              t_start: float, t_end: float,
                              thermal_active: bool, process_noise_var: float,
                              T_W_sec: float = 1.0,
                              snapshots: list = None) -> None:
    """Advance thermal GT with fixed T_W step, independent of W_sec.

    u_total: (N,) total thermal drive for the whole interval (held constant).
    snapshots: if provided, append {"t": t, "theta": [...], "u": [...]} at each tick.
    """
    if not thermal_active or t_end <= t_start:
        if snapshots is not None:
            snapshots.append({"t": float(t_start), "theta": state.theta.tolist(),
                              "u": u_total.tolist()})
        return
    total_dt = t_end - t_start
    n_full = int(total_dt // T_W_sec)
    rem = total_dt - n_full * T_W_sec

    # Per-tick theta arrays (stored as N-vectors across nodes, updated sync per tick)
    # We step all nodes in lockstep at each tick to allow mid-interval snapshots.
    theta = state.theta.copy()
    t_cur = t_start
    for tick in range(n_full):
        for n_idx, node in enumerate(cluster.nodes):
            theta_ss = cluster.theta_amb + node.nu * min(1.0, u_total[n_idx])
            tau = node.tau_up_s if theta_ss > theta[n_idx] else node.tau_down_s
            alpha = 1.0 - np.exp(-T_W_sec / tau)
            theta[n_idx] = theta[n_idx] + alpha * (theta_ss - theta[n_idx])
            theta[n_idx] += state.rng.normal(0, np.sqrt(process_noise_var * T_W_sec))
            theta[n_idx] = max(cluster.theta_amb, theta[n_idx])
        t_cur += T_W_sec
        if snapshots is not None:
            snapshots.append({"t": float(t_cur), "theta": theta.tolist(),
                              "u": u_total.tolist()})
    if rem > 1e-6:
        for n_idx, node in enumerate(cluster.nodes):
            theta_ss = cluster.theta_amb + node.nu * min(1.0, u_total[n_idx])
            tau = node.tau_up_s if theta_ss > theta[n_idx] else node.tau_down_s
            alpha = 1.0 - np.exp(-rem / tau)
            theta[n_idx] = theta[n_idx] + alpha * (theta_ss - theta[n_idx])
            theta[n_idx] += state.rng.normal(0, np.sqrt(process_noise_var * rem))
            theta[n_idx] = max(cluster.theta_amb, theta[n_idx])
        t_cur += rem
        if snapshots is not None:
            snapshots.append({"t": float(t_cur), "theta": theta.tolist(),
                              "u": u_total.tolist()})
    state.theta = theta


def advance_network(state: DriftState, net_cfg: dict, network_active: bool) -> None:
    if not network_active:
        return
    p_norm_self = net_cfg["p_normal_self"]
    p_deg_self = net_cfg["p_degraded_self"]
    for key, st in list(state.link_state.items()):
        if st == "normal":
            if state.rng.random() > p_norm_self:
                state.link_state[key] = "degraded"
                state.link_inflation[key] = (
                    state.rng.uniform(*net_cfg["degraded_inflation_range"]),
                    state.rng.uniform(*net_cfg["degraded_inflation_range"]),
                )
        else:
            if state.rng.random() > p_deg_self:
                state.link_state[key] = "normal"
                state.link_inflation[key] = (1.0, 1.0)
        if state.link_state[key] == "normal":
            da = state.rng.normal(0, net_cfg["ar1_std_frac"])
            db = state.rng.normal(0, net_cfg["ar1_std_frac"])
            a_m, b_m = state.link_inflation[key]
            a_new = max(0.1, net_cfg["ar1_persistence"] * (a_m - 1.0) + 1.0 + da)
            b_new = max(0.1, net_cfg["ar1_persistence"] * (b_m - 1.0) + 1.0 + db)
            state.link_inflation[key] = (a_new, b_new)


def observed_sensor(state: DriftState, cluster: Cluster, cfg: dict,
                    rng: np.random.Generator, one_block_bytes: float,
                    q_cmp_true: np.ndarray, q_mem_true: np.ndarray) -> Dict:
    """Noisy observations available to the controller at window start.

    Thermal: sensor_noise_var (°C²). Workload: 3% relative std on each channel.
    """
    sv = cfg["devices"]["thermal"]["sensor_noise_var"]
    theta_obs = state.theta + rng.normal(0, np.sqrt(sv), size=state.theta.shape)

    rel_std = cfg["drift"]["workload"]["obs_noise_rel_std"]
    q_cmp_obs = q_cmp_true * (1.0 + rng.normal(0, rel_std, size=q_cmp_true.shape))
    q_mem_obs = q_mem_true * (1.0 + rng.normal(0, rel_std, size=q_mem_true.shape))
    q_cmp_obs = np.clip(q_cmp_obs, 0.0, None)
    q_mem_obs = np.clip(q_mem_obs, 0.0, None)

    alpha0 = cfg["drift"]["network"]["alpha_normal_s"]
    beta0 = cfg["drift"]["network"]["beta_normal_s_per_byte"]
    link_obs = {}
    for key, (a_m, b_m) in state.link_inflation.items():
        link_obs[key] = (alpha0 * a_m, beta0 * b_m)
    return {"theta_obs": theta_obs,
            "q_cmp_obs": q_cmp_obs, "q_mem_obs": q_mem_obs,
            "link_obs": link_obs}


def compute_true_phi(cluster: Cluster, q_cmp: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """phi_n = phi_th * phi_wk from GT state. phi_wk = 1 + rho * q_cmp / mu_n."""
    N = cluster.N
    phi = np.ones(N)
    for n_idx, node in enumerate(cluster.nodes):
        phi_th = 1.0 + node.gamma * max(0.0, theta[n_idx] - node.theta_th)
        phi_wk = 1.0 + node.rho * (q_cmp[n_idx] / node.mu_flops)
        phi[n_idx] = phi_th * phi_wk
    return phi


def apply_regime_overrides(cluster: Cluster, cfg: dict) -> None:
    """Mutate cluster nodes per regime-specific overrides. Idempotent: resets base
    values from cfg first so repeated calls don't accumulate multipliers.
    """
    tiers = cfg["devices"]["tiers"]
    # Reset nu, theta_th, and rho to JSON baseline (guards against cross-run pollution)
    for node in cluster.nodes:
        t = tiers[node.tier]
        node.nu = t["nu"]
        node.theta_th = t["theta_th"]
        node.rho = t["rho"]

    regime_name = cfg["drift"]["regime"]["active"]
    regime = cfg["drift"]["regime"][regime_name]
    
    nu_mult = regime.get("nu_multiplier", 1.0)
    theta_th_ovr = regime.get("theta_th_override", None)
    for node in cluster.nodes:
        node.nu = node.nu * nu_mult
        if theta_th_ovr is not None:
            node.theta_th = theta_th_ovr

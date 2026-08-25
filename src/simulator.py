"""Single-trace execution: run a scheme end-to-end over one GT drift realization."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from src.cluster import Cluster
from src.model_spec import ModelSpec
from src.drift_gt import (
    DriftState, init_drift, advance_workload, advance_thermal_fixed_dt,
    advance_network, observed_sensor, compute_true_phi, apply_regime_overrides,
    _compute_q,
)
from src.cost_model import (
    T_startup, T_prefill, T_decode_window, T_decode_window_detail,
    Omega_reconfig, compute_u_thermal, stage_mem_bytes
)
from src.schemes.schemes import SchemeState, build_scheme, OracleScheme


@dataclass
class TraceResult:
    scheme: str
    seed: int
    TTLT_s: float
    TTFT_s: float
    TPOT_series_s: List[float] = field(default_factory=list)
    overhead_s: float = 0.0
    controller_s: float = 0.0
    n_reconfigs: int = 0
    windows: List[dict] = field(default_factory=list)
    device_seconds: List[dict] = field(default_factory=list)  # per-second snapshots
    tokens: List[dict] = field(default_factory=list)          # per-token breakdown


def _precompute_oracle(cl: Cluster, cfg: dict, seed: int,
                       n_windows_max: int, H_max: int,
                       P: int, G_hat: int, W: int, one_block_bytes: float) -> Dict:
    """Dry-run GT to record future phi and q_mem per window (no self-heating approx)."""
    ds = init_drift(cl, cfg, seed=seed, one_block_bytes=one_block_bytes)
    regime_name = cfg["drift"]["regime"]["active"]
    regime = cfg["drift"]["regime"][regime_name]
    W_sec_nom = W * 0.05
    phi_seq = []
    q_mem_seq = []
    t = 0.0
    for r in range(n_windows_max):
        q_cmp, q_mem = advance_workload(ds, cl, cfg["drift"]["workload"], t, t + W_sec_nom,
                                         regime.get("workload_active", True), one_block_bytes)
        u = np.zeros(cl.N)
        advance_thermal_fixed_dt(ds, cl, u, t, t + W_sec_nom,
                                 regime.get("thermal_active", True),
                                 cfg["devices"]["thermal"]["process_noise_var"],
                                 cl.T_W_sec)
        advance_network(ds, cfg["drift"]["network"], regime.get("network_active", True))
        phi = compute_true_phi(cl, q_cmp, ds.theta)
        phi_seq.append(phi)
        q_mem_seq.append(q_mem)
        t += W_sec_nom
    return {"phi_seq": np.array(phi_seq).T, "q_mem_seq": np.array(q_mem_seq).T}



def _controller_ms(cfg: dict, scheme_name: str, N: int, S: int) -> float:
    """Per-window controller cost in ms, or 0.0 when not charged.

    The simulator prices migration and not deciding. That asymmetry favours
    whichever scheme searches hardest, because its search is free: FM
    enumerates ordered placements every window while DACI runs one boundary DP.
    Measured on a Jetson core, FM costs 81 ms/window at N=8 against DACI's 7.5.

    Off by default. A missing table entry charges zero and warns once rather
    than guessing, because a silently-zero cost is what this exists to fix.
    """
    cc = cfg.get("experiment", {}).get("controller_cost", {})
    if not cc.get("enabled", False):
        return 0.0
    key = f"N{N}_S{S}"
    entry = cc.get("table_ms", {}).get(key)
    if entry is None or scheme_name not in entry:
        if key not in _CTRL_WARNED:
            _CTRL_WARNED.add(key)
            print(f"  [controller_cost] no entry for {key}/{scheme_name}; "
                  f"charging 0. Measure it with bench_controller.py.")
        return 0.0
    return float(entry[scheme_name])


_CTRL_WARNED = set()

def run_trace(cl: Cluster, ms: ModelSpec, cfg: dict, scheme_name: str,
              seed: int, verbose: bool = False) -> TraceResult:
    # Apply regime overrides (e.g., R2 nu_multiplier)
    apply_regime_overrides(cl, cfg)

    exp = cfg["experiment"]
    algo = cfg["algo"]
    drift_cfg = cfg["drift"]
    regime_name = drift_cfg["regime"]["active"]
    regime = drift_cfg["regime"][regime_name]

    P = exp["request"]["P_prompt_tokens"]
    G_hat = exp["request"]["G_hat_tokens"]
    W = algo["window"]["W_tokens"]
    H_max = algo["window"]["H_max"]
    R_total = (G_hat + W - 1) // W

    one_block = ms.omega_block_bytes

    ds = init_drift(cl, cfg, seed=seed, one_block_bytes=one_block)
    obs_rng = np.random.default_rng(seed + 500_000)

    oracle_data = None
    if scheme_name == "OR":
        oracle_data = _precompute_oracle(cl, cfg, seed, R_total + H_max, H_max,
                                          P, G_hat, W, one_block)

    scheme = build_scheme(scheme_name, cfg)

    # Initial observation: use t_mid=0 q
    q_cmp_0, q_mem_0 = _compute_q(ds, cl, one_block, 0.0)
    obs0 = observed_sensor(ds, cl, cfg, obs_rng, one_block, q_cmp_0, q_mem_0)
    st = scheme.decide_initial(cl, ms, obs0, P, G_hat)

    T_start = T_startup(cl, ms, st.a, st.b)
    phi_now = compute_true_phi(cl, q_cmp_0, ds.theta)
    T_pf = T_prefill(cl, ms, st.b, st.a, phi_now, obs0["link_obs"], P)
    TTFT = T_start + T_pf
    t_wall = T_start + T_pf

    TTLT = TTFT
    overhead_total = 0.0
    controller_total = 0.0
    n_reconf = 0
    tpot_series: List[float] = []
    windows_log: List[dict] = []
    device_seconds: List[dict] = []
    tokens_log: List[dict] = []

    # Per-second snapshot of initial state (prefill doesn't run thermal dynamics here)
    device_seconds.append({
        "t": 0.0, "phase": "init",
        "theta": ds.theta.tolist(),
        "q_cmp_frac": [q_cmp_0[i] / cl.nodes[i].mu_flops for i in range(cl.N)],
        "q_mem_gb": [q_mem_0[i] / (1024 ** 3) for i in range(cl.N)],
        "u_thermal": compute_u_thermal(cl, ms, st.a, st.b, q_mem_0, P, 0).tolist(),
    })

    b_prev = list(st.b)
    a_prev = list(st.a)
    token_idx = 0

    for r in range(1, R_total + 1):
        q_cmp_win, q_mem_win = advance_workload(
            ds, cl, drift_cfg["workload"], t_wall, t_wall + st.W_sec,
            regime.get("workload_active", True), one_block
        )
        u_drive = compute_u_thermal(cl, ms, st.a, st.b, q_mem_win, P, (r - 1) * W)

        # Per-second thermal snapshots
        thermal_snaps: list = []
        advance_thermal_fixed_dt(
            ds, cl, u_drive, t_wall, t_wall + st.W_sec,
            regime.get("thermal_active", True),
            cfg["devices"]["thermal"]["process_noise_var"],
            cl.T_W_sec, snapshots=thermal_snaps
        )
        advance_network(ds, drift_cfg["network"], regime.get("network_active", True))

        # Record per-second log for this window's evolution
        for snap in thermal_snaps:
            device_seconds.append({
                "t": snap["t"], "phase": "decode", "r": r,
                "theta": snap["theta"],
                "q_cmp_frac": [q_cmp_win[i] / cl.nodes[i].mu_flops for i in range(cl.N)],
                "q_mem_gb": [q_mem_win[i] / (1024 ** 3) for i in range(cl.N)],
                "u_thermal": snap["u"],
            })

        obs_r = observed_sensor(ds, cl, cfg, obs_rng, one_block, q_cmp_win, q_mem_win)

        if scheme_name == "OR" and oracle_data is not None:
            ph = oracle_data["phi_seq"][:, r-1 : r-1 + H_max]
            qm = oracle_data["q_mem_seq"][:, r-1 : r-1 + H_max]
            if ph.shape[1] < H_max:
                pad = H_max - ph.shape[1]
                ph = np.concatenate([ph, np.tile(ph[:, -1:], (1, pad))], axis=1)
                qm = np.concatenate([qm, np.tile(qm[:, -1:], (1, pad))], axis=1)
            scheme.set_ground_truth(ph, qm)

        t_r = (r - 1) * W
        G_rem = G_hat - t_r
        b_new, accepted, meta = scheme.decide_runtime(cl, ms, obs_r, st, P, t_r, G_rem, W)

        omega = 0.0
        if accepted:
            a_for_omega = meta.get("a_prev_for_omega", a_prev)
            omega = Omega_reconfig(cl, ms, b_new, st.a, b_prev, a_for_omega, P, t_r, obs_r["link_obs"])
            overhead_total += omega
            n_reconf += 1
            TTLT += omega
            t_wall += omega

        # The controller's own optimisation time, charged like any other
        # wall-clock cost. It is paid every window whether or not the decision
        # is accepted -- deciding not to move still costs the search.
        ctrl_s = _controller_ms(cfg, scheme_name, cl.N, st.S) / 1000.0
        if ctrl_s:
            TTLT += ctrl_s
            t_wall += ctrl_s
            controller_total += ctrl_s

        st.b = b_new
        b_prev = list(st.b)
        a_prev = list(st.a)

        true_phi_r = compute_true_phi(cl, q_cmp_win, ds.theta)
        tpot_r = T_decode_window(cl, ms, st.b, st.a, true_phi_r, obs_r["link_obs"], P, t_r)
        tokens_this_win = min(W, G_rem)

        # Per-token log (representative of window; realized tpot is constant in a window here)
        for tok_i in range(tokens_this_win):
            t_abs = t_r + tok_i
            detail = T_decode_window_detail(cl, ms, st.b, st.a, true_phi_r, obs_r["link_obs"], P, t_abs)
            tokens_log.append({
                "token": token_idx, "t_abs": t_abs, "r": r,
                "tpot_ms": detail["tpot_ms"],
                "wall_t_s": t_wall + tok_i * tpot_r,
                "stages": detail["stages"],
            })
            token_idx += 1

        win_time = tokens_this_win * tpot_r
        TTLT += win_time
        t_wall += win_time
        tpot_series.extend([tpot_r] * tokens_this_win)

        windows_log.append({
            "r": r, "t_r": t_r, "b": list(st.b), "a": list(st.a),
            "accepted": bool(accepted), "omega": float(omega), "tpot": float(tpot_r),
            "theta_gt": ds.theta.tolist(),
            "q_cmp_gt_frac": [q_cmp_win[i] / cl.nodes[i].mu_flops for i in range(cl.N)],
            "q_mem_gt_gb": [q_mem_win[i] / (1024 ** 3) for i in range(cl.N)],
            "phi_true": true_phi_r.tolist(),
            "phi_hat_curr": meta.get("phi_hat_curr", []),
            "phi_hat_horizon": meta.get("phi_hat_horizon", []),
            "u_thermal": meta.get("u_thermal", []),
            "H_r_star": meta.get("H_r_star", 0),
            "J_new": meta.get("J_new", None),
            "J_incumbent": meta.get("J_incumbent", None),
        })

        if verbose and r % 20 == 0:
            print(f"  r={r} t_r={t_r} b={st.b} a={st.a} acc={accepted} "
                  f"tpot={tpot_r*1000:.1f}ms theta={[f'{x:.0f}' for x in ds.theta]} "
                  f"u={[f'{x:.2f}' for x in u_drive]}")

    return TraceResult(
        scheme=scheme_name, seed=seed,
        TTLT_s=TTLT, TTFT_s=TTFT, TPOT_series_s=tpot_series,
        overhead_s=overhead_total, n_reconfigs=n_reconf,
        controller_s=controller_total,
        windows=windows_log,
        device_seconds=device_seconds,
        tokens=tokens_log,
    )
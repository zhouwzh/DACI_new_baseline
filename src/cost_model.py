"""Cost model: Eqs. (3)-(10) updated.

Key change: memory feasibility uses m_n - q_mem (bytes), not m_n - delta*q.
phi_wk = 1 + rho * q_cmp / mu_n (dimensionless input).
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.cluster import Cluster
from src.model_spec import ModelSpec


def C_stage(cluster: Cluster, ms: ModelSpec, a_s: int, n_blocks: int,
            phi: float, P: int, t: int, phase: str) -> float:
    """Compute latency Eq.(3): C = kappa / mu * phi."""
    node = cluster.nodes[a_s]
    if phase == "pf":
        kappa = ms.kappa_pf_stage(n_blocks, P)
    elif phase == "dec":
        kappa = ms.kappa_dec_stage(n_blocks, P, t)
    else:
        raise ValueError(phase)
    return (kappa / node.mu_flops) * phi


def D_stage(ms: ModelSpec, a_s: int, a_next: Optional[int],
            link_state: Dict[Tuple[int, int], Tuple[float, float]],
            P: int, phase: str) -> float:
    """Hidden-state forwarding Eq.(6). Last stage emits no hidden state (LM head local)."""
    if a_next is None:
        return 0.0
    alpha, beta = link_state[(a_s, a_next)]
    z = ms.z_pf_bytes_with_P(P) if phase == "pf" else ms.z_dec_bytes()
    return alpha + beta * z


def T_startup(cluster: Cluster, ms: ModelSpec, a: List[int], b: List[int]) -> float:
    vals = []
    for s in range(1, len(b)):
        n_blocks = b[s] - b[s-1]
        node = cluster.nodes[a[s-1]]
        vals.append(ms.omega_stage_bytes(n_blocks) / node.B_load_bps)
    return max(vals) if vals else 0.0


def H_stage(cluster: Cluster, ms: ModelSpec,
            s_idx: int, i: int, j: int, a: List[int],
            b_prev: List[int], a_prev: List[int],
            P: int, t_r: int,
            link_state: Dict[Tuple[int, int], Tuple[float, float]]) -> float:
    """Per-stage handoff Eq.(7). s_idx is 1-indexed."""
    node_new = cluster.nodes[a[s_idx - 1]]
    prior_on_same_node = set()
    for s_prev in range(1, len(b_prev)):
        if a_prev[s_prev - 1] == a[s_idx - 1]:
            for blk in range(b_prev[s_prev - 1] + 1, b_prev[s_prev] + 1):
                prior_on_same_node.add(blk)
            break
    current = set(range(i + 1, j + 1))
    delta_plus = current - prior_on_same_node
    if not delta_plus:
        return 0.0
    loading_time = len(delta_plus) * ms.omega_block_bytes / node_new.B_load_bps
    kv_by_donor: Dict[int, int] = {}
    for blk in delta_plus:
        donor_stage_prev = None
        for s_prev in range(1, len(b_prev)):
            if b_prev[s_prev - 1] < blk <= b_prev[s_prev]:
                donor_stage_prev = s_prev
                break
        if donor_stage_prev is None:
            continue
        donor_node = a_prev[donor_stage_prev - 1]
        if donor_node == a[s_idx - 1]:
            continue
        kv_by_donor[donor_node] = kv_by_donor.get(donor_node, 0) + 1
    kv_time = 0.0
    for donor_node, n_blk in kv_by_donor.items():
        kv_bytes = n_blk * (P + t_r) * ms.chi_unit_block_bytes_per_token
        alpha, beta = link_state[(donor_node, a[s_idx - 1])]
        kv_time += alpha + beta * kv_bytes
    swap_cost = 0.0
    if s_idx - 1 < len(a_prev) and a_prev[s_idx - 1] != a[s_idx - 1]:
        swap_cost = node_new.H_swap_s
    return loading_time + kv_time + swap_cost


def Omega_reconfig(cluster: Cluster, ms: ModelSpec,
                   b: List[int], a: List[int],
                   b_prev: List[int], a_prev: List[int],
                   P: int, t_r: int,
                   link_state: Dict[Tuple[int, int], Tuple[float, float]]) -> float:
    if b == b_prev and a == a_prev:
        return 0.0
    S = len(b) - 1
    vals = [H_stage(cluster, ms, s, b[s-1], b[s], a, b_prev, a_prev, P, t_r, link_state)
            for s in range(1, S + 1)]
    return max(vals) if vals else 0.0


def stage_mem_bytes(ms: ModelSpec, n_blocks: int, P: int, t: int) -> float:
    return ms.omega_stage_bytes(n_blocks) + ms.chi_stage_bytes(n_blocks, P, t)


def memory_feasible(cluster: Cluster, ms: ModelSpec, b: List[int], a: List[int],
                    q_mem: np.ndarray, P: int, t: int) -> bool:
    """Eq.(10): omega + chi <= m_n - q_mem."""
    for s in range(1, len(b)):
        n = cluster.nodes[a[s - 1]]
        n_blocks = b[s] - b[s - 1]
        eff_mem = max(0.0, n.m_bytes - q_mem[n.idx])
        if stage_mem_bytes(ms, n_blocks, P, t) > eff_mem:
            return False
    return True


def T_decode_window(cluster: Cluster, ms: ModelSpec,
                    b: List[int], a: List[int], phi: np.ndarray,
                    link_state: Dict[Tuple[int, int], Tuple[float, float]],
                    P: int, t: int) -> float:
    S = len(b) - 1
    total = 0.0
    for s in range(1, S + 1):
        n_blocks = b[s] - b[s - 1]
        a_s = a[s - 1]
        a_next = a[s] if s < S else None
        total += C_stage(cluster, ms, a_s, n_blocks, phi[a_s], P, t, "dec")
        total += D_stage(ms, a_s, a_next, link_state, P, "dec")
    return total


def T_decode_window_detail(cluster: Cluster, ms: ModelSpec,
                           b: List[int], a: List[int], phi: np.ndarray,
                           link_state: Dict[Tuple[int, int], Tuple[float, float]],
                           P: int, t: int) -> dict:
    """Same as T_decode_window but returns per-stage breakdown."""
    S = len(b) - 1
    stages = []
    total = 0.0
    for s in range(1, S + 1):
        n_blocks = b[s] - b[s - 1]
        a_s = a[s - 1]
        a_next = a[s] if s < S else None
        c = C_stage(cluster, ms, a_s, n_blocks, phi[a_s], P, t, "dec")
        d = D_stage(ms, a_s, a_next, link_state, P, "dec")
        stages.append({
            "s": s, "a": a_s, "n_blocks": n_blocks,
            "C_s_ms": c * 1000, "D_s_ms": d * 1000,
            "phi": float(phi[a_s]),
            "kappa_dec_gflops": ms.kappa_dec_stage(n_blocks, P, t) / 1e9,
            "mem_fg_gb": stage_mem_bytes(ms, n_blocks, P, t) / (1024 ** 3),
        })
        total += c + d
    return {"tpot_ms": total * 1000, "stages": stages}


def T_prefill(cluster: Cluster, ms: ModelSpec,
              b: List[int], a: List[int], phi: np.ndarray,
              link_state: Dict[Tuple[int, int], Tuple[float, float]],
              P: int) -> float:
    S = len(b) - 1
    total = 0.0
    for s in range(1, S + 1):
        n_blocks = b[s] - b[s - 1]
        a_s = a[s - 1]
        a_next = a[s] if s < S else None
        total += C_stage(cluster, ms, a_s, n_blocks, phi[a_s], P, 0, "pf")
        total += D_stage(ms, a_s, a_next, link_state, P, "pf")
    return total


def compute_u_thermal(cluster: Cluster, ms: ModelSpec,
                      a: List[int], b: List[int],
                      q_mem: np.ndarray, P: int, t: int) -> np.ndarray:
    """Eq.(12): u_n = min(1, (omega(L_n) + chi(L_n, P, t) + q_mem) / m_n).

    L_n = union over stages s with a_s=n of block sets. For fixed placement each
    node hosts at most one stage (by uniqueness constraint), so L_n = stage s's blocks.
    """
    N = cluster.N
    u = np.zeros(N)
    # Map each node to the blocks it hosts
    for s in range(1, len(b)):
        n = a[s - 1]
        n_blocks = b[s] - b[s - 1]
        mem_fg = stage_mem_bytes(ms, n_blocks, P, t)
        node = cluster.nodes[n]
        u[n] = min(1.0, (mem_fg + q_mem[n]) / node.m_bytes)
    # Nodes not hosting any stage: u = q_mem / m (background only)
    hosting = set(a)
    for n_idx, node in enumerate(cluster.nodes):
        if n_idx not in hosting:
            u[n_idx] = min(1.0, q_mem[n_idx] / node.m_bytes)
    return u

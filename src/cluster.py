"""Cluster structure: nodes with baseline params, all-to-all links."""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Node:
    idx: int
    tier: str
    prototype: str
    mu_baseline_tflops: float
    m_bytes: float
    B_load_bps: float
    gamma: float
    theta_th: float
    nu: float
    rho: float
    tau_up_s: float
    tau_down_s: float
    q_cmp_base_flops: float = 0.0    # tier-level baseline background compute load (FLOPs/s)
    q_mem_base_bytes: float = 0.0    # tier-level baseline memory residency (bytes)
    H_swap_s: float = 0.0 

    @property
    def mu_flops(self) -> float:
        return self.mu_baseline_tflops * 1e12

    @property
    def tau_bar_s(self) -> float:
        return 2.0 * self.tau_up_s * self.tau_down_s / (self.tau_up_s + self.tau_down_s)


@dataclass
class LinkBaseline:
    alpha_s: float
    beta_s_per_byte: float


@dataclass
class Cluster:
    nodes: List[Node]
    baseline_link: LinkBaseline
    theta_amb: float
    T_W_sec: float              # fixed thermal discretization step

    @property
    def N(self) -> int:
        return len(self.nodes)


def build_cluster(cfg: dict, mix: Dict[str, int], seed: int = 0) -> Cluster:
    tiers = cfg["devices"]["tiers"]
    therm = cfg["devices"]["thermal"]

    nodes: List[Node] = []
    idx = 0
    for tier_name in ["high", "mid", "low"]:
        t = tiers[tier_name]
        count = mix.get(tier_name, t["count"])
        for _ in range(count):
            nodes.append(Node(
                idx=idx,
                tier=tier_name,
                prototype=t["prototype"],
                mu_baseline_tflops=t["mu_baseline_tflops"],
                m_bytes=t["m_gb"] * (1024 ** 3),
                B_load_bps=t["B_load_gbps"] * (1024 ** 3),
                gamma=t["gamma"],
                theta_th=t["theta_th"],
                nu=t["nu"],
                rho=t["rho"],
                tau_up_s=t["tau_up_s"],
                tau_down_s=t["tau_down_s"],
                q_cmp_base_flops=t.get("q_cmp_base_tflops", 0.0) * 1e12,
                q_mem_base_bytes=t.get("q_mem_base_gb", 0.0) * (1024 ** 3),
                # BUGFIX: this was omitted, so every node silently took the
                # dataclass default of 0.0 while devices.json specified
                # 2.5/3/4 s per tier. cost_model.H_stage reads
                # node_new.H_swap_s, so NO scheme was ever charged for a
                # placement swap -- which understates exactly the baselines
                # (RT, FM) that swap placement, and only those.
                H_swap_s=t["H_swap_s"],
            ))
            idx += 1
    # ONE SOURCE OF TRUTH. This field used to be built from devices.json's
    # own `link` block, and NOTHING EVER READ IT -- every DP and cost path
    # takes its alpha/beta from obs["link_obs"], which drift_gt builds from
    # drift.network. So an M5a correction applied to the devices.json block
    # changed no result at all, while looking authoritative enough that the
    # error survived a full 30-trace regeneration and a hardware comparison.
    #
    # It is now read from the same place drift_gt reads, so the two cannot
    # disagree. If you are looking for the constants that actually shape a
    # result, they are in configs/drift.json under `network`.
    net = cfg.get("drift", {}).get("network", {})
    bl = LinkBaseline(alpha_s=net.get("alpha_normal_s", 487.1e-6),
                      beta_s_per_byte=net.get("beta_normal_s_per_byte", 9.126e-9))
    return Cluster(nodes=nodes, baseline_link=bl,
                   theta_amb=therm["theta_amb_c"],
                   T_W_sec=therm.get("T_W_sec", 1.0))


if __name__ == "__main__":
    import json
    cfg = {"devices": json.load(open("configs/devices.json"))}
    cl = build_cluster(cfg, {"high": 2, "mid": 3, "low": 3})
    print(f"N={cl.N}, theta_amb={cl.theta_amb}, T_W={cl.T_W_sec}s")
    for n in cl.nodes:
        print(f"  n{n.idx} [{n.tier}] mu={n.mu_baseline_tflops:.1f} TF, "
              f"m={n.m_bytes/1e9:.1f} GB, nu={n.nu}, tau_bar={n.tau_bar_s:.1f}s")

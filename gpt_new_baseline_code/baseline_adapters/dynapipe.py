"""DynaPipe semantic adapter for DACI's single-request simulator.

DynaPipe's original implementation targets batched pipeline serving.  It
measures sampling work at the final pipeline stage, moves a small number of
layers from that stage to upstream stages, and commits only after a stable
window.  DACI models one long request, so the original source's guard for
fewer than five active decode requests is preserved by default.  In that
default setting this class intentionally behaves like a static initial
partition after startup, rather than inventing a DynaPipe gain that the source
workload does not support.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.cost_model import C_stage, D_stage, compute_u_thermal, memory_feasible
from src.schemes.schemes import DACIScheme, SchemeState


DEFAULTS = {
    # DynaPipe paper Sec. 4.1 uses a window threshold of 25.
    "stability_windows": 25,
    # The released worker.py resets its adjustment counter when decode < 5.
    "minimum_active_decode_requests": 5,
    # DACI's paper evaluates one latency-sensitive autoregressive request.
    "active_decode_requests": 1,
    # Qwen-14B profile in DynaPipe's released gllm/worker.py, in milliseconds.
    "sample_time_ms_intercept": 1.795752,
    "sample_time_ms_per_decode_request": 0.044437,
    # This must be explicit because batch pipeline latency is not represented
    # by DACI's single-request cost model.
    "allow_exploratory_batch_mode": False,
}


def resolved_options(cfg: dict) -> dict:
    options = dict(DEFAULTS)
    options.update(cfg.get("new_baselines", {}).get("dynapipe", {}))
    return options


class DynaPipeScheme(DACIScheme):
    """Layer-redistribution controller with DynaPipe's source-level guard.

    Placement remains fixed.  That matches DynaPipe's pipeline workers: it
    redistributes layers among existing workers rather than allocating a new
    node.  Initial placement is supplied by DACI's common simulator harness,
    because DynaPipe's original A100/gLLM deployment does not define a policy
    for DACI's heterogeneous Jetson node pool.
    """

    name = "DynaPipe"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.options = resolved_options(cfg)
        self.stability_windows = max(1, int(self.options["stability_windows"]))
        self.minimum_active_decode_requests = max(
            1, int(self.options["minimum_active_decode_requests"])
        )
        self.active_decode_requests = max(1, int(self.options["active_decode_requests"]))
        self.sample_time_s = (
            float(self.options["sample_time_ms_intercept"])
            + float(self.options["sample_time_ms_per_decode_request"])
            * self.active_decode_requests
        ) / 1000.0
        self.allow_exploratory_batch_mode = bool(
            self.options["allow_exploratory_batch_mode"]
        )

        if (
            self.active_decode_requests >= self.minimum_active_decode_requests
            and not self.allow_exploratory_batch_mode
        ):
            raise ValueError(
                "DynaPipe batch adaptation is exploratory in DACI's single-request "
                "simulator. Re-run with --allow-exploratory-batch-mode to enable it."
            )

    def decide_initial(self, cluster, ms, obs, P: int, G_hat: int) -> SchemeState:
        state = super().decide_initial(cluster, ms, obs, P, G_hat)
        # SchemeState is intentionally non-slotted, so adapter-local state does
        # not alter DACI's public dataclass or serialized result contract.
        state.dynapipe_candidate = None
        state.dynapipe_candidate_streak = 0
        return state

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        phi_now = self._phi_from_obs(obs, cluster)
        u_curr = compute_u_thermal(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        incumbent_score = self._pipeline_bottleneck_s(
            cluster, ms, st.a, st.b, phi_now, obs["link_obs"], P, t_r
        )

        # This is the released DynaPipe source behavior for DACI's one-request
        # workload.  It does not make a batch-pipeline adjustment when fewer
        # than five sequences are decoding concurrently.
        if self.active_decode_requests < self.minimum_active_decode_requests:
            return st.b, False, {
                "H_r_star": 0,
                "K_r": 0,
                "J_new": None,
                "J_incumbent": incumbent_score,
                "phi_hat_curr": phi_now.tolist(),
                "phi_hat_horizon": [],
                "b_new_candidate": None,
                "u_thermal": u_curr.tolist(),
                "dynapipe_status": "inactive_below_original_min_decode_guard",
                "dynapipe_active_decode_requests": self.active_decode_requests,
            }

        candidate, candidate_score = self._choose_candidate(
            cluster, ms, st, obs, phi_now, P, t_r
        )
        candidate_key = tuple(candidate)
        previous_key = getattr(st, "dynapipe_candidate", None)
        if candidate_key == tuple(st.b):
            st.dynapipe_candidate = None
            st.dynapipe_candidate_streak = 0
            accepted = False
        elif candidate_key == previous_key:
            st.dynapipe_candidate_streak += 1
            accepted = st.dynapipe_candidate_streak >= self.stability_windows
        else:
            st.dynapipe_candidate = candidate_key
            st.dynapipe_candidate_streak = 1
            accepted = False

        if accepted:
            # Reset so that an unchanged boundary is not repeatedly charged.
            st.dynapipe_candidate = None
            st.dynapipe_candidate_streak = 0

        return (candidate if accepted else st.b), accepted, {
            "H_r_star": 0,
            "K_r": 0,
            "J_new": candidate_score,
            "J_incumbent": incumbent_score,
            "phi_hat_curr": phi_now.tolist(),
            "phi_hat_horizon": [],
            "b_new_candidate": candidate,
            "u_thermal": u_curr.tolist(),
            "dynapipe_status": "exploratory_batch_semantic_adapter",
            "dynapipe_candidate_streak": getattr(st, "dynapipe_candidate_streak", 0),
            "dynapipe_active_decode_requests": self.active_decode_requests,
        }

    def _choose_candidate(self, cluster, ms, st, obs, phi_now, P: int, t_r: int) -> Tuple[List[int], float]:
        """Select the DynaPipe-style boundary that best balances stage time.

        The original controller removes at most one layer per upstream stage
        from the last stage.  We enumerate that small set, preserve contiguous
        layer ownership, and score its pipeline bottleneck using DACI's cost
        primitives.  This is intentionally not DACI's horizon DP.
        """
        counts = [st.b[i + 1] - st.b[i] for i in range(st.S)]
        # DynaPipe's ratio is clamped by pp_size - 1; never empty the tail.
        max_shift = min(counts[-1] - 1, st.S - 1)
        best_b = list(st.b)
        best_score = self._pipeline_bottleneck_s(
            cluster, ms, st.a, best_b, phi_now, obs["link_obs"], P, t_r
        )

        for shift in range(1, max_shift + 1):
            target_counts = list(counts)
            target_counts[-1] -= shift
            # Move one layer to each upstream stage, matching DynaPipe's
            # evenly distributed tail-layer redistribution for shift <= S - 1.
            for stage_idx in range(shift):
                target_counts[stage_idx] += 1
            candidate_b = self._boundaries_from_counts(target_counts)
            if not memory_feasible(
                cluster, ms, candidate_b, st.a, obs["q_mem_obs"], P, t_r
            ):
                continue
            score = self._pipeline_bottleneck_s(
                cluster, ms, st.a, candidate_b, phi_now, obs["link_obs"], P, t_r
            )
            if score < best_score - 1e-12:
                best_b, best_score = candidate_b, score
        return best_b, best_score

    @staticmethod
    def _boundaries_from_counts(counts: List[int]) -> List[int]:
        boundaries = [0]
        for count in counts:
            boundaries.append(boundaries[-1] + count)
        return boundaries

    def _pipeline_bottleneck_s(self, cluster, ms, a, b, phi, link_obs, P: int, t_r: int) -> float:
        stage_times = []
        stage_count = len(b) - 1
        for stage in range(1, stage_count + 1):
            node_idx = a[stage - 1]
            next_node = a[stage] if stage < stage_count else None
            n_blocks = b[stage] - b[stage - 1]
            elapsed = C_stage(
                cluster, ms, node_idx, n_blocks, phi[node_idx], P, t_r, "dec"
            ) + D_stage(ms, node_idx, next_node, link_obs, P, "dec")
            if stage == stage_count:
                elapsed += self.sample_time_s
            stage_times.append(elapsed)
        return max(stage_times) if stage_times else 0.0

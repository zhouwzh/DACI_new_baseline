"""Registration and compatibility metadata for supplied new baselines."""

from __future__ import annotations

from .dynapipe import DynaPipeScheme, resolved_options


UNSUPPORTED_BASELINES = {
    "FlexPipe": (
        "FlexPipe targets multi-request serverless serving with elastic GPU allocation, "
        "pipeline-granularity changes, and resource fragmentation. DACI models one fixed "
        "edge cluster and one latency-sensitive request, so it cannot be a Table 3/Figure 5 "
        "baseline without a new workload and resource model."
    ),
    "Seesaw": (
        "Seesaw targets offline throughput and re-shards pipeline/tensor parallelism between "
        "prefill and decode. DACI does not model throughput batches, tensor parallelism, or "
        "CPU KV buffering, so direct Table 3/Figure 5 integration is unsupported."
    ),
}


def register_supported_schemes() -> None:
    """Register only in the wrapper process; do not modify DACI's SCHEMES map on disk."""
    from src.schemes.schemes import SCHEMES

    SCHEMES["DynaPipe"] = DynaPipeScheme


def validate_requested_schemes(schemes: list[str]) -> None:
    unsupported = [name for name in schemes if name in UNSUPPORTED_BASELINES]
    if unsupported:
        messages = "; ".join(f"{name}: {UNSUPPORTED_BASELINES[name]}" for name in unsupported)
        raise ValueError(
            "The requested baseline cannot be honestly executed in the DACI paper simulator. "
            f"{messages} See docs/COMPATIBILITY_AND_LIMITATIONS.md."
        )


def baseline_metadata(name: str, cfg: dict) -> dict:
    if name == "DynaPipe":
        options = resolved_options(cfg)
        active = int(options["active_decode_requests"])
        minimum = int(options["minimum_active_decode_requests"])
        mode = (
            "static_fallback_preserving_original_decode_guard"
            if active < minimum
            else "exploratory_batch_semantic_adapter"
        )
        return {
            "name": "DynaPipe",
            "paper": "DynaPipe: Dynamic Layer Redistribution for Efficient Serving of LLMs with Pipeline Parallelism, NeurIPS 2025",
            "upstream_repository": "https://github.com/xhx1022/DynaPipe",
            "upstream_revision_inspected": "69f69ad",
            "integration_mode": "semantic_equivalent_simulation_adapter",
            "adapter_mode": mode,
            "original_semantics_preserved": [
                "fixed pipeline worker set during runtime",
                "tail-stage sampling-aware layer redistribution",
                "stable-window gate before a redistribution",
                "no adjustment below the source implementation's minimum active decode guard",
            ],
            "active_decode_requests": active,
            "minimum_active_decode_requests": minimum,
            "strict_paper_table3_figure5_comparability": False,
            "reason": (
                "DynaPipe evaluates batched A100/gLLM serving; DACI evaluates one long "
                "edge request. Default DACI runs therefore preserve DynaPipe's no-adjustment "
                "guard and should be interpreted as the nearest static-pipeline fallback, not "
                "as a claim that the original DynaPipe system was reproduced."
            ),
        }
    if name in UNSUPPORTED_BASELINES:
        return {
            "name": name,
            "integration_mode": "unsupported_in_current_daci_simulator",
            "strict_paper_table3_figure5_comparability": False,
            "reason": UNSUPPORTED_BASELINES[name],
        }
    return {"name": name, "integration_mode": "existing_daci_scheme"}

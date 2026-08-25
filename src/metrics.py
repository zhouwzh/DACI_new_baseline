"""Metrics aggregation + per-window JSONL logger."""
import json
import os
import numpy as np
from typing import List, Dict
from src.simulator import TraceResult


def summarize_trace(tr: TraceResult) -> dict:
    tpot = np.array(tr.TPOT_series_s) if tr.TPOT_series_s else np.array([0.0])
    return {
        "scheme": tr.scheme,
        "seed": tr.seed,
        "TTLT_s": tr.TTLT_s,
        "TTFT_s": tr.TTFT_s,
        "TPOT_mean_ms": float(np.mean(tpot) * 1000),
        "TPOT_p99_ms": float(np.percentile(tpot, 99) * 1000),
        "overhead_s": tr.overhead_s,
        "n_reconfigs": tr.n_reconfigs,
    }


def aggregate(results: List[TraceResult]) -> Dict[str, dict]:
    by_scheme: Dict[str, List[TraceResult]] = {}
    for r in results:
        by_scheme.setdefault(r.scheme, []).append(r)
    agg = {}
    for scheme, rs in by_scheme.items():
        ttlt = [r.TTLT_s for r in rs]
        ttft = [r.TTFT_s for r in rs]
        p99_tpot = [float(np.percentile(r.TPOT_series_s, 99) * 1000) if r.TPOT_series_s else 0.0 for r in rs]
        ovhd = [r.overhead_s for r in rs]
        nrec = [r.n_reconfigs for r in rs]
        agg[scheme] = {
            "TTLT_mean_s": float(np.mean(ttlt)),
            "TTLT_std_s": float(np.std(ttlt)),
            "TTFT_mean_s": float(np.mean(ttft)),
            "TTFT_std_s": float(np.std(ttft)),
            "P99_TPOT_mean_ms": float(np.mean(p99_tpot)),
            "P99_TPOT_std_ms": float(np.std(p99_tpot)),
            "Ovhd_mean_s": float(np.mean(ovhd)),
            "Ovhd_std_s": float(np.std(ovhd)),
            "Nreconf_mean": float(np.mean(nrec)),
            "Nreconf_std": float(np.std(nrec)),
            "n_traces": len(rs),
        }
    return agg


def dump_trace_jsonl(tr: TraceResult, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({
            "header": {
                "scheme": tr.scheme, "seed": tr.seed,
                "TTLT_s": tr.TTLT_s, "TTFT_s": tr.TTFT_s,
                "overhead_s": tr.overhead_s, "n_reconfigs": tr.n_reconfigs,
            }
        }) + "\n")
        for w in tr.windows:
            f.write(json.dumps(w) + "\n")


def dump_device_jsonl(tr: TraceResult, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"header": {"scheme": tr.scheme, "seed": tr.seed,
                                         "type": "per_second_devices"}}) + "\n")
        for snap in tr.device_seconds:
            f.write(json.dumps(snap) + "\n")


def dump_tokens_jsonl(tr: TraceResult, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"header": {"scheme": tr.scheme, "seed": tr.seed,
                                         "type": "per_token"}}) + "\n")
        for tok in tr.tokens:
            f.write(json.dumps(tok) + "\n")


def dump_summary_csv(agg: Dict[str, dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["scheme", "TTLT_mean_s", "TTLT_std_s",
              "TTFT_mean_s", "TTFT_std_s",
              "P99_TPOT_mean_ms", "P99_TPOT_std_ms",
              "Ovhd_mean_s", "Ovhd_std_s",
              "Nreconf_mean", "Nreconf_std", "n_traces"]
    with open(out_path, "w") as f:
        f.write(",".join(fields) + "\n")
        for scheme, a in agg.items():
            row = [scheme] + [f"{a[k]:.4f}" if isinstance(a[k], float) else str(a[k])
                              for k in fields[1:]]
            f.write(",".join(row) + "\n")
#!/usr/bin/env python3
"""Run supported new-baseline adapters with DACI-compatible output files.

This wrapper intentionally leaves ``run.py`` and ``src/`` untouched.  It loads
the same DACI config and simulator, registers adapters in memory, then writes
the same config snapshot, metadata, summary CSV, and JSONL traces expected by
the existing paper experiment tooling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from baseline_adapters import baseline_metadata, register_supported_schemes, validate_requested_schemes
from src.cluster import build_cluster
from src.config import Config
from src.metrics import (
    aggregate,
    dump_device_jsonl,
    dump_summary_csv,
    dump_tokens_jsonl,
    dump_trace_jsonl,
    summarize_trace,
)
from src.model_spec import build_model_spec
from src.simulator import run_trace


def apply_overrides(cfg_dict: dict, args: argparse.Namespace) -> dict:
    if args.regime:
        cfg_dict["drift"]["regime"]["active"] = args.regime
    if args.W_tokens is not None:
        cfg_dict["algo"]["window"]["W_tokens"] = args.W_tokens
    if args.H_max is not None:
        cfg_dict["algo"]["window"]["H_max"] = args.H_max
    if args.lambda_slack is not None:
        cfg_dict["algo"]["switching"]["lambda_slack"] = args.lambda_slack
    if args.delta_max is not None:
        cfg_dict["algo"].setdefault("dp", {})["max_boundary_shift"] = args.delta_max
    if args.G_hat is not None:
        cfg_dict["experiment"]["request"]["G_hat_tokens"] = args.G_hat
    if args.model_name:
        cfg_dict["experiment"]["model_name"] = args.model_name
    if args.cluster_mix:
        values = [int(value) for value in args.cluster_mix.split(",")]
        if len(values) != 3:
            raise ValueError("cluster_mix must be 'high,mid,low'")
        cfg_dict["experiment"]["cluster"] = {
            "high": values[0],
            "mid": values[1],
            "low": values[2],
        }
    cfg_dict.setdefault("new_baselines", {})["dynapipe"] = {
        "stability_windows": args.dynapipe_stability_windows,
        "minimum_active_decode_requests": args.dynapipe_min_active_decode_requests,
        "active_decode_requests": args.dynapipe_active_decode_requests,
        "sample_time_ms_intercept": args.dynapipe_sample_time_ms_intercept,
        "sample_time_ms_per_decode_request": args.dynapipe_sample_time_ms_per_decode_request,
        "allow_exploratory_batch_mode": args.allow_exploratory_batch_mode,
    }
    return cfg_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated new-baseline adapters with DACI-compatible outputs."
    )
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--n_traces", type=int, default=None)
    parser.add_argument("--schemes", default="DynaPipe", help="comma-separated scheme names")
    parser.add_argument("--regime", default=None)
    parser.add_argument("--W_tokens", type=int, default=None)
    parser.add_argument("--H_max", type=int, default=None)
    parser.add_argument("--lambda_slack", type=float, default=None)
    parser.add_argument("--delta_max", type=int, default=None)
    parser.add_argument("--G_hat", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--cluster_mix", default=None, help="high,mid,low counts")
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--log_level", default="summary_only", choices=["full", "summary_only"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dynapipe-stability-windows", type=int, default=25)
    parser.add_argument("--dynapipe-min-active-decode-requests", type=int, default=5)
    parser.add_argument("--dynapipe-active-decode-requests", type=int, default=1)
    parser.add_argument("--dynapipe-sample-time-ms-intercept", type=float, default=1.795752)
    parser.add_argument("--dynapipe-sample-time-ms-per-decode-request", type=float, default=0.044437)
    parser.add_argument(
        "--allow-exploratory-batch-mode",
        action="store_true",
        help="Allow DynaPipe layer redistribution for synthetic multi-request batches. "
        "This mode is not valid for the DACI paper's single-request Table 3/Figure 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schemes = [item.strip() for item in args.schemes.split(",") if item.strip()]
    if not schemes:
        raise ValueError("At least one scheme is required")
    validate_requested_schemes(schemes)
    register_supported_schemes()

    cfg = Config.from_dir(args.config_dir)
    cfg_dict = apply_overrides(cfg.to_dict(), args)
    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    n_traces = args.n_traces or cfg_dict["experiment"]["n_traces"]
    seed_start = (
        args.seed_start
        if args.seed_start is not None
        else cfg_dict["experiment"]["seed_base"]
    )
    metadata = {
        "run_id": args.run_id,
        "runner": "gpt_new_baseline_code/run_new_baselines.py",
        "regime": cfg_dict["drift"]["regime"]["active"],
        "model_name": cfg_dict["experiment"]["model_name"],
        "cluster": cfg_dict["experiment"]["cluster"],
        "W_tokens": cfg_dict["algo"]["window"]["W_tokens"],
        "H_max": cfg_dict["algo"]["window"]["H_max"],
        "lambda_slack": cfg_dict["algo"]["switching"]["lambda_slack"],
        "G_hat": cfg_dict["experiment"]["request"]["G_hat_tokens"],
        "n_traces": n_traces,
        "seed_start": seed_start,
        "schemes": schemes,
        "log_level": args.log_level,
        "baseline_adapters": [baseline_metadata(name, cfg_dict) for name in schemes],
    }
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8"
    )
    (out_dir / "experiment_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    cluster = build_cluster(cfg_dict, cfg_dict["experiment"]["cluster"])
    model_spec = build_model_spec(cfg_dict["experiment"]["model_name"], cfg_dict["models"])
    all_results = []
    started = time.time()
    print(f"== New-baseline run: {args.run_id} ==")
    print(f"Output: {out_dir}")
    print(
        f"Model={model_spec.name} | G_hat={metadata['G_hat']} | "
        f"traces={n_traces} | schemes={','.join(schemes)}"
    )

    for scheme_name in schemes:
        for offset in range(n_traces):
            seed = seed_start + offset
            trace = run_trace(
                cluster, model_spec, cfg_dict, scheme_name, seed, verbose=args.verbose
            )
            summary = summarize_trace(trace)
            all_results.append(trace)
            print(
                f"  {scheme_name} seed={seed} TTLT={trace.TTLT_s:.2f}s "
                f"P99TPOT={summary['TPOT_p99_ms']:.2f}ms "
                f"Ovhd={trace.overhead_s:.2f}s #Rec={trace.n_reconfigs}"
            )
            trace_base = out_dir / "traces" / f"{scheme_name}_seed{seed}"
            dump_trace_jsonl(trace, str(trace_base) + ".jsonl")
            if args.log_level == "full":
                dump_device_jsonl(trace, str(trace_base) + "_devices.jsonl")
                dump_tokens_jsonl(trace, str(trace_base) + "_tokens.jsonl")

    dump_summary_csv(aggregate(all_results), str(out_dir / "summary.csv"))
    print(f"Completed in {time.time() - started:.1f}s: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

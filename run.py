"""CLI: load config, run traces x schemes, dump outputs.

Supports CLI overrides for batch experiments:
  --regime, --W_tokens, --H_max, --lambda_slack, --G_hat
  --model_name, --cluster_mix "h,m,l"
  --ablation {none,no_predictor,no_lazy,no_bottleneck,no_adaptive_H}
  --seed_start, --log_level {full,summary_only}
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.cluster import build_cluster
from src.model_spec import build_model_spec
from src.simulator import run_trace
from src.metrics import aggregate, dump_trace_jsonl, dump_summary_csv, summarize_trace, dump_device_jsonl, dump_tokens_jsonl


def apply_overrides(cfg_dict: dict, args) -> dict:
    if args.regime:
        cfg_dict["drift"]["regime"]["active"] = args.regime
    if args.W_tokens is not None:
        cfg_dict["algo"]["window"]["W_tokens"] = args.W_tokens
    if args.H_max is not None:
        cfg_dict["algo"]["window"]["H_max"] = args.H_max
    if args.lambda_slack is not None:
        cfg_dict["algo"]["switching"]["lambda_slack"] = args.lambda_slack
    if args.delta_max is not None:
        cfg_dict["algo"].setdefault("dp", {})
        cfg_dict["algo"]["dp"]["max_boundary_shift"] = args.delta_max
    if args.G_hat is not None:
        cfg_dict["experiment"]["request"]["G_hat_tokens"] = args.G_hat
    if args.model_name:
        cfg_dict["experiment"]["model_name"] = args.model_name
    if args.cluster_mix:
        parts = [int(x) for x in args.cluster_mix.split(",")]
        assert len(parts) == 3, "cluster_mix must be 'high,mid,low'"
        cfg_dict["experiment"]["cluster"] = {
            "high": parts[0], "mid": parts[1], "low": parts[2]
        }
    if args.ablation and args.ablation != "none":
        cfg_dict.setdefault("ablation", {})
        cfg_dict["ablation"]["mode"] = args.ablation
    return cfg_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_dir", default="configs")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument("--n_traces", type=int, default=None)
    p.add_argument("--schemes", default=None, help="comma-separated")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--regime", default=None)
    p.add_argument("--W_tokens", type=int, default=None)
    p.add_argument("--H_max", type=int, default=None)
    p.add_argument("--lambda_slack", type=float, default=None)
    p.add_argument("--delta_max", type=int, default=None,
                   help="Max per-window boundary shift (|b_s^r - b_s^{r-1}| <= delta_max)")
    p.add_argument("--G_hat", type=int, default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--cluster_mix", default=None, help="'high,mid,low' counts e.g. '1,3,4'")
    p.add_argument("--ablation", default="none",
                   choices=["none", "no_predictor", "no_lazy", "no_bottleneck", "no_adaptive_H"])
    p.add_argument("--seed_start", type=int, default=None)
    p.add_argument("--log_level", default="full", choices=["full", "summary_only"])
    args = p.parse_args()

    cfg = Config.from_dir(args.config_dir)
    cfg_dict = cfg.to_dict()
    apply_overrides(cfg_dict, args)

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or cfg_dict["experiment"]["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config_snapshot.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    meta = {
        "run_id": run_id,
        "regime": cfg_dict["drift"]["regime"]["active"],
        "model_name": cfg_dict["experiment"]["model_name"],
        "cluster": cfg_dict["experiment"]["cluster"],
        "W_tokens": cfg_dict["algo"]["window"]["W_tokens"],
        "H_max": cfg_dict["algo"]["window"]["H_max"],
        "lambda_slack": cfg_dict["algo"]["switching"]["lambda_slack"],
        "G_hat": cfg_dict["experiment"]["request"]["G_hat_tokens"],
        "ablation": args.ablation,
        "n_traces": args.n_traces or cfg_dict["experiment"]["n_traces"],
        "schemes": (args.schemes.split(",") if args.schemes else cfg_dict["experiment"]["schemes"]),
    }
    with open(out_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    cl = build_cluster(cfg_dict, cfg_dict["experiment"]["cluster"])
    ms = build_model_spec(cfg_dict["experiment"]["model_name"], cfg_dict["models"])

    schemes = meta["schemes"]
    n_traces = meta["n_traces"]
    seed_base = args.seed_start if args.seed_start is not None else cfg_dict["experiment"]["seed_base"]

    print(f"== Run: {run_id} ==")
    print(f"Output: {out_dir}")
    print(f"Regime={meta['regime']} | Ablation={args.ablation} | Model={ms.name} L={ms.L}")
    print(f"W={meta['W_tokens']} Hmax={meta['H_max']} lambda={meta['lambda_slack']} G_hat={meta['G_hat']}")
    print(f"Cluster={meta['cluster']} N={cl.N}")
    print(f"Schemes={schemes} | n_traces={n_traces} seed_start={seed_base}")

    all_results = []
    t0 = time.time()
    for scheme_name in schemes:
        print(f"\n-- Scheme: {scheme_name} --")
        for i in range(n_traces):
            seed = seed_base + i
            t_start = time.time()
            tr = run_trace(cl, ms, cfg_dict, scheme_name, seed, verbose=args.verbose)
            dt = time.time() - t_start
            summary = summarize_trace(tr)
            print(f"  seed={seed} TTLT={tr.TTLT_s:.2f}s TTFT={tr.TTFT_s:.2f}s "
                  f"P99TPOT={summary['TPOT_p99_ms']:.1f}ms "
                  f"Ovhd={tr.overhead_s:.2f}s #Rec={tr.n_reconfigs} ({dt:.1f}s)")
            all_results.append(tr)
            if args.log_level == "full":
                dump_trace_jsonl(tr, str(out_dir / "traces" / f"{scheme_name}_seed{seed}.jsonl"))
                dump_device_jsonl(tr, str(out_dir / "traces" / f"{scheme_name}_seed{seed}_devices.jsonl"))
                dump_tokens_jsonl(tr, str(out_dir / "traces" / f"{scheme_name}_seed{seed}_tokens.jsonl"))
            else:
                dump_trace_jsonl(tr, str(out_dir / "traces" / f"{scheme_name}_seed{seed}.jsonl"))

    agg = aggregate(all_results)
    dump_summary_csv(agg, str(out_dir / "summary.csv"))
    print(f"\n== Summary ==")
    for scheme, a in agg.items():
        print(f"  {scheme}: TTLT={a['TTLT_mean_s']:.2f}±{a['TTLT_std_s']:.2f}s, "
              f"P99TPOT={a['P99_TPOT_mean_ms']:.1f}±{a['P99_TPOT_std_ms']:.1f}ms, "
              f"Ovhd={a['Ovhd_mean_s']:.2f}±{a['Ovhd_std_s']:.2f}s, "
              f"#Rec={a['Nreconf_mean']:.2f}±{a['Nreconf_std']:.2f}")
    print(f"Total wall time: {time.time()-t0:.1f}s")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
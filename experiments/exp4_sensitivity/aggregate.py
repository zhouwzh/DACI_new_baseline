#!/usr/bin/env python3
"""Aggregate experiment outputs into CSVs for plotting.

Usage:
    python experiments/aggregate.py <mode> <experiment_dir>
    modes: overall, ablation, regime, sensitivity, scalability, predictor
"""
import sys
import json
import csv
from pathlib import Path
import numpy as np


def read_summary(subdir: Path):
    """Read summary.csv into list of dicts (one per scheme)."""
    p = subdir / "summary.csv"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_meta(subdir: Path):
    p = subdir / "experiment_meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def collect_flat(exp_dir: Path):
    """Walk all subdirs with summary.csv, return list of rows with meta merged."""
    out = []
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue
        meta = read_meta(sub)
        for row in read_summary(sub):
            merged = {"subdir": sub.name, **meta, **row}
            out.append(merged)
    return out


def write_csv(rows, out_path: Path, fields=None):
    if not rows:
        print(f"no rows for {out_path}")
        return
    fields = fields or sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {out_path}")


def agg_overall(exp_dir: Path):
    rows = collect_flat(exp_dir)
    fields = ["subdir", "scheme",
              "TTLT_mean_s", "TTLT_std_s",
              "TTFT_mean_s", "TTFT_std_s",
              "P99_TPOT_mean_ms", "P99_TPOT_std_ms",
              "Ovhd_mean_s", "Ovhd_std_s",
              "Nreconf_mean", "Nreconf_std"]
    write_csv(rows, exp_dir / "_aggregate_overall.csv", fields)


def agg_ablation(exp_dir: Path):
    rows = collect_flat(exp_dir)
    fields = ["subdir", "ablation", "scheme",
              "TTLT_mean_s", "TTLT_std_s",
              "P99_TPOT_mean_ms", "P99_TPOT_std_ms",
              "Ovhd_mean_s", "Ovhd_std_s",
              "Nreconf_mean", "Nreconf_std"]
    write_csv(rows, exp_dir / "_aggregate_ablation.csv", fields)


def agg_regime(exp_dir: Path):
    # subdirs like "R1_calibrated_DACI"; split back
    rows = []
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue
        meta = read_meta(sub)
        for row in read_summary(sub):
            merged = {"subdir": sub.name, "regime": meta.get("regime", "?"),
                      **row}
            rows.append(merged)
    fields = ["subdir", "regime", "scheme",
              "TTLT_mean_s", "TTLT_std_s",
              "P99_TPOT_mean_ms", "P99_TPOT_std_ms",
              "Ovhd_mean_s", "Ovhd_std_s",
              "Nreconf_mean", "Nreconf_std"]
    write_csv(rows, exp_dir / "_aggregate_regime.csv", fields)


def agg_sensitivity(exp_dir: Path):
    # Run for each of {W_sweep, H_sweep, lambda_sweep}
    for sub_exp in ["W_sweep", "H_sweep", "lambda_sweep"]:
        sub_path = exp_dir / sub_exp
        if not sub_path.exists():
            continue
        rows = collect_flat(sub_path)
        fields = ["subdir", "W_tokens", "H_max", "lambda_slack", "scheme",
                  "TTLT_mean_s", "TTLT_std_s",
                  "P99_TPOT_mean_ms", "P99_TPOT_std_ms",
                  "Ovhd_mean_s", "Ovhd_std_s",
                  "Nreconf_mean", "Nreconf_std"]
        write_csv(rows, sub_path / "_aggregate.csv", fields)


def agg_scalability(exp_dir: Path):
    for sub_exp in ["N_sweep", "L_sweep", "G_sweep"]:
        sub_path = exp_dir / sub_exp
        if not sub_path.exists():
            continue
        rows = collect_flat(sub_path)
        fields = ["subdir", "model_name", "cluster", "G_hat", "scheme",
                  "TTLT_mean_s", "TTLT_std_s",
                  "P99_TPOT_mean_ms", "P99_TPOT_std_ms",
                  "Ovhd_mean_s", "Ovhd_std_s",
                  "Nreconf_mean", "Nreconf_std"]
        write_csv(rows, sub_path / "_aggregate.csv", fields)


def agg_predictor(exp_dir: Path):
    """Compute RMSE of phi_hat vs phi_true at multiple leads k.

    For each window r, the simulator dumps:
      - phi_true: ground-truth phi evaluated at the window itself (r)
      - phi_hat_horizon: the H_max-step forecast made AT r, i.e. phi_hat[:, 0..H-1]
        where index k is the forecast for r+k.

    For lead k, residual = phi_true[r+k] - phi_hat_horizon[r][:, k].
    RMSE_k = sqrt(mean over (r, node) of residual^2).

    Also computes a persistence baseline (zero-knowledge): predicting phi(r+k)
    as phi(r) itself -> RMSE_k_persist = sqrt(mean (phi_true[r+k] - phi_true[r])^2).
    """
    per_lead = {}       # k -> list of squared residuals (kalman+ar1)
    per_lead_pers = {}  # k -> list (persistence baseline)

    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue
        traces_dir = sub / "traces"
        if not traces_dir.exists():
            continue
        for jsonl in traces_dir.glob("DACI_seed*.jsonl"):
            if "devices" in jsonl.name or "tokens" in jsonl.name:
                continue
            with open(jsonl) as f:
                windows = [json.loads(l) for l in f][1:]  # skip header
            if not windows:
                continue
            # Build arrays: phi_true[r, n] and phi_hat_horizon[r, n, k]
            phi_true_seq = []
            phi_hat_seq = []
            for w in windows:
                pt = w.get("phi_true", [])
                phh = w.get("phi_hat_horizon", [])
                if not pt or not phh:
                    continue
                phi_true_seq.append(pt)
                phi_hat_seq.append(phh)
            if len(phi_true_seq) < 2:
                continue
            phi_true = np.array(phi_true_seq)            # (R, N)
            phi_hat_hz = np.array(phi_hat_seq)           # (R, N, H_max)
            R, N = phi_true.shape
            H_max = phi_hat_hz.shape[2]

            # Kalman+AR(1) residuals at each k
            for k in range(H_max):
                if R <= k:
                    continue
                pred = phi_hat_hz[:R - k, :, k]          # forecast from r for time r+k
                actual = phi_true[k:R, :]                # phi_true at r+k
                diff2 = (actual - pred) ** 2
                per_lead.setdefault(k, []).extend(diff2.flatten().tolist())

            # Persistence baseline at each k: predict phi(r+k) as phi(r)
            for k in range(H_max):
                if R <= k:
                    continue
                pred = phi_true[:R - k, :]
                actual = phi_true[k:R, :]
                diff2 = (actual - pred) ** 2
                per_lead_pers.setdefault(k, []).extend(diff2.flatten().tolist())

    out_path = exp_dir / "_predictor_rmse.csv"
    with open(out_path, "w") as f:
        f.write("lead_k,method,RMSE,n_samples\n")
        for k in sorted(per_lead.keys()):
            rmse = float(np.sqrt(np.mean(per_lead[k])))
            f.write(f"{k},kalman_ar1,{rmse:.6f},{len(per_lead[k])}\n")
        for k in sorted(per_lead_pers.keys()):
            rmse = float(np.sqrt(np.mean(per_lead_pers[k])))
            f.write(f"{k},persistence,{rmse:.6f},{len(per_lead_pers[k])}\n")

    print(f"wrote -> {out_path}")
    if per_lead:
        print("\nRMSE at key leads:")
        for k in [1, 4, 8]:
            if k in per_lead:
                rmse_ka = float(np.sqrt(np.mean(per_lead[k])))
                rmse_pe = float(np.sqrt(np.mean(per_lead_pers[k]))) if k in per_lead_pers else float("nan")
                print(f"  k={k}: Kalman+AR1 RMSE={rmse_ka:.4f}  |  Persistence RMSE={rmse_pe:.4f}")
            else:
                print(f"  k={k}: not in horizon (H_max < {k+1})")
    else:
        print("No phi_hat_horizon data found. Re-run traces with log_level=full after simulator patch.")


MODES = {
    "overall": agg_overall,
    "ablation": agg_ablation,
    "regime": agg_regime,
    "sensitivity": agg_sensitivity,
    "scalability": agg_scalability,
    "predictor": agg_predictor,
}


def main():
    if len(sys.argv) < 3:
        print("Usage: python aggregate.py <mode> <experiment_dir>")
        print(f"Modes: {list(MODES.keys())}")
        sys.exit(1)
    mode, exp_dir = sys.argv[1], Path(sys.argv[2])
    if mode not in MODES:
        print(f"Unknown mode: {mode}. Choose from {list(MODES.keys())}")
        sys.exit(1)
    if not exp_dir.exists():
        print(f"Dir not found: {exp_dir}")
        sys.exit(1)
    MODES[mode](exp_dir)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Create a long-form Table 3 extension from standard DACI summary.csv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_DIRS = {
    "Gemma3-4B": "exp1_overall_small",
    "Llama-3-8B": "exp1_overall_medium",
    "Qwen3-14B": "exp1_overall_large",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    rows = []
    for model, directory in MODEL_DIRS.items():
        experiment_dir = outputs_root / directory
        if not experiment_dir.is_dir():
            continue
        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            summary_path = run_dir / "summary.csv"
            if not summary_path.is_file():
                continue
            with summary_path.open(newline="", encoding="utf-8") as handle:
                for summary in csv.DictReader(handle):
                    rows.append({
                        "model": model,
                        "run_id": run_dir.name,
                        "source_summary_csv": str(summary_path),
                        **summary,
                    })
    fieldnames = [
        "model", "run_id", "scheme", "TTLT_mean_s", "TTLT_std_s",
        "TTFT_mean_s", "TTFT_std_s", "P99_TPOT_mean_ms", "P99_TPOT_std_ms",
        "Ovhd_mean_s", "Ovhd_std_s", "Nreconf_mean", "Nreconf_std",
        "n_traces", "source_summary_csv",
    ]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

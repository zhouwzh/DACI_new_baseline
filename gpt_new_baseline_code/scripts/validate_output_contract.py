#!/usr/bin/env python3
"""Validate the files required by DACI's experiment-output contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUMMARY_FIELDS = {
    "scheme", "TTLT_mean_s", "TTLT_std_s", "TTFT_mean_s", "TTFT_std_s",
    "P99_TPOT_mean_ms", "P99_TPOT_std_ms", "Ovhd_mean_s", "Ovhd_std_s",
    "Nreconf_mean", "Nreconf_std", "n_traces",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    for name in ("config_snapshot.json", "experiment_meta.json", "summary.csv"):
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required output: {path}")
    for name in ("config_snapshot.json", "experiment_meta.json"):
        with (run_dir / name).open(encoding="utf-8") as handle:
            json.load(handle)
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = SUMMARY_FIELDS - fields
        if missing:
            raise ValueError(f"summary.csv is missing fields: {sorted(missing)}")
        if not list(reader):
            raise ValueError("summary.csv contains no result rows")
    traces = list((run_dir / "traces").glob("*.jsonl")) if (run_dir / "traces").is_dir() else []
    if not traces:
        raise FileNotFoundError("No trace JSONL files found")
    for trace_path in traces:
        with trace_path.open(encoding="utf-8") as handle:
            first_line = handle.readline()
        if not first_line or "header" not in json.loads(first_line):
            raise ValueError(f"Invalid trace header: {trace_path}")
    print(f"Output contract is valid: {run_dir}")


if __name__ == "__main__":
    main()

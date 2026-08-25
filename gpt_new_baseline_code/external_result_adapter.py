#!/usr/bin/env python3
"""Convert externally measured, already-aligned baseline traces to DACI output.

This is intentionally a result adapter, not a fake launcher for FlexPipe or
Seesaw.  It accepts empirical measurements only after the caller has matched
the DACI request definition, model, hardware assumptions, and metric units.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import aggregate, dump_device_jsonl, dump_summary_csv, dump_tokens_jsonl, dump_trace_jsonl
from src.simulator import TraceResult


REQUIRED = {"seed", "TTLT_s", "TTFT_s", "TPOT_series_s", "overhead_s", "n_reconfigs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt externally measured traces to DACI output files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--attestation",
        required=True,
        help="Path to a human-written record of matched model, hardware, workload, and units.",
    )
    parser.add_argument("--log-level", default="summary_only", choices=["full", "summary_only"])
    return parser.parse_args()


def load_trace(record: dict, baseline: str) -> TraceResult:
    missing = REQUIRED - set(record)
    if missing:
        raise ValueError(f"External trace is missing fields: {sorted(missing)}")
    return TraceResult(
        scheme=record.get("scheme", baseline),
        seed=int(record["seed"]),
        TTLT_s=float(record["TTLT_s"]),
        TTFT_s=float(record["TTFT_s"]),
        TPOT_series_s=[float(value) for value in record["TPOT_series_s"]],
        overhead_s=float(record["overhead_s"]),
        n_reconfigs=int(record["n_reconfigs"]),
        windows=list(record.get("windows", [])),
        device_seconds=list(record.get("device_seconds", [])),
        tokens=list(record.get("tokens", [])),
    )


def main() -> None:
    args = parse_args()
    attestation = Path(args.attestation)
    if not attestation.is_file():
        raise FileNotFoundError(f"Attestation file does not exist: {attestation}")
    records = []
    with Path(args.input_jsonl).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(load_trace(json.loads(line), args.baseline))
                except Exception as exc:
                    raise ValueError(f"Invalid record at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("No trace records found in input JSONL")

    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": args.run_id,
        "runner": "gpt_new_baseline_code/external_result_adapter.py",
        "baseline": args.baseline,
        "integration_mode": "external_result_adapter",
        "attestation": str(attestation.resolve()),
        "n_traces": len(records),
        "schema": sorted(REQUIRED),
        "warning": "The adapter preserves supplied measurements; it does not validate scientific comparability.",
    }
    (out_dir / "experiment_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "config_snapshot.json").write_text(
        json.dumps({"external_result_adapter": metadata}, indent=2), encoding="utf-8"
    )
    for trace in records:
        base = out_dir / "traces" / f"{trace.scheme}_seed{trace.seed}"
        dump_trace_jsonl(trace, str(base) + ".jsonl")
        if args.log_level == "full":
            dump_device_jsonl(trace, str(base) + "_devices.jsonl")
            dump_tokens_jsonl(trace, str(base) + "_tokens.jsonl")
    dump_summary_csv(aggregate(records), str(out_dir / "summary.csv"))
    print(f"Adapted {len(records)} externally measured traces to {out_dir}")


if __name__ == "__main__":
    main()

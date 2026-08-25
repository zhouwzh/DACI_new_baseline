#!/usr/bin/env python3
"""Create Figure 5 extension data from standard DACI G-sweep summaries."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RUN_ID = re.compile(r"^G_(?P<G>\d+)_(?P<scheme>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for run_dir in sorted(path for path in Path(args.sweep_dir).iterdir() if path.is_dir()):
        match = RUN_ID.match(run_dir.name)
        summary_path = run_dir / "summary.csv"
        if match is None or not summary_path.is_file():
            continue
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            records.append({
                "G_hat": int(match.group("G")),
                "run_id": run_dir.name,
                "scheme": row["scheme"],
                "TTLT_mean_s": float(row["TTLT_mean_s"]),
                "TTLT_std_s": float(row["TTLT_std_s"]),
                "P99_TPOT_mean_ms": float(row["P99_TPOT_mean_ms"]),
                "Ovhd_mean_s": float(row["Ovhd_mean_s"]),
                "n_traces": int(row["n_traces"]),
                "source_summary_csv": str(summary_path),
            })
    ttlt_by_g_scheme = {(row["G_hat"], row["scheme"]): row["TTLT_mean_s"] for row in records}
    for row in records:
        daci = ttlt_by_g_scheme.get((row["G_hat"], "DACI"))
        if daci is None or row["scheme"] == "DACI":
            row["daci_relative_ttlt_change_pct"] = ""
        else:
            row["daci_relative_ttlt_change_pct"] = round(
                100.0 * (row["TTLT_mean_s"] - daci) / row["TTLT_mean_s"], 4
            )
    records.sort(key=lambda row: (row["G_hat"], row["scheme"]))
    fieldnames = [
        "G_hat", "run_id", "scheme", "TTLT_mean_s", "TTLT_std_s",
        "P99_TPOT_mean_ms", "Ovhd_mean_s", "n_traces",
        "daci_relative_ttlt_change_pct", "source_summary_csv",
    ]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()

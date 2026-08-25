"""Smoke tests for the isolated new-baseline runner."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
NEW_BASELINE_ROOT = HERE.parents[1]
REPO_ROOT = NEW_BASELINE_ROOT.parent
RUNNER = NEW_BASELINE_ROOT / "run_new_baselines.py"
VALIDATOR = NEW_BASELINE_ROOT / "scripts" / "validate_output_contract.py"
EXTERNAL_ADAPTER = NEW_BASELINE_ROOT / "external_result_adapter.py"


class DynaPipeAdapterTest(unittest.TestCase):
    def test_single_request_fallback_writes_daci_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            command = [
                sys.executable,
                str(RUNNER),
                "--config_dir", str(REPO_ROOT / "configs"),
                "--output_dir", temp_dir,
                "--run_id", "dynapipe_smoke",
                "--schemes", "DynaPipe",
                "--n_traces", "1",
                "--seed_start", "42",
                "--model_name", "gemma3-4b",
                "--cluster_mix", "1,1,0",
                "--G_hat", "40",
                "--log_level", "full",
            ]
            subprocess.run(command, check=True, cwd=REPO_ROOT, capture_output=True, text=True)
            run_dir = Path(temp_dir) / "dynapipe_smoke"
            subprocess.run(
                [sys.executable, str(VALIDATOR), str(run_dir)],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            with (run_dir / "experiment_meta.json").open(encoding="utf-8") as handle:
                meta = json.load(handle)
            adapter = meta["baseline_adapters"][0]
            self.assertEqual(adapter["adapter_mode"], "static_fallback_preserving_original_decode_guard")
            self.assertFalse(adapter["strict_paper_table3_figure5_comparability"])
            with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["scheme"], "DynaPipe")

    def test_batch_mode_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            command = [
                sys.executable,
                str(RUNNER),
                "--config_dir", str(REPO_ROOT / "configs"),
                "--output_dir", temp_dir,
                "--run_id", "must_fail",
                "--schemes", "DynaPipe",
                "--n_traces", "1",
                "--G_hat", "20",
                "--dynapipe-active-decode-requests", "5",
            ]
            result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("allow-exploratory-batch-mode", result.stderr)

    def test_unsupported_baseline_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--output_dir", temp_dir,
                    "--run_id", "seesaw_must_fail",
                    "--schemes", "Seesaw",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot be honestly executed", result.stderr)

    def test_external_result_adapter_writes_daci_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "external.jsonl"
            source.write_text(
                json.dumps({
                    "seed": 42,
                    "TTLT_s": 10.0,
                    "TTFT_s": 1.0,
                    "TPOT_series_s": [0.1, 0.2],
                    "overhead_s": 0.5,
                    "n_reconfigs": 1,
                }) + "\n",
                encoding="utf-8",
            )
            attestation = temp_path / "attestation.md"
            attestation.write_text("Synthetic test measurement only.\n", encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(EXTERNAL_ADAPTER),
                    "--baseline", "Seesaw",
                    "--input-jsonl", str(source),
                    "--attestation", str(attestation),
                    "--output-dir", temp_dir,
                    "--run-id", "external_smoke",
                    "--log-level", "full",
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            run_dir = temp_path / "external_smoke"
            subprocess.run(
                [sys.executable, str(VALIDATOR), str(run_dir)],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env bash
# Exp4: §5.5 Sensitivity Analysis + Predictor Accuracy (all four sub-experiments).
# Runs W_sweep, H_sweep, lambda_sweep, predictor_accuracy.
# Default model: qwen3-14b. 5 traces per point. Sweeps run sequentially; within each
# sweep, sweep values run in parallel (PARALLEL_JOBS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================================"
echo "  Exp4-A: W sweep"
echo "============================================================"
bash "${SCRIPT_DIR}/W_sweep.sh"

echo ""
echo "============================================================"
echo "  Exp4-B: H_max sweep"
echo "============================================================"
bash "${SCRIPT_DIR}/H_sweep.sh"

echo ""
echo "============================================================"
echo "  Exp4-C: lambda sweep"
echo "============================================================"
bash "${SCRIPT_DIR}/lambda_sweep.sh"

echo ""
echo "============================================================"
echo "  Exp4-D: predictor accuracy"
echo "============================================================"
bash "${SCRIPT_DIR}/predictor_accuracy.sh"

echo ""
echo "=== All exp4 sweeps complete ==="
echo "Aggregate sensitivity CSVs:"
echo "  python experiments/aggregate.py sensitivity outputs/exp4_sensitivity"
echo "Aggregate predictor RMSE:"
echo "  python experiments/aggregate.py predictor outputs/exp4_sensitivity/predictor_accuracy"

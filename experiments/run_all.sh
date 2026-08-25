#!/usr/bin/env bash
# Run all 5 experiments end-to-end. Each experiment internally parallelizes.
# Across-experiment parallelism is OFF by default (they fight for CPU); enable
# via PARALLEL_ACROSS=1 if you have spare cores.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARALLEL_ACROSS=${PARALLEL_ACROSS:-0}
export PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

EXPS=(
    "experiments/exp1_overall/run.sh"
    "experiments/exp2_ablation/run.sh"
    "experiments/exp3_regime/run.sh"
    "experiments/exp4_sensitivity/W_sweep.sh"
    "experiments/exp4_sensitivity/H_sweep.sh"
    "experiments/exp4_sensitivity/lambda_sweep.sh"
    "experiments/exp4_sensitivity/delta_sweep.sh"
    "experiments/exp4_sensitivity/predictor_accuracy.sh"
    "experiments/exp5_scalability/N_sweep.sh"
    "experiments/exp5_scalability/L_sweep.sh"
    "experiments/exp5_scalability/G_sweep.sh"
)

if [ "${PARALLEL_ACROSS}" = "1" ]; then
    for e in "${EXPS[@]}"; do
        bash "${e}" &
    done
    wait
else
    for e in "${EXPS[@]}"; do
        echo ""
        echo "=============================================="
        echo "Running: ${e}"
        echo "=============================================="
        bash "${e}"
    done
fi

echo ""
echo "=== All experiments done. Aggregating ==="
python experiments/aggregate.py overall outputs/exp1_overall
python experiments/aggregate.py ablation outputs/exp2_ablation
python experiments/aggregate.py regime outputs/exp3_regime
python experiments/aggregate.py sensitivity outputs/exp4_sensitivity
python experiments/aggregate.py scalability outputs/exp5_scalability
python experiments/aggregate.py predictor outputs/exp4_sensitivity/predictor_accuracy

echo ""
echo "Done. CSVs for plotting are in outputs/*/_aggregate*.csv"

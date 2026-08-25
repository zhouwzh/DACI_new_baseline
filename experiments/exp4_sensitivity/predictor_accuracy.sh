#!/usr/bin/env bash
# Exp4-D: §5.5 Predictor accuracy — RMSE of phi_hat vs phi_true at lead k in {0,...,H_max}.
# Default model: qwen3-14b. Uses FULL log level so per-window phi_hat_curr and phi_true
# are recorded for offline RMSE computation via aggregate.py predictor mode.
# Runs one trace per seed (n_traces=1) across N_TRACES distinct seeds.
set -euo pipefail

EXP_NAME="exp4_sensitivity/predictor_accuracy"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=${N_TRACES:-1}
SEED_START=${SEED_START:-42}
MODEL_NAME=${MODEL_NAME:-qwen3-14b}
PARALLEL_JOBS=${PARALLEL_JOBS:-5}

cd "${PROJECT_ROOT}"

run_one() {
    local seed="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "seed_${seed}" \
        --schemes DACI \
        --n_traces 1 \
        --seed_start "${seed}" \
        --regime default \
        --model_name "${MODEL_NAME}" \
        --log_level full \
        2>&1 | sed "s|^|[seed=${seed}] |" | tee "${OUTPUT_DIR}/seed_${seed}.log"
    echo "[done] seed=${seed}"
}

export -f run_one
export OUTPUT_DIR MODEL_NAME PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
SEEDS=()
for i in $(seq 0 $((N_TRACES-1))); do
    SEEDS+=($((SEED_START+i)))
done

printf '%s\n' "${SEEDS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo ""
echo "=== predictor_accuracy complete. Model=${MODEL_NAME} Seeds=${N_TRACES} ==="
echo "Next: python experiments/aggregate.py predictor ${OUTPUT_DIR}"
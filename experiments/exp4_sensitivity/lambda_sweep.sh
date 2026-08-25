#!/usr/bin/env bash
# Exp4-C: §5.5 Robust slack lambda sweep.
# Default model: qwen3-14b. 5 traces per lambda value.
set -euo pipefail

EXP_NAME="exp4_sensitivity/lambda_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=${N_TRACES:-1}
SEED_START=${SEED_START:-42}
MODEL_NAME=${MODEL_NAME:-qwen3-14b}
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

LAMBDA_VALUES=(5 10)

run_one() {
    local L="$1"
    local tag="lambda_${L//./_}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "${tag}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --model_name "${MODEL_NAME}" \
        --lambda_slack "${L}" \
        --log_level summary_only \
        2>&1 | sed "s|^|[${tag}] |" | tee "${OUTPUT_DIR}/${tag}.log"
    echo "[done] lambda=${L}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START MODEL_NAME PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${LAMBDA_VALUES[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo ""
echo "=== lambda_sweep complete. Model=${MODEL_NAME} Traces=${N_TRACES} ==="

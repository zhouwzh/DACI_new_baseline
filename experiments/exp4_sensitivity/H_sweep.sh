#!/usr/bin/env bash
# Exp4-B: §5.5 Horizon ceiling H_max sweep.
# Default model: qwen3-14b. 5 traces per H value.
set -euo pipefail

EXP_NAME="exp4_sensitivity/H_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=${N_TRACES:-1}
SEED_START=${SEED_START:-42}
MODEL_NAME=${MODEL_NAME:-qwen3-14b}
PARALLEL_JOBS=${PARALLEL_JOBS:-5}

cd "${PROJECT_ROOT}"

H_VALUES=(6 10 14)

run_one() {
    local H="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "H_${H}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --model_name "${MODEL_NAME}" \
        --H_max "${H}" \
        --log_level summary_only \
        2>&1 | sed "s|^|[H=${H}] |" | tee "${OUTPUT_DIR}/H_${H}.log"
    echo "[done] H_max=${H}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START MODEL_NAME PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${H_VALUES[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo ""
echo "=== H_sweep complete. Model=${MODEL_NAME} Traces=${N_TRACES} ==="

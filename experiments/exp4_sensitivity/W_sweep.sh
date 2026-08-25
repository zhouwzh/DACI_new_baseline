#!/usr/bin/env bash
# Exp4-A: §5.5 Window length W sweep.
# Default model: qwen3-14b (largest). 5 traces per W value.
set -euo pipefail

EXP_NAME="exp4_sensitivity/W_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=${N_TRACES:-1}
SEED_START=${SEED_START:-42}
MODEL_NAME=${MODEL_NAME:-qwen3-14b}
PARALLEL_JOBS=${PARALLEL_JOBS:-5}

cd "${PROJECT_ROOT}"

W_VALUES=(5 10 20 50 100)

run_one() {
    local W="$1"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "W_${W}" \
        --schemes DACI \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --model_name "${MODEL_NAME}" \
        --W_tokens "${W}" \
        --log_level summary_only \
        2>&1 | sed "s|^|[W=${W}] |" | tee "${OUTPUT_DIR}/W_${W}.log"
    echo "[done] W=${W}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START MODEL_NAME PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${W_VALUES[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo ""
echo "=== W_sweep complete. Model=${MODEL_NAME} Traces=${N_TRACES} ==="

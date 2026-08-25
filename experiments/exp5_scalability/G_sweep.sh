#!/usr/bin/env bash
# Exp5-C: §5.6 Request length G_hat sweep.
# G_hat in {1024, 2048, 4096, 10240} tokens.
# Short requests: drift has less time to accumulate; less opportunity to amortize reconfig cost.
# Long requests: amplifies DACI's advantage.
# Default model: qwen3-14b (largest). Runs DACI + SDA for relative improvement.
set -euo pipefail

EXP_NAME="exp5_scalability/G_sweep"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=${N_TRACES:-5}
SEED_START=${SEED_START:-42}
MODEL_NAME=${MODEL_NAME:-qwen3-14b}
PARALLEL_JOBS=${PARALLEL_JOBS:-10}

cd "${PROJECT_ROOT}"

G_VALUES=(5000 10000 15000 20000 40000)
SCHEMES=(DACI SDA)

run_one() {
    local spec="$1"
    local G="${spec%%:*}"
    local scheme="${spec##*:}"
    local tag="G_${G}_${scheme}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "${tag}" \
        --schemes "${scheme}" \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --model_name "${MODEL_NAME}" \
        --G_hat "${G}" \
        --log_level summary_only \
        2>&1 | sed "s|^|[${tag}] |" | tee "${OUTPUT_DIR}/${tag}.log"
    echo "[done] ${tag}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START MODEL_NAME PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"

JOBS=()
for G in "${G_VALUES[@]}"; do
    for scheme in "${SCHEMES[@]}"; do
        JOBS+=("${G}:${scheme}")
    done
done
printf '%s\n' "${JOBS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo ""
echo "=== G_sweep complete. Model=${MODEL_NAME} Traces=${N_TRACES} ==="
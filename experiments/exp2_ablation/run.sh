#!/usr/bin/env bash
# Exp2: §5.3 Ablation Study
# Isolates contribution of each DACI component on the largest model (Qwen3-14B).
# 4 variants run in parallel:
#   full         : full DACI
#   no_freeze    : FM scheme (runtime DP jointly optimizes (a, b))
#   no_lazy      : DACI ablation — always commit b^{r,*}
#   no_predictor : DACI ablation — persistence predictor
# 4-panel output: TTLT, #Reconf, Ovhd, P99 TPOT
set -euo pipefail

EXP_NAME="exp2_ablation"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=${N_TRACES:-5}
SEED_START=${SEED_START:-42}
MODEL_NAME=${MODEL_NAME:-qwen3-14b}
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

# (label, scheme, ablation_mode)
# no_freeze uses the FM scheme (joint a/b optimization each window).
VARIANTS=(
    # "full:DACI:none"
    # "no_freeze:FM:none"
    # "no_lazy:DACI:no_lazy"
    "no_predictor:DACI:no_predictor"
)

run_one() {
    local spec="$1"
    local label="${spec%%:*}"
    local rest="${spec#*:}"
    local scheme="${rest%%:*}"
    local mode="${rest##*:}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "${label}" \
        --schemes "${scheme}" \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --model_name "${MODEL_NAME}" \
        --ablation "${mode}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/${label}.log" 2>&1
    echo "[done] ${label}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START MODEL_NAME PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${VARIANTS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

echo ""
echo "=== ${EXP_NAME} complete. Model=${MODEL_NAME} Traces=${N_TRACES} ==="
echo "Outputs: ${OUTPUT_DIR}"
echo "Aggregate with: python experiments/aggregate.py ablation ${OUTPUT_DIR}"
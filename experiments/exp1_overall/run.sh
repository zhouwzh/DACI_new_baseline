#!/usr/bin/env bash
# Exp1: §5.2 Overall Performance Comparison
# Runs all 5 schemes (SDA, RT, FM, DACI, OR) on default regime for all 3 models.
# Output: per-scheme TTLT, TTFT, P99 TPOT, Ovhd, #Reconf for Table 3.
# Results stored in separate folders: exp1_overall_small, _medium, _large.
set -euo pipefail

EXP_NAME="exp1_overall"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
N_TRACES=5
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-5}

cd "${PROJECT_ROOT}"

SCHEMES=(SDA RT FM DACI)

# model_name -> size suffix
declare -A MODEL_SUFFIX=(
    ["gemma3-4b"]="small"
    ["llama-3.2-8b"]="medium"
    ["qwen3-14b"]="large"
)
MODELS=(gemma3-4b llama-3.2-8b qwen3-14b)

run_one() {
    local scheme="$1"
    local model="$2"
    local out_dir="$3"
    python -u run.py \
        --config_dir configs \
        --output_dir "${out_dir}" \
        --run_id "${scheme}" \
        --schemes "${scheme}" \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime default \
        --model_name "${model}" \
        --log_level summary_only \
         2>&1 | sed "s/^/[${model}][${scheme}] /" | tee "${out_dir}/${scheme}.log"
    echo "[done] ${model}/${scheme}"
}

export -f run_one
export N_TRACES SEED_START PROJECT_ROOT

for model in "${MODELS[@]}"; do
    suffix="${MODEL_SUFFIX[$model]}"
    out_dir="${PROJECT_ROOT}/outputs/${EXP_NAME}_${suffix}"
    mkdir -p "${out_dir}"
    export CURRENT_MODEL="${model}"
    export CURRENT_OUT_DIR="${out_dir}"
    echo ""
    echo "=== Running model: ${model} (${suffix}) ==="
    printf '%s\n' "${SCHEMES[@]}" | \
        xargs -I{} -P "${PARALLEL_JOBS}" bash -c \
            'run_one "$@"' _ {} "${CURRENT_MODEL}" "${CURRENT_OUT_DIR}"
    echo "=== ${model} complete. Outputs: ${out_dir} ==="
    echo "Aggregate with: python experiments/aggregate.py ${out_dir}"
done

echo ""
echo "=== All models complete ==="

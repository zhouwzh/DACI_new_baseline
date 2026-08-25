#!/usr/bin/env bash
# Add supported new baselines to DACI paper §5.5 / Figure 5 output directories.
# This does not edit experiments/exp5_scalability/G_sweep.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="${PROJECT_ROOT}/gpt_new_baseline_code/run_new_baselines.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
N_TRACES="${N_TRACES:-30}"
SEED_START="${SEED_START:-42}"
LOG_LEVEL="${LOG_LEVEL:-summary_only}"
MODEL_NAME="${MODEL_NAME:-qwen3-14b}"
NEW_BASELINES="${NEW_BASELINES:-DynaPipe}"
G_VALUES=(5000 10000 15000 20000 40000)

cd "${PROJECT_ROOT}"
output_dir="${OUTPUT_ROOT}/exp5_scalability/G_sweep"
mkdir -p "${output_dir}"
for generated_tokens in "${G_VALUES[@]}"; do
  for baseline in ${NEW_BASELINES}; do
    run_id="G_${generated_tokens}_${baseline}"
    python "${RUNNER}" \
      --config_dir configs \
      --output_dir "${output_dir}" \
      --run_id "${run_id}" \
      --schemes "${baseline}" \
      --n_traces "${N_TRACES}" \
      --seed_start "${SEED_START}" \
      --regime default \
      --model_name "${MODEL_NAME}" \
      --G_hat "${generated_tokens}" \
      --log_level "${LOG_LEVEL}" \
      2>&1 | tee "${output_dir}/${run_id}.new_baseline.log"
  done
done

python gpt_new_baseline_code/scripts/aggregate_figure5_extension.py \
  --sweep-dir "${output_dir}" \
  --output-csv "${output_dir}/figure5_with_new_baselines.csv"

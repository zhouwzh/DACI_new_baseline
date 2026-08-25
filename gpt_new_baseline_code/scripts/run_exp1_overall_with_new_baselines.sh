#!/usr/bin/env bash
# Add supported new baselines to DACI paper §5.2 / Table 3 output directories.
# This does not edit experiments/exp1_overall/run.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="${PROJECT_ROOT}/gpt_new_baseline_code/run_new_baselines.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
N_TRACES="${N_TRACES:-30}"
SEED_START="${SEED_START:-42}"
LOG_LEVEL="${LOG_LEVEL:-summary_only}"
NEW_BASELINES="${NEW_BASELINES:-DynaPipe}"

# The current config uses llama-3-8b.  Do not copy the stale llama-3.2-8b key
# from experiments/exp1_overall/run.sh.
declare -A MODEL_SUFFIX=(
  ["gemma3-4b"]="small"
  ["llama-3-8b"]="medium"
  ["qwen3-14b"]="large"
)
MODELS=(gemma3-4b llama-3-8b qwen3-14b)

cd "${PROJECT_ROOT}"
for model in "${MODELS[@]}"; do
  output_dir="${OUTPUT_ROOT}/exp1_overall_${MODEL_SUFFIX[$model]}"
  mkdir -p "${output_dir}"
  for baseline in ${NEW_BASELINES}; do
    python "${RUNNER}" \
      --config_dir configs \
      --output_dir "${output_dir}" \
      --run_id "${baseline}" \
      --schemes "${baseline}" \
      --n_traces "${N_TRACES}" \
      --seed_start "${SEED_START}" \
      --regime default \
      --model_name "${model}" \
      --log_level "${LOG_LEVEL}" \
      2>&1 | tee "${output_dir}/${baseline}.new_baseline.log"
  done
done

python gpt_new_baseline_code/scripts/aggregate_table3_extension.py \
  --outputs-root "${OUTPUT_ROOT}" \
  --output-csv "${OUTPUT_ROOT}/exp1_overall_table3_with_new_baselines.csv"

#!/usr/bin/env bash
# Exp3: §5.4 Robustness Across Drift Regimes
# 4 regimes × 5 schemes. Plus one seed=42 "representative trace" per regime with full logs
# for the R2 per-window latency line plot.
set -euo pipefail

EXP_NAME="exp3_regime"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
N_TRACES=20
SEED_START=42
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

cd "${PROJECT_ROOT}"

REGIMES=(R1_calibrated R2_thermal R3_workload R4_network)
SCHEMES=(SDA RT FM DACI OR)

run_one() {
    local spec="$1"
    local regime="${spec%%:*}"
    local scheme="${spec##*:}"
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}" \
        --run_id "${regime}_${scheme}" \
        --schemes "${scheme}" \
        --n_traces "${N_TRACES}" \
        --seed_start "${SEED_START}" \
        --regime "${regime}" \
        --log_level summary_only \
        > "${OUTPUT_DIR}/${regime}_${scheme}.log" 2>&1
    echo "[done] ${regime} ${scheme}"
}

export -f run_one
export OUTPUT_DIR N_TRACES SEED_START PROJECT_ROOT

mkdir -p "${OUTPUT_DIR}"

# 20 jobs (4 regimes × 5 schemes)
JOBS=()
for regime in "${REGIMES[@]}"; do
    for scheme in "${SCHEMES[@]}"; do
        JOBS+=("${regime}:${scheme}")
    done
done
printf '%s\n' "${JOBS[@]}" | xargs -I{} -P "${PARALLEL_JOBS}" bash -c 'run_one "$@"' _ {}

# Bonus: full log for R2 representative trace (seed=42) for per-window latency line plot
echo ""
echo "=== Dumping R2 representative trace (seed=42) with full logs ==="
for scheme in "${SCHEMES[@]}"; do
    python run.py \
        --config_dir configs \
        --output_dir "${OUTPUT_DIR}/representative_R2" \
        --run_id "${scheme}" \
        --schemes "${scheme}" \
        --n_traces 1 \
        --seed_start 42 \
        --regime R2_thermal \
        --log_level full \
        > "${OUTPUT_DIR}/representative_R2/${scheme}.log" 2>&1
    echo "[done] representative R2 ${scheme}"
done

echo ""
echo "=== ${EXP_NAME} complete. Outputs: ${OUTPUT_DIR} ==="

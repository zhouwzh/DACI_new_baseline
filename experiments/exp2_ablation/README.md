# Exp2: Ablation Study (§5.3)

## Goal
4-panel bar chart: TTLT / #Reconf / Ovhd / P99 TPOT, each with 5 bars:
`Full DACI` (baseline, dashed line) vs 4 knock-outs.

## Variants
| Label | Ablation Mode | What it tests |
|---|---|---|
| `full` | none | Baseline DACI |
| `no_predictor` | persistence | Forward-looking drift prediction |
| `no_lazy` | always accept improving b | Acceptance rule vs wasted reconfigs |
| `no_bottleneck` | greedy additive DP | Handoff-bottleneck awareness |
| `no_adaptive_H` | H_r* = H_max fixed | Variance-aware horizon sizing |

## Run
```bash
bash run.sh                    # 5 variants × 20 seeds × ~60s each ≈ 25 min at P=4
```

## Output
```
outputs/exp2_ablation/
  full/ no_predictor/ no_lazy/ no_bottleneck/ no_adaptive_H/
    summary.csv  traces/  experiment_meta.json
```

## Expected pattern (§5.3 placeholder)
- `no_predictor` hurts on thermal-ramp tails
- `no_lazy` explodes #Reconf and Ovhd while barely improving nominal DP objective
- `no_bottleneck` → cheap-looking decoding but large one-shot handoff spikes
- `no_adaptive_H` hurts when predictor variance spikes

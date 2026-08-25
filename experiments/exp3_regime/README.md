# Exp3: Robustness Across Drift Regimes (§5.4)

## Goal
- **Figure (4-panel bar chart)**: one panel per regime (R1/R2/R3/R4), 5 bars each (SDA/RT/FM/DACI/OR), y-axis TTLT.
- **Line plot**: R2 seed=42, per-window decoding latency curves for all 5 schemes. Shows DACI tracking thermal ramp.
- **Table 4**: regime × scheme TTLT numbers (precise values backing the bar chart).

## Regimes
| Label | Active drift | Purpose |
|---|---|---|
| R1_calibrated | none | No-hurt sanity test |
| R2_thermal | thermal only (nu×2, theta_th=50) | Classic throttling |
| R3_workload | workload only | Drift least predictable by persistence |
| R4_network | network only | Exogenous drift, boundary can't help |

## Run
```bash
bash run.sh    # 20 combos × 20 seeds + 5 representative R2 traces
```

## Output
```
outputs/exp3_regime/
  {regime}_{scheme}/
    summary.csv
    traces/
  representative_R2/
    {scheme}/
      traces/{scheme}_seed42.jsonl           # per-window latency
      traces/{scheme}_seed42_devices.jsonl   # per-second theta
```

## Expected pattern (§5.4)
- R1: DACI ≈ SDA (no-hurt property)
- R2, R3: DACI recovers most of SDA→OR gap; largest gains in R3
- R4: DACI ≈ SDA (boundary can't route around link drift)
- RT underperforms DACI in R3/R4 (trigger on transient, pay full migration)

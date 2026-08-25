# Exp1: Overall Performance Comparison (§5.2)

## Goal
Table 3 numbers: mean TTLT / TTFT / P99 TPOT / Ovhd per scheme under default regime.

## What it does
- 5 schemes: SDA, RT, FM, DACI, OR
- 20 traces (seed 42..61)
- Default regime (thermal + workload + network all active)
- Gemma-7B default model, cluster (1 AGX, 3 NX, 4 Nano)

## Run
```bash
bash run.sh                    # uses PARALLEL_JOBS=4
PARALLEL_JOBS=8 bash run.sh    # more parallelism
```

## Output
```
outputs/exp1_overall/
  SDA/  RT/  FM/  DACI/  OR/       # one subdir per scheme
    config_snapshot.json
    experiment_meta.json
    summary.csv                    # per-scheme aggregates
    traces/
      {scheme}_seed{N}.jsonl       # per-window log (lazy/accept decisions)
  {scheme}.log                     # stdout
```

## Key metrics extracted
- TTLT mean ± std (from summary.csv)
- TTFT mean
- P99 TPOT mean
- Ovhd mean, #Reconf mean

## Expected pattern
- OR < DACI < FM < RT < SDA (TTLT, lower better)
- OR Ovhd likely ≥ DACI (oracle over-switches when GT says it's worth it)
- SDA #Reconf = 0

# Exp5: Scalability (§5.6)

## Goal
Three line plots showing DACI scales with cluster, model, request.

### 5-A: Cluster size N ∈ {4, 6, 8, 10, 12}
```bash
bash N_sweep.sh
```
Plot: TTLT vs N + initial DP wall-clock time.
Expected: mild TTLT improvement up to N≈8, sub-linear DP cost growth thanks to top-K pruning.

### 5-B: Model depth L (three models)
```bash
bash L_sweep.sh
```
Plot: runtime DP cost per window vs L ∈ {28, 40, 64}.
Expected: DP cost < 1ms even at L=64.

### 5-C: Request length G_hat ∈ {256, 1024, 2048, 4096}
```bash
bash G_sweep.sh
```
Plot: relative TTLT improvement (DACI vs SDA) % vs G_hat.
Expected: improvement grows with G_hat; at G_hat=256 DACI collapses to SDA (lazy rule refuses).

## Run all in parallel
```bash
bash N_sweep.sh & bash L_sweep.sh & bash G_sweep.sh & wait
```

## Output
```
outputs/exp5_scalability/
  N_sweep/{N_4, N_6, ...}/summary.csv
  L_sweep/{gemma3-4b, llama-3.2-8b, qwen3-14b}/summary.csv
  G_sweep/{G_256_DACI, G_256_SDA, ...}/summary.csv
```

## Note
For L_sweep, LLaMA-13B and Qwen-32B weights don't fit in AGX (32GB) alone so
cross-node partitioning is mandatory. Memory feasibility constraint in DP will
drive placement decisions.
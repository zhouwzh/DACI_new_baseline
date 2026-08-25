# Table 3, regenerated -- paper vs BEFORE vs AFTER vs AFTERC

30 seeds (42-71) per cell. Every figure computed from the traces in one pass (S6a.4).

| arm | what it isolates |
|---|---|
| BEFORE | pristine simulator, alpha=2 ms beta=2 ns/B |
| AFTER | + fixes (a) H_swap, (c) permutation search, (d) memory & dead keys |
| AFTERC | + MEASURED link constants, alpha=487 us beta=9.126 ns/B |

Fix (b) is absent by design: it edited Cluster.baseline_link, which nothing reads, so it changed no result. AFTERC corrects the live constants in drift.json instead.

## qwen3-14b

| scheme | paper | BEFORE | AFTER | AFTERC | Ovhd B/A/C | #Rec B/A/C | a-chg% B/A/C |
|---|---|---|---|---|---|---|---|
| SDA | 425.16 | 407.1+-55.7 | 416.9+-69.9 | 324.1+-58.7 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.000/0.000/0.000 |
| RT | 488.17 | 504.8+-49.4 | 395.0+-50.5 | 293.1+-33.8 | 94.94/31.38/23.41 | 10.43/3.77/3.37 | 1.391/0.373/0.236 |
| FM | 416.5 | 396.0+-42.5 | 376.1+-31.5 | 282.4+-18.6 | 19.35/14.29/8.06 | 2.87/2.50/2.20 | 0.276/0.164/0.058 |
| DACI | 371.89 | 370.5+-44.8 | 372.1+-45.1 | 287.1+-35.0 | 6.18/5.95/6.05 | 2.43/2.37/2.40 | 0.000/0.000/0.000 |

**DACI's TTLT lead, regenerated:**

| vs | BEFORE | AFTER | AFTERC |
|---|---|---|---|
| SDA | +8.99% | +10.74% | **+11.42%** |
| RT | +26.62% | +5.79% | **+2.05%** |
| FM | +6.44% | +1.05% | **-1.67%** |

**Verdict: (iii) lead small or negative -- vs FM -1.67%, vs RT +2.05%. Drop the large models; report the calibrated 1B/3B points instead.**

**S5.2 prose vs regenerated (S0g):**

| quantity | prose | Table 3 printed | AFTERC |
|---|---|---|---|
| DACI lead vs FM | 16.0% | 10.7% | **-1.67%** |
| DACI lead vs RT | 25.0% | 23.8% | **2.05%** |
| RT overhead | 85.3 s | 90.29 s | 23.41 s |

## gemma3-4b

_AFTERC still running -- BEFORE/AFTER only._

| scheme | paper | BEFORE | AFTER | AFTERC | Ovhd B/A/C | #Rec B/A/C | a-chg% B/A/C |
|---|---|---|---|---|---|---|---|
| SDA | -- | 95.4+-12.6 | 95.4+-12.6 | -- | 0.00/0.00/-- | 0.00/0.00/-- | 0.000/0.000/-- |
| RT | -- | 113.3+-45.6 | 110.0+-8.3 | -- | 9.44/22.29/-- | 2.43/6.13/-- | 0.324/0.760/-- |
| FM | -- | 117.1+-43.9 | 100.3+-5.0 | -- | 5.11/12.22/-- | 1.27/3.33/-- | 0.169/0.444/-- |
| DACI | -- | 95.9+-11.5 | 95.9+-11.5 | -- | 0.21/0.21/-- | 0.27/0.27/-- | 0.000/0.000/-- |

**DACI's TTLT lead, regenerated:**

| vs | BEFORE | AFTER | AFTERC |
|---|---|---|---|
| SDA | -0.54% | -0.54% | -- |
| RT | +15.38% | +12.78% | -- |
| FM | +18.10% | +4.40% | -- |

## llama-3-8b

| scheme | paper | BEFORE | AFTER | AFTERC | Ovhd B/A/C | #Rec B/A/C | a-chg% B/A/C |
|---|---|---|---|---|---|---|---|
| SDA | -- | 178.1+-44.7 | 178.1+-44.7 | 136.0+-29.5 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0.000/0.000/0.000 |
| RT | -- | 167.8+-38.1 | 149.5+-25.2 | 114.4+-13.7 | 15.63/13.71/6.85 | 2.93/4.40/3.60 | 0.391/0.324/0.093 |
| FM | -- | 147.9+-36.8 | 144.1+-17.5 | 116.0+-13.4 | 6.06/7.48/5.30 | 1.57/1.53/1.90 | 0.151/0.169/0.062 |
| DACI | -- | 148.3+-32.9 | 148.3+-32.9 | 120.6+-23.4 | 3.56/3.56/3.97 | 2.43/2.43/2.00 | 0.000/0.000/0.000 |

**DACI's TTLT lead, regenerated:**

| vs | BEFORE | AFTER | AFTERC |
|---|---|---|---|
| SDA | +16.75% | +16.75% | **+11.32%** |
| RT | +11.62% | +0.78% | **-5.45%** |
| FM | -0.27% | -2.89% | **-3.99%** |

**Verdict: (iii) lead small or negative -- vs FM -3.99%, vs RT -5.45%. Drop the large models; report the calibrated 1B/3B points instead.**


wrote results/m5a_fixes/table3_regenerated.csv (12 rows)
wrote results/m5a_fixes/table3_regenerated.tex (8 AFTERC rows)

## Counter availability (S5b.4)

| counter | available | why |
|---|---|---|
| #reconfigs | yes | summary.csv |
| acceptance rate | yes | per-window `accepted` |
| **a-changed rate** | **yes** | per-window `a`; the "has the baseline gone inert" test |
| pool size | no | control/mechanism.py post-dates these runs |
| placements enumerated | no | same |

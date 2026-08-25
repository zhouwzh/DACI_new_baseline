# Four correctness fixes, and what they did to Table 3

Regenerated over the same 30 seeds (42–71), same config, before and after the
fixes in the parent commit. `outputs/` is gitignored, so the two `summary.csv`
files are copied here verbatim.

Reproduce:

```bash
python run.py --n_traces 30 --schemes SDA,RT,FM,DACI --run_id AFTER30
git stash && python run.py --n_traces 30 --schemes SDA,RT,FM,DACI --run_id BEFORE30
```

## The fixes

**(a) `H_swap_s` never reached `Node`.** `build_cluster()` read every other
tier field and silently dropped this one, so every node took the dataclass
default of `0.0` while `devices.json` specified 2.5/3/4 s per tier. Placement
changes were therefore **free** — the single most consequential of the four,
because the whole DACI argument rests on a cost asymmetry between boundary
shifts and placement changes, and one side of that asymmetry was zero.

**(b) `beta_s_per_byte = 1e-9`** is 1 GB/s = **8 Gbps**, described throughout
as GbE-class. 1 GbE is `8e-9`. Both link constants now live in
`configs/devices.json` under `link` rather than being hardcoded in
`cluster.py`.

**(c) RT and FM searched only idle nodes.** Both built `in_stage = set(st.a)`
and skipped any node already hosting a stage, so neither baseline could ever
consider a *swap* — only a move onto a spare. Replaced with a permutation
search over all N nodes, with per-stage feasibility evaluated *inside* the
enumeration (a one-shot filter keyed on the largest stage empties the pool
below S and degenerates back to "never move").

**(d) Dead config keys and a wrong memory figure.** Removed
`devices.effective_utilization`, `algo.dp.u_discretization`,
`algo.dp.u_grid_n_bins`, `algo.latency_model`, `algo.weight_residency` — all
read by nothing. Orin Nano `m_gb` 12.0 → 8.0, which is the part it actually
ships with.

## Result — qwen3-14b, N=8 (1 high / 3 mid / 4 low), W=20, Ĝ=15000, 30 seeds

| scheme | TTLT before (s) | TTLT after (s) | Δ | Ovhd before | Ovhd after | #Rec before | #Rec after |
|---|---|---|---|---|---|---|---|
| SDA  | 407.1 ± 55.7 | 416.9 ± 69.9 |  +9.8  |  0.00 |  0.00 |  0.00 | 0.00 |
| RT   | 504.8 ± 49.4 | 395.0 ± 50.5 | −109.8 | 94.94 | 31.38 | 10.43 | 3.77 |
| FM   | 396.0 ± 42.5 | 376.1 ± 31.5 |  −19.9 | 19.35 | 14.29 |  2.87 | 2.50 |
| DACI | 370.5 ± 44.8 | 372.1 ± 45.1 |  +1.6  |  6.18 |  5.95 |  2.43 | 2.37 |

DACI's advantage:

| | vs SDA | vs RT | vs FM |
|---|---|---|---|
| before | +8.99% | +26.62% | +6.44% |
| after  | +10.74% | +5.79% | +1.05% |

## Reading it

**The baselines got much stronger and DACI did not move.** DACI's own numbers
are essentially unchanged (+1.6 s, well inside a 45 s std) — it was already
paying `H_swap` implicitly by rarely changing placement, so making the charge
explicit cost it nothing. RT dropped 110 s and FM 20 s.

Mechanically, fix (a) is what did it: with swaps free, RT reconfigured 10.4
times per trace and burned 95 s of overhead chasing them. Charging 2.5–4 s per
swap makes most of those unprofitable, so RT does 3.8 and burns 31 s. The
baseline is not faster because it got smarter; it is faster because it stopped
doing something that was never free in the first place.

**The headline number moves.** Against FM — the strongest baseline — DACI's
lead goes from 6.4% to **1.05%**, which is inside the noise at n=30. Against
RT it goes from 26.6% to 5.8%. Only the SDA comparison survives intact, and
SDA is the static baseline that exists to show adaptation helps at all.

This is a finding, not a regression. The earlier margins were partly an
artifact of baselines that could not swap and were not charged when they did.
Any claim of the form "DACI beats reactive/frequency-matched policies by
double digits" is not supported by the corrected simulator on this
configuration, and the paper's §5.2 prose needs to be rewritten from
regenerated numbers rather than adjusted.

The one defensible claim that gets *stronger*: DACI achieves near-FM latency
at **2.4× less reconfiguration overhead** (5.95 s vs 14.29 s) and with the
tightest overhead variance of any adaptive scheme. That is a stability
argument, not a raw-latency argument, and it is the one the corrected data
supports.

## Caveat

Only `qwen3-14b` has been regenerated at 30 traces so far; `gemma3-4b` and
`llama-3.2-8b` are running and land in a follow-up commit. The three models
differ enough in memory pressure that fix (d) — the 8 GB Nano — could plausibly
bite harder on the larger one.

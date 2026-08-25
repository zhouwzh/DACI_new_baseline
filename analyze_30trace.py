"""Regenerate Table 3 and every derived figure, from the traces, in one pass.

§6a.4 / §0g: the current draft's §5.2 prose and Table 3 were generated from
different runs and disagree — the text claims DACI leads FM by 16% where the
table's own numbers give 10.7%. That happened because derived percentages were
hand-transcribed. Everything this script prints is computed from the same
traces in the same pass, so prose and table cannot drift apart again.

Three arms, one new set of runs, so each correction is attributable:

  BEFORE  pristine simulator            α = 2 ms,   β = 2 ns/B
  AFTER   + fixes (a) H_swap, (c) permutation search, (d) memory & dead keys
  AFTERC  + measured link constants     α = 487 µs, β = 9.126 ns/B

Fix (b) is deliberately absent from that list: it edited
``Cluster.baseline_link``, which is read nowhere, so it changed nothing. The
live constants are ``drift.json``'s ``network`` block, and correcting those is
what AFTERC isolates.

Mechanism counters (§5b.4): the per-window traces carry ``b``, ``a`` and
``accepted``, so the ``a``-changed rate, acceptance rate and #reconfigs are all
recoverable. Pool size and permutations-enumerated are NOT — ``mechanism.py``
post-dates these runs — and are reported as unavailable rather than silently
omitted. The ``a``-changed rate is the one that answers "have RT and FM gone
inert again", which is what §5b.4 exists to detect.

Usage:
    PYTHONIOENCODING=utf-8 python analyze_30trace.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional

SCHEMES = ["SDA", "RT", "FM", "DACI"]

# (model, BEFORE id, AFTER id, AFTERC id). Written out rather than derived,
# because the 8B model was renamed mid-stream: its BEFORE/AFTER runs carry the
# old, wrong `llama-3.2-8b` key and only AFTERC uses the corrected
# `llama-3-8b`. Deriving ids from the model name would silently miss them.
MODELS = [
    ("qwen3-14b",  "BEFORE30",              "AFTER30",              "AFTERC30_qwen3-14b"),
    ("gemma3-4b",  "BEFORE30_gemma3-4b",    "AFTER30_gemma3-4b",    "AFTERC30_gemma3-4b"),
    ("llama-3-8b", "BEFORE30_llama-3.2-8b", "AFTER30_llama-3.2-8b", "AFTERC30_llama-3-8b"),
]

# Paper Table 3 as printed, Qwen3-14B row (plan §0g). The other model rows are
# not quoted in the plan, so they are absent rather than guessed.
PAPER_TABLE3 = {
    "qwen3-14b": {
        "SDA":  {"TTLT": 425.16},
        "RT":   {"TTLT": 488.17, "Ovhd": 90.29},
        "FM":   {"TTLT": 416.50},
        "DACI": {"TTLT": 371.89, "Ovhd": 7.69},
    },
}
PAPER_PROSE = {
    "qwen3-14b": {"lead_vs_FM_pct": 16.0, "lead_vs_RT_pct": 25.0,
                  "RT_Ovhd": 85.3, "DACI_Ovhd": 8.2},
}


def _summary(root: str, run_id: str) -> Optional[Dict[str, Dict[str, float]]]:
    p = os.path.join(root, "outputs", run_id, "summary.csv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["scheme"]] = {k: float(v) for k, v in row.items() if k != "scheme"}
    return out if len(out) == len(SCHEMES) else None


def _mechanism(root: str, run_id: str) -> Dict[str, Dict[str, float]]:
    """a-changed and acceptance rates, from the per-window traces."""
    out: Dict[str, Dict[str, float]] = {}
    tdir = os.path.join(root, "outputs", run_id, "traces")
    for scheme in SCHEMES:
        a_changed = accepted = windows = 0
        traces = sorted(glob.glob(os.path.join(tdir, f"{scheme}_seed*.jsonl")))
        for f in traces:
            prev_a = None
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if "header" in rec:
                        continue
                    windows += 1
                    if rec.get("accepted"):
                        accepted += 1
                    a = rec.get("a")
                    if a is not None:
                        if prev_a is not None and list(a) != list(prev_a):
                            a_changed += 1
                        prev_a = a
        if windows:
            out[scheme] = {
                "windows": windows,
                "accept_rate_pct": 100.0 * accepted / windows,
                "a_changed_rate_pct": 100.0 * a_changed / windows,
                "a_changes_total": a_changed,
                "n_traces": len(traces),
            }
    return out


def _lead(d, other: str) -> float:
    """DACI's TTLT advantage over `other`, in percent."""
    return 100.0 * (d[other]["TTLT_mean_s"] - d["DACI"]["TTLT_mean_s"]) / d[other]["TTLT_mean_s"]


def _verdict(d) -> str:
    """The (i)/(ii)/(iii) call, computed rather than eyeballed."""
    vs_fm, vs_rt = _lead(d, "FM"), _lead(d, "RT")
    worst = min(vs_fm, vs_rt)
    if worst < 1.0:
        return (f"(iii) lead small or negative -- vs FM {vs_fm:+.2f}%, vs RT "
                f"{vs_rt:+.2f}%. Drop the large models; report the calibrated "
                f"1B/3B points instead.")
    if worst < 5.0:
        return f"(ii) lead positive but thin -- vs FM {vs_fm:+.2f}%, vs RT {vs_rt:+.2f}%"
    return f"(i) lead healthy -- vs FM {vs_fm:+.2f}%, vs RT {vs_rt:+.2f}%"


def _f(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None else "--"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-root", default="../daci-sim-orig")
    ap.add_argument("--after-root", default=".")
    ap.add_argument("--csv", default="results/m5a_fixes/table3_regenerated.csv")
    ap.add_argument("--tex", default="results/m5a_fixes/table3_regenerated.tex")
    args = ap.parse_args()

    rows: List[Dict] = []
    print("# Table 3, regenerated -- paper vs BEFORE vs AFTER vs AFTERC\n")
    print("30 seeds (42-71) per cell. Every figure computed from the traces in "
          "one pass (S6a.4).\n")
    print("| arm | what it isolates |")
    print("|---|---|")
    print("| BEFORE | pristine simulator, alpha=2 ms beta=2 ns/B |")
    print("| AFTER | + fixes (a) H_swap, (c) permutation search, (d) memory & dead keys |")
    print("| AFTERC | + MEASURED link constants, alpha=487 us beta=9.126 ns/B |")
    print()
    print("Fix (b) is absent by design: it edited Cluster.baseline_link, which "
          "nothing reads, so it changed no result. AFTERC corrects the live "
          "constants in drift.json instead.\n")

    for model, rid_b, rid_a, rid_c in MODELS:
        before = _summary(args.before_root, rid_b)
        after = _summary(args.after_root, rid_a)
        afterc = _summary(args.after_root, rid_c)
        if before is None or after is None:
            print(f"## {model}\n\n_(BEFORE or AFTER missing)_\n")
            continue
        mech_b = _mechanism(args.before_root, rid_b)
        mech_a = _mechanism(args.after_root, rid_a)
        mech_c = _mechanism(args.after_root, rid_c) if afterc else {}
        paper = PAPER_TABLE3.get(model, {})

        print(f"## {model}\n")
        if afterc is None:
            print("_AFTERC still running -- BEFORE/AFTER only._\n")
        print("| scheme | paper | BEFORE | AFTER | AFTERC | Ovhd B/A/C | "
              "#Rec B/A/C | a-chg% B/A/C |")
        print("|---|---|---|---|---|---|---|---|")
        for s in SCHEMES:
            b, a = before[s], after[s]
            c = afterc[s] if afterc else None
            mb, ma, mc = mech_b.get(s, {}), mech_a.get(s, {}), mech_c.get(s, {})
            pv = paper.get(s, {}).get("TTLT")
            print(f"| {s} | {pv if pv is not None else '--'} "
                  f"| {b['TTLT_mean_s']:.1f}+-{b['TTLT_std_s']:.1f} "
                  f"| {a['TTLT_mean_s']:.1f}+-{a['TTLT_std_s']:.1f} "
                  f"| {_f(c['TTLT_mean_s'] if c else None, '{:.1f}')}"
                  f"{('+-%.1f' % c['TTLT_std_s']) if c else ''} "
                  f"| {b['Ovhd_mean_s']:.2f}/{a['Ovhd_mean_s']:.2f}/"
                  f"{_f(c['Ovhd_mean_s'] if c else None)} "
                  f"| {b['Nreconf_mean']:.2f}/{a['Nreconf_mean']:.2f}/"
                  f"{_f(c['Nreconf_mean'] if c else None)} "
                  f"| {mb.get('a_changed_rate_pct', 0):.3f}/"
                  f"{ma.get('a_changed_rate_pct', 0):.3f}/"
                  f"{_f(mc.get('a_changed_rate_pct'), '{:.3f}')} |")
            rows.append({
                "model": model, "scheme": s, "paper_TTLT": pv,
                "before_TTLT": b["TTLT_mean_s"], "after_TTLT": a["TTLT_mean_s"],
                "afterc_TTLT": c["TTLT_mean_s"] if c else None,
                "afterc_TTLT_std": c["TTLT_std_s"] if c else None,
                "afterc_P99TPOT": c["P99_TPOT_mean_ms"] if c else None,
                "before_Ovhd": b["Ovhd_mean_s"], "after_Ovhd": a["Ovhd_mean_s"],
                "afterc_Ovhd": c["Ovhd_mean_s"] if c else None,
                "before_Nrec": b["Nreconf_mean"], "after_Nrec": a["Nreconf_mean"],
                "afterc_Nrec": c["Nreconf_mean"] if c else None,
                "before_a_changed_pct": mb.get("a_changed_rate_pct"),
                "after_a_changed_pct": ma.get("a_changed_rate_pct"),
                "afterc_a_changed_pct": mc.get("a_changed_rate_pct"),
            })

        print()
        print("**DACI's TTLT lead, regenerated:**\n")
        print("| vs | BEFORE | AFTER | AFTERC |")
        print("|---|---|---|---|")
        for other in ("SDA", "RT", "FM"):
            cl = f"**{_lead(afterc, other):+.2f}%**" if afterc else "--"
            print(f"| {other} | {_lead(before, other):+.2f}% "
                  f"| {_lead(after, other):+.2f}% | {cl} |")
        print()
        if afterc:
            print(f"**Verdict: {_verdict(afterc)}**\n")

        prose = PAPER_PROSE.get(model)
        if prose and afterc:
            print("**S5.2 prose vs regenerated (S0g):**\n")
            print("| quantity | prose | Table 3 printed | AFTERC |")
            print("|---|---|---|---|")
            fm_p = 100 * (paper["FM"]["TTLT"] - paper["DACI"]["TTLT"]) / paper["FM"]["TTLT"]
            rt_p = 100 * (paper["RT"]["TTLT"] - paper["DACI"]["TTLT"]) / paper["RT"]["TTLT"]
            print(f"| DACI lead vs FM | {prose['lead_vs_FM_pct']:.1f}% | {fm_p:.1f}% "
                  f"| **{_lead(afterc, 'FM'):.2f}%** |")
            print(f"| DACI lead vs RT | {prose['lead_vs_RT_pct']:.1f}% | {rt_p:.1f}% "
                  f"| **{_lead(afterc, 'RT'):.2f}%** |")
            print(f"| RT overhead | {prose['RT_Ovhd']} s | {paper['RT']['Ovhd']} s "
                  f"| {afterc['RT']['Ovhd_mean_s']:.2f} s |")
            print()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_tex = 0
    with open(args.tex, "w", encoding="utf-8") as fh:
        fh.write("% Generated by analyze_30trace.py -- do not edit by hand (S0g).\n")
        fh.write("% AFTERC arm only: fixed simulator with MEASURED link constants.\n")
        fh.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        fh.write("Model & Scheme & TTLT (s) & P99 TPOT (ms) & Ovhd (s) & "
                 "\\#Rec & $a$-chg (\\%) \\\\\n\\midrule\n")
        for r in rows:
            if r["afterc_TTLT"] is None:
                continue          # only the corrected arm belongs in the paper
            n_tex += 1
            fh.write(f"{r['model']} & {r['scheme']} & "
                     f"{r['afterc_TTLT']:.2f} $\\pm$ {r['afterc_TTLT_std']:.2f} & "
                     f"{r['afterc_P99TPOT']:.1f} & {r['afterc_Ovhd']:.2f} & "
                     f"{r['afterc_Nrec']:.2f} & "
                     f"{(r['afterc_a_changed_pct'] or 0):.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote {args.csv} ({len(rows)} rows)")
    print(f"wrote {args.tex} ({n_tex} AFTERC rows)")

    print("\n## Counter availability (S5b.4)\n")
    print("| counter | available | why |")
    print("|---|---|---|")
    print("| #reconfigs | yes | summary.csv |")
    print("| acceptance rate | yes | per-window `accepted` |")
    print("| **a-changed rate** | **yes** | per-window `a`; the "
          "\"has the baseline gone inert\" test |")
    print("| pool size | no | control/mechanism.py post-dates these runs |")
    print("| placements enumerated | no | same |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

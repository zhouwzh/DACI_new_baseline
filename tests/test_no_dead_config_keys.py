"""Every config key must be read by something. (Item 7.)

Six dead knobs have been found by hand so far:

    algo.latency_model.mode          algo.dp.u_discretization
    algo.dp.u_grid_n_bins            algo.weight_residency.mode
    devices.effective_utilization    devices.link  (Cluster.baseline_link)

That is systemic rather than bad luck, and the last one was expensive: it was
"fixed" during M5a, the fix changed no result because nothing read it, and the
error then survived a full 30-trace regeneration *and* a hardware comparison
against the wrong constant. A dead knob is worse than a missing one, because it
looks authoritative.

This walks every leaf key in ``configs/*.json`` and fails if the name appears
nowhere in ``src/``. It would have caught all six.

Deliberately a *name* search rather than a taint analysis: configs are read as
plain dicts with string subscripts, so the key name is what appears in the
source. That makes false negatives possible (a key read only via a computed
string) but keeps false positives rare, and the failure mode of a name search
is the safe one -- it under-reports rather than blocking on a key that is
genuinely used.

Usage:
    python tests/test_no_dead_config_keys.py
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Containers whose CHILDREN are selected by value at runtime, not by literal
# name -- a model name, a tier name, an active-regime name. Their children are
# exempt; the container itself is not.
DYNAMIC_CONTAINERS = {
    "models",            # models.json: cfg["models"][model_name]
    "tiers",             # devices.json: tiers[node.tier]
    "per_tier",          # drift.json: per_tier[node.tier]
    "regime",            # drift.json: regime[regime["active"]]
    "cluster",           # experiment.json: mix.get(tier_name)
    "table_ms",          # experiment.json: table_ms[f"N{N}_S{S}"]
}

# Keys read outside src/ but legitimately part of the contract.
ALLOW = {"schemes"}


def leaf_keys(obj, prefix=""):
    """Yield (dotted_path, key_name) for every key in a nested dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            yield path, k
            yield from leaf_keys(v, path)
    elif isinstance(obj, list):
        for item in obj:
            yield from leaf_keys(item, prefix)


def main() -> int:
    src = []
    for p in glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True):
        if "__pycache__" in p:
            continue
        src.append(io.open(p, encoding="utf-8").read())
    # run.py drives the experiment and legitimately reads config too.
    for extra in ("run.py",):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            src.append(io.open(p, encoding="utf-8").read())
    blob = "\n".join(src)

    dead = []
    checked = 0
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "configs", "*.json"))):
        cfg = json.load(io.open(cfg_path, encoding="utf-8"))
        name = os.path.basename(cfg_path)
        seen = set()
        for path, key in leaf_keys(cfg):
            segs = path.split(".")
            # Anything under an underscore-prefixed ancestor is documentation,
            # not configuration. `_measured.method` is a note about how a value
            # was obtained; nothing should read it.
            if any(x.startswith("_") for x in segs):
                continue
            # Children of a dynamic container are looked up by a computed
            # string, so their names never appear literally in the source.
            if len(segs) >= 2 and segs[-2] in DYNAMIC_CONTAINERS:
                continue
            if any(x in DYNAMIC_CONTAINERS for x in segs[:-1]):
                continue
            if key in ALLOW or key in seen:
                continue
            seen.add(key)
            checked += 1
            # The key name as it would appear in a subscript or a .get().
            if re.search(r"""["']%s["']""" % re.escape(key), blob):
                continue
            dead.append((name, path))

    print(f"checked {checked} distinct config keys across "
          f"{len(glob.glob(os.path.join(ROOT, 'configs', '*.json')))} files")
    if dead:
        print(f"\nFAIL - {len(dead)} config key(s) are never read in src/:\n")
        for f, path in dead:
            print(f"  {f}: {path}")
        print("\nA key nothing reads is worse than a missing one: it looks")
        print("authoritative. Either wire it up, delete it, or prefix it with")
        print("'_' to mark it as documentation.")
        return 1
    print("PASS - every config key is read somewhere in src/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pool two training draws of each recipe into one eight-member ensemble.

EXPLORATORY. Each recipe was trained on two seeds and each draw scored on its
own: A gave -6.1 % and A2 -2.6 % against their control arms, Pavo -7.1 % and
Pavo 2 -6.6 %. Pooling the two draws of a recipe into one ensemble, and the
two control arms (control2000 and F) into another, puts both seeds on both
sides of a single comparison. That is the fairest one test of a recipe, and
the extra members also show how much four members hold every model back.

    scripts/pool_seeds.py \\
        --cases ~/bris-runs/calibration-test/with-baseline-cases.npz \\
        --cases ~/bris-runs/a-vs-pavo/cases.npz \\
        --cases ~/bris-runs/a-replication/testyear-cases.npz \\
        --cases ~/bris-runs/c2-replication/testyear-cases.npz \\
        --out ~/bris-runs/pooled-seeds/pooled-cases.npz

The output is a cache in the layout scripts/hits_and_calibration.py writes,
so it is scored by that script and by scripts/reliability.py unchanged:

    control8  control2000 + control2000s2 (F)     seeds 20260909 + 20261003
    tailA8    tailA2000 + tailA2000s2              the same two seeds
    tailC8    tailC2000 + tailC2000s2              the same two seeds

The six four-member models are kept beside them under their own names.
Other pairs are given as --pool LABEL=first+second (repeatable), which
replaces the three above.
scripts/_cases.py refuses unless every cache holds the same station-days.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("numpy")

import numpy as np  # noqa: E402

import _cases  # noqa: E402

POOLS = {
    "control8": ("control2000", "control2000s2"),
    "tailA8": ("tailA2000", "tailA2000s2"),
    "tailC8": ("tailC2000", "tailC2000s2"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", action="append", required=True, type=Path)
    ap.add_argument("--pool", action="append", default=[], metavar="LABEL=FIRST+SECOND")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pools = dict(POOLS)
    if args.pool:
        pools = {}
        for item in args.pool:
            label, members = item.split("=", 1)
            pools[label] = tuple(members.split("+"))

    kinds = set()
    for path in args.cases:
        with np.load(path.expanduser(), allow_pickle=False) as data:
            kinds.add(str(data["interpolation"]) if "interpolation" in data.files else "nearest")
    if len(kinds) != 1:
        raise SystemExit(f"the caches mix interpolations {sorted(kinds)}; pool only like with like")

    singles = list(dict.fromkeys(m for pair in pools.values() for m in pair))
    c = _cases.load(args.cases, singles)
    lead = _cases.LEAD
    out = {f"obs_{lead}": c["obs"], f"dates_{lead}": c["dates"], f"station_{lead}": c["station"],
           "stations": c["stations"], "interpolation": np.array(kinds.pop())}
    if c["meps"] is not None:
        out[f"meps_{lead}"] = c["meps"]
    for label in singles:
        out[f"model_{label}_{lead}"] = c["models"][label]
    for label, members in pools.items():
        out[f"model_{label}_{lead}"] = np.concatenate([c["models"][m] for m in members], axis=1)
        print(f"{label}: {' + '.join(members)} -> {out[f'model_{label}_{lead}'].shape[1]} members")

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.expanduser(), **out)
    print(f"{c['obs'].size:,} station-days; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

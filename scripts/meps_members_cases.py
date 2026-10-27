#!/usr/bin/env python3
"""Add MEPS ensemble members to an existing scoring cache, case by case.

    scripts/meps_members_cases.py --cache ~/bris-runs/operational-calibration/testyear-pooled.npz \\
        --members ~/bris-runs/meps-ens --tag "" \\
        --set meps4=0,1,2,9 --set meps15=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \\
        --out ~/bris-runs/meps-ens/testyear-cases.npz

Reads the per-member files of scripts/fetch_meps_members.py with
evaluate_experiment.MepsReader, the reader the single MEPS run is scored
through, and sums the same 06-06 UTC day ending at +30 h from 00 UTC the day
before. Each set becomes one ensemble, model_<name>_30, beside the models
already in the cache. Only cases where every member of every set is present
are kept, and every per-case array is cut to them, so all models and all
sets are scored on one footing. The cut is reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402

import evaluate_experiment as ev  # noqa: E402

LEAD = 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", action="append", required=True, metavar="NAME=M,M,...")
    ap.add_argument("--max-distance-km", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.cache.expanduser(), allow_pickle=False) as data:
        cases = {k: data[k] for k in data.files}
    dates = cases[f"dates_{LEAD}"].astype("datetime64[D]")
    station = cases[f"station_{LEAD}"]
    names = cases["stations"]
    cycle_day = dates - np.timedelta64(1, "D")
    n = dates.size

    sets = {}
    for item in args.set:
        name, members = item.split("=", 1)
        sets[name] = [int(m) for m in members.split(",")]
    wanted = sorted({m for ms in sets.values() for m in ms})

    values = {}
    for m in wanted:
        path = args.members.expanduser() / f"mbr{m:03d}{args.tag}.npz"
        reader = ev.MepsReader(path, {"stations": names}, args.max_distance_km)
        out = np.full(n, np.nan)
        for day in np.unique(cycle_day):
            key = ev.cycle_key(np.datetime64(f"{day}T00:00:00"))
            if not reader.has_cycle(key):
                continue
            row = reader.read(key, [LEAD])[0]
            sel = cycle_day == day
            out[sel] = row[station[sel]]
        values[m] = out
        print(f"  member {m:2d}: {np.isfinite(out).mean():.1%} of cases present")

    keep = np.all([np.isfinite(values[m]) for m in wanted], axis=0)
    print(f"{keep.sum():,} of {n:,} cases have every member ({keep.mean():.1%}); the rest are dropped from all models")
    result = {}
    for k, v in cases.items():
        result[k] = v[keep] if (k.endswith(f"_{LEAD}") and v.shape[:1] == (n,)) else v
    for name, ms in sets.items():
        result[f"model_{name}_{LEAD}"] = np.stack([values[m][keep] for m in ms], axis=1)
        print(f"  {name}: members {ms}")
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.expanduser(), **result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

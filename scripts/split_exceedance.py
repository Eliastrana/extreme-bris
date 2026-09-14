#!/usr/bin/env python3
"""Count where precipitation passes the tail threshold: the Nordic cutout or the globe.

    sbatch -p defq -c 8 --time=01:00:00 -o logs/split-exceed-%j.out \
        --wrap "~/bris-env/.venv/bin/python scripts/split_exceedance.py \
                ~/bris-runs/tail-weight/cpu-shard-4-of-4.json"

WHY. The tail term was meant to be close to zero on ordinary days and to wake
up on the extreme ones. The first measured shard says otherwise: the raw term
was about the same size on ordinary states as on extreme ones, and an ordinary
state still had hundreds of grid points above 20 mm in six hours. The ranking
that defines "extreme" looked only at the Nordic half of the grid. The tail
term looks at the whole stretched grid, and somewhere in the tropics it rains
more than 20 mm in six hours every day.

That is a hypothesis about the data, and the data can settle it without the
model: read the target state for each measured date and count the points over
the threshold in each half. The totals should match the counts the
measurement took from the model's own targets, which also checks this script.

The Nordic cutout is the first block of points. The graph's cutout_mask marks
nodes 0 to 822,680 and boundary_mask the 536,600 global nodes after them; the
dataset's own grid sizes are used when it reports them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import _venv  # noqa: E402

_venv.ensure("anemoi")

import numpy as np  # noqa: E402

NORDIC_POINTS = 822_681


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measurements", nargs="+", type=Path,
                    help="outputs of measure_tail_weight.py (shards or merged)")
    ap.add_argument("--config-name", default="finetune_tail")
    ap.add_argument("--threshold-mm", type=float, default=20.0)
    args = ap.parse_args()

    import _compose
    from anemoi.datasets import open_dataset

    rows = []
    for f in args.measurements:
        rows += json.loads(f.read_text())["states"]
    print(f"=== {len(rows)} measured states from {len(args.measurements)} file(s)")

    cfg = _compose.compose(REPO / "bris/train", args.config_name)
    ds = open_dataset(cfg.dataloader.dataset)
    grids = getattr(ds, "grids", None)
    nordic = int(grids[0]) if grids else NORDIC_POINTS
    print(f"=== grid sizes {grids}, Nordic block = first {nordic} of {ds.shape[-1]} points")

    itp = ds.name_to_index["tp"]
    index = {str(d)[:19]: i for i, d in enumerate(ds.dates)}
    thr = args.threshold_mm / 1000.0  # both halves are in metres after the dataloader rescale

    print(f"\n{'date':20s} {'kind':9s} {'model':>7s} {'nordic':>7s} {'global':>7s} "
          f"{'nordic max':>10s} {'global max':>10s}")
    totals = {"extreme": [0, 0], "ordinary": [0, 0]}
    agree = 0
    for r in sorted(rows, key=lambda r: (r["kind"], r["date"])):
        t = index[r["date"][:19]]
        tp = np.asarray(ds[t:t + 1, itp:itp + 1, 0:1, :], dtype="float64").reshape(-1)
        n_over = int((tp[:nordic] > thr).sum())
        g_over = int((tp[nordic:] > thr).sum())
        totals[r["kind"]][0] += n_over
        totals[r["kind"]][1] += g_over
        agree += int(n_over + g_over == r.get("n_exceed"))
        print(f"{r['date']:20s} {r['kind']:9s} {r.get('n_exceed', -1):7d} {n_over:7d} {g_over:7d} "
              f"{1000 * np.nanmax(tp[:nordic]):8.1f}mm {1000 * np.nanmax(tp[nordic:]):8.1f}mm")

    print(f"\n=== counts agree with the measurement's own targets on {agree} of {len(rows)} states")
    for kind, (n_over, g_over) in totals.items():
        total = n_over + g_over
        if total:
            print(f"=== {kind:8s}: {n_over} Nordic, {g_over} global points over "
                  f"{args.threshold_mm:g} mm, {g_over / total:.0%} of them global")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

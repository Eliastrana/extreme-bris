#!/usr/bin/env python3
"""Set two forecasts of the same date side by side, field by field.

    scripts/compare_fields.py ~/bris-runs/cpu-smoke/20251004T00Z-baseline/nordic_*.nc \
                              ~/bris-runs/cpu-smoke/20251004T00Z-control/nordic_*.nc

WHAT IT IS FOR. Before a score says whether fine-tuning helped, this says what
it changed. Two runs from the same inputs and code path differ only in their
weights, so anything here is the fine-tuning. A control arm whose fields are
identical to the baseline did not load its own weights; one that differs by
tens of kelvin broke something; one that moves rain a little and temperature
hardly at all is what three thousand steps at a small learning rate should do.

Step zero is the initial state and should agree exactly. If it does not, the
two runs were not given the same inputs, and nothing after it means anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()

    with xr.open_dataset(args.reference) as a, xr.open_dataset(args.candidate) as b:
        ta, tb = a["time"].values, b["time"].values
        if not np.array_equal(ta, tb):
            print(f"time axes differ: {ta[0]}..{ta[-1]} ({len(ta)}) vs {tb[0]}..{tb[-1]} ({len(tb)})")
            return 1
        names = [n for n in a.data_vars if n in b.data_vars and "time" in a[n].dims
                 and a[n].ndim >= 3]
        print(f"=== {args.reference.parent.name}  vs  {args.candidate.parent.name}")
        print(f"{'field':28s} {'step':>5s} {'ref mean':>10s} {'cand mean':>10s} {'ref max':>9s} "
              f"{'cand max':>9s} {'mean |diff|':>11s} {'max |diff|':>10s} {'corr':>6s}")
        for name in names:
            for t in range(len(ta)):
                x = np.asarray(a[name].isel(time=t).values, dtype="float64").ravel()
                y = np.asarray(b[name].isel(time=t).values, dtype="float64").ravel()
                good = np.isfinite(x) & np.isfinite(y)
                x, y = x[good], y[good]
                d = np.abs(x - y)
                corr = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
                lead = int((ta[t] - ta[0]) / np.timedelta64(1, "h"))
                print(f"{name:28s} {lead:4d}h {x.mean():10.3f} {y.mean():10.3f} {x.max():9.3f} "
                      f"{y.max():9.3f} {d.mean():11.4f} {d.max():10.3f} {corr:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

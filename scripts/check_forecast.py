#!/usr/bin/env python3
"""Check a Bris forecast is physically plausible, not merely well-formed.

A run that completes proves the plumbing works. It does not prove the fields
mean anything — a wrong unit, an unrotated wind or a mismatched grid all produce
NetCDF that opens cleanly.

    python scripts/check_forecast.py ~/bris-runs/smoke/nordic_*.nc

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

# name -> (min, max, unit, note)
RANGES = {
    "2t":  (180.0, 340.0, "K", "surface air temperature"),
    "msl": (87000.0, 110000.0, "Pa", "mean sea-level pressure"),
    "tp":  (0.0, 0.5, "m", "total precipitation, must not be negative"),
    "ws":  (0.0, 120.0, "m/s", "wind speed"),
    "tcc": (0.0, 1.0, "-", "cloud fraction"),
    "hcc": (0.0, 1.0, "-", "cloud fraction"),
    "mcc": (0.0, 1.0, "-", "cloud fraction"),
    "lcc": (0.0, 1.0, "-", "cloud fraction"),
}


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__, file=sys.stderr)
        return 1

    try:
        import numpy as np
        import xarray as xr
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc.name} missing — run inside the Bris environment.", file=sys.stderr)
        return 1

    failures = 0
    for path in paths:
        if not path.exists():
            print(f"ERROR: no such file: {path}", file=sys.stderr)
            failures += 1
            continue

        print(f"=== {path.name}")
        ds = xr.open_dataset(path)
        print(f"  dims: {dict(ds.sizes)}")
        print()
        print(f"  {'field':<8}{'min':>14}{'max':>14}{'mean':>14}   verdict")

        for name, var in ds.data_vars.items():
            vals = np.asarray(var.values, dtype="float64")
            finite = np.isfinite(vals)
            n_bad = int((~finite).sum())
            if not finite.any():
                print(f"  {name:<8}{'':>42}   ALL NaN")
                failures += 1
                continue

            lo, hi, mean = float(np.nanmin(vals)), float(np.nanmax(vals)), float(np.nanmean(vals))
            notes = []
            if n_bad:
                notes.append(f"{n_bad:,} non-finite")

            spec = RANGES.get(str(name))
            if spec:
                exp_lo, exp_hi, unit, _ = spec
                if lo < exp_lo or hi > exp_hi:
                    notes.append(f"outside [{exp_lo:g}, {exp_hi:g}] {unit}")

            if str(name) == "tp" and lo < 0:
                notes.append("NEGATIVE precipitation — ReluBounding did not apply")

            if lo == hi:
                notes.append("constant field")

            verdict = "ok" if not notes else "; ".join(notes)
            if notes:
                failures += 1
            print(f"  {name:<8}{lo:>14.4g}{hi:>14.4g}{mean:>14.4g}   {verdict}")

        unchecked = [str(n) for n in ds.data_vars if str(n) not in RANGES]
        if unchecked:
            print(f"\n  no range defined for: {', '.join(unchecked)}")
        print()

    if failures:
        print(f"{failures} check(s) failed.", file=sys.stderr)
        print("A field can be well-formed and still wrong: check units, wind", file=sys.stderr)
        print("rotation and the LAM/global seam before trusting this.", file=sys.stderr)
        return 2

    print("All checks passed. Still worth plotting the seam by eye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

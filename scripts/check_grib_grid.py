#!/usr/bin/env python3
"""Confirm a retrieved GRIB field is on the grid the checkpoint expects.

`grid: N320` in a MARS request is a hint, not a guarantee — the tape archive can
return something else, and a silently regridded field would build a dataset the
model rejects for reasons that are hard to read.

    ~/bris-data-env/bin/python scripts/check_grib_grid.py /tmp/era5_syntax_test.grib
"""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_POINTS = 542_080          # N320 reduced Gaussian


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return 1

    try:
        import earthkit.data as ekd
    except ModuleNotFoundError:
        print("ERROR: earthkit-data missing. Use ~/bris-data-env/bin/python.", file=sys.stderr)
        return 1

    field = ekd.from_source("file", str(path))[0]
    facts = {}
    for key in ("gridType", "numberOfDataPoints", "N", "shortName", "dataDate", "dataTime"):
        try:
            facts[key] = field.metadata(key)
        except Exception:
            facts[key] = "(unavailable)"

    for k, v in facts.items():
        print(f"  {k:<20} {v}")
    print()

    pts = facts.get("numberOfDataPoints")
    ok = True
    if isinstance(pts, int):
        if pts == EXPECTED_POINTS:
            print(f"  points match N320 ({EXPECTED_POINTS:,}) — matches the checkpoint graph")
        else:
            print(f"  MISMATCH: {pts:,} points, expected {EXPECTED_POINTS:,}")
            print("  A dataset built from this will not align with the graph.")
            ok = False
    if "reduced_gg" not in str(facts.get("gridType", "")):
        print(f"  WARNING: gridType is {facts.get('gridType')!r}, expected a reduced Gaussian grid")
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

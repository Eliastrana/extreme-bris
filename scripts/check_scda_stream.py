#!/usr/bin/env python3
"""Check that accumulation requests go to the stream that holds them.

    ~/bris-data-env/bin/python scripts/check_scda_stream.py

WHY THIS IS A SCRIPT AND NOT A COMMENT. ECMWF retired the short cut-off
stream on 2026-05-12: the 06Z and 18Z cycles moved from `scda` into `oper`.
anemoi does not know that, and sends every 06Z and 18Z accumulation request to
scda regardless of date. A year-one build spent ten hours retrieving and then
died on `Expected 90, got 33` for May 2026, because scda holds only the first
eleven days of that month.

The date is the whole content of the fix, and a date is exactly the kind of
constant that gets rounded, moved a day, or generalised away by someone who
does not know it was measured. It was measured, against MARS, one request per
stream and month asking for a single grid box:

    scda 18Z, April 2026     30 of 30     scda 06Z, May 2026    1-11 May
    scda 18Z, May 2026       1-11 May     scda 18Z, Aug 2026    nothing
    oper 06Z, May 2026      12-31 May     oper 18Z, May 2026   12-31 May
    oper 06Z, Aug 2026       31 of 31     oper 18Z, Aug 2026    31 of 31

The streams meet exactly, with no overlap and no gap. Both sides matter: a
patch that always picks oper would lose every day before the cutover, which is
the only place those days exist.

Exits non-zero on any disagreement, so it can gate a build.
"""

from __future__ import annotations

# Hand over to an interpreter that has these, if this one does not. These
# scripts are run by path, so the shebang picks up whatever python3 is on
# PATH, and on the login node that one has none of the stack.
import sys as _sys, pathlib as _pathlib  # noqa: E401
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure('anemoi')


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anemoi_patches import apply_all                      # noqa: E402

# (date, time, expected stream, what it pins down)
CASES = [
    (20251115, 1800, "scda", "year one start, well before the cutover"),
    (20260430, 1800, "scda", "before the cutover"),
    (20260511, 1800, "scda", "last day scda holds"),
    (20260511, 600, "scda", "06Z moved on the same day as 18Z"),
    (20260512, 1800, "oper", "first day oper holds"),
    (20260512, 600, "oper", "06Z after the cutover"),
    (20260831, 1800, "oper", "well after the cutover"),
    (20260430, 1200, "oper", "12Z was always oper"),
    (20260512, 0, "oper", "00Z was always oper"),
]


def main() -> int:
    apply_all()
    from anemoi.datasets.create.sources import accumulations as acc

    bad = 0
    for date, time, want, why in CASES:
        got = acc._scda({"date": date, "time": time, "stream": "oper"})["stream"]
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {date} {time:04d} -> {got:4s} "
              f"(want {want})  {why}")

    print(f"\n{len(CASES) - bad} of {len(CASES)} correct")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

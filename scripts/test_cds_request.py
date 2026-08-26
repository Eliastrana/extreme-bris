#!/usr/bin/env python3
"""Smallest possible era5-complete request, to validate syntax before committing.

The full recipe asks for 178 fields from the MARS tape archive, which can take
days. If the request is malformed, that is a long wait for a rejection. This
asks for a single field with the same class/stream/type/grid, so any syntax
problem surfaces in minutes.

    ~/bris-data-env/bin/python scripts/test_cds_request.py

Two things it is really testing:

  * whether era5-complete accepts class: ea + stream: oper + type: an + grid:
    N320 as the recipe writes them
  * whether short parameter names ("2t") work, or whether MARS wants numeric
    codes ("167.128"). The generated recipe uses short names.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path("/tmp/era5_syntax_test.grib")


def main() -> int:
    try:
        import cdsapi
    except ModuleNotFoundError:
        print("ERROR: cdsapi not installed. Use ~/bris-data-env/bin/python.", file=sys.stderr)
        return 1

    request = {
        "class": "ea",
        "expver": "1",
        "stream": "oper",
        "type": "an",
        "date": "2025-04-01",
        "time": "00:00:00",
        "levtype": "sfc",
        "param": "2t",          # short name, as the generated recipe uses
        "grid": "N320",
        "format": "grib",
    }

    print("requesting one field from reanalysis-era5-complete:")
    for k, v in request.items():
        print(f"  {k:<10} {v}")
    print()

    client = cdsapi.Client()
    try:
        client.retrieve("reanalysis-era5-complete", request, str(OUT))
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        print(f"\nREQUEST FAILED:\n  {exc}\n", file=sys.stderr)
        msg = str(exc).lower()
        if "licence" in msg or "license" in msg:
            print("  -> accept the era5-complete licence in the CDS web form first.",
                  file=sys.stderr)
        elif "param" in msg:
            print("  -> short names may be rejected; try numeric MARS codes,",
                  file=sys.stderr)
            print("     e.g. 167.128 for 2t, and update make_era5_recipe.py.",
                  file=sys.stderr)
        elif "not found" in msg or "unknown" in msg:
            print("  -> one of class/stream/type/grid is wrong for this dataset.",
                  file=sys.stderr)
        return 2

    size = OUT.stat().st_size if OUT.exists() else 0
    print(f"\nOK: {OUT} ({size/1e6:.1f} MB)")
    print("Syntax accepted — the full recipe should retrieve cleanly.")
    if size:
        print("\nN320 has 542,080 points; a packed GRIB field should be roughly 0.5-1 MB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

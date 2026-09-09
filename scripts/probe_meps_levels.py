#!/usr/bin/env python3
"""Show what anemoi actually sees in the MEPS pressure-level file.

The level selection matches nothing and the build reports "No data found" with
no further explanation. Rather than vary the recipe again, this opens the file
the same way anemoi does and prints what it finds: the pressure coordinate's
dtype, values and attributes, and — the decisive part — the metadata anemoi
attaches to each field, including whatever it considers the level.

    ~/bris-data-env/bin/python scripts/probe_meps_levels.py
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

URL = ("https://thredds.met.no/thredds/dodsC/meps25epsarchive/"
       "2025/03/31/meps_det_pl_20250331T18Z.ncml")
VARS = ["air_temperature_pl", "x_wind_pl"]


def main() -> int:
    import xarray as xr

    # The surface branch of the build works and the pressure branch does not,
    # both through pydap. Test them side by side before concluding anything
    # about levels — a fetch that fails and is swallowed looks exactly like
    # "no data found".
    print("=== 0. can pydap open each file at all?\n")
    SFC = URL.replace("meps_det_pl_", "meps_det_sfc_")
    opened = {}
    for label, u in (("sfc", SFC), ("pl", URL)):
        try:
            opened[label] = xr.open_dataset(u, engine="pydap")
            print(f"  {label}: OK, {len(opened[label].variables)} variables")
        except Exception as exc:
            print(f"  {label}: FAILED  {type(exc).__name__}: {str(exc)[:110]}")
    if "pl" not in opened:
        print("\n  The pressure-level file cannot be opened at all. The build's")
        print("  'No data found' is a swallowed fetch failure, not a level")
        print("  selection problem — the recipe was never the issue.")
        if "sfc" in opened:
            print("  The surface file opens, so this is specific to that URL.")
        return 2
    print()

    print("=== 1. the pressure coordinate as xarray sees it\n")
    ds = opened["pl"]
    if "pressure" not in ds.coords and "pressure" not in ds.variables:
        print("  no `pressure` in coords or variables", file=sys.stderr)
        print(f"  coords: {list(ds.coords)}", file=sys.stderr)
        return 1
    p = ds["pressure"]
    print(f"  dtype      {p.dtype}")
    print(f"  shape      {p.shape}")
    print(f"  values     {list(p.values)}")
    print(f"  attrs      {dict(p.attrs)}")
    print(f"  dims of air_temperature_pl: {ds['air_temperature_pl'].dims}")

    print("\n=== 2. what anemoi makes of it\n")
    try:
        from anemoi.datasets.create.sources.xarray_support import XarrayFieldList
    except ImportError:
        from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList

    sub = ds[VARS]
    fl = XarrayFieldList.from_xarray(sub)
    print(f"  fields produced: {len(fl)}")
    seen = {}
    for f in fl:
        md = f.metadata()
        key = (md.get("variable"), md.get("level"), md.get("levelist"), md.get("levtype"))
        seen.setdefault(key, 0)
        seen[key] += 1
    print(f"  {'variable':<24}{'level':>10}{'levelist':>10}{'levtype':>10}   count")
    for (var, lev, levlist, levtype), n in sorted(seen.items(), key=lambda x: str(x[0])):
        print(f"  {str(var):<24}{str(lev):>10}{str(levlist):>10}{str(levtype):>10}   {n}")

    print("\n  If level is None, anemoi is not treating `pressure` as a level")
    print("  coordinate, and no `level:` selection can ever match.")
    print("  If level is a float, the recipe's integers will not compare equal.")

    print("\n=== 3. full metadata of one field\n")
    print(f"  {fl[0].metadata()}")

    # The decisive test: run the selection the build runs, in isolation.
    # LevelCoordinate.mars_names is ("level", "levelist") and normalise turns
    # 50.0 into 50, so `level` ought to match — but the build says otherwise.
    print("\n=== 4. the selection itself\n")
    LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000]
    for label, kw in (
        ("level (ints)",      dict(level=LEVELS)),
        ("level (floats)",    dict(level=[float(x) for x in LEVELS])),
        ("levelist",          dict(levelist=LEVELS)),
        ("pressure",          dict(pressure=LEVELS)),
        ("variable + level",  dict(variable=VARS, level=LEVELS)),
        ("variable only",     dict(variable=VARS)),
        ("no selection",      dict()),
    ):
        try:
            got = fl.sel(**kw)
            n = len(got)
        except Exception as exc:
            print(f"  {label:<20} raised {type(exc).__name__}: {str(exc)[:60]}")
            continue
        expect = "" if n else "   <- matches nothing"
        print(f"  {label:<20} {n:>6} fields{expect}")

    print("\n  Expect 6 variables x 12 levels x 67 leadtimes if a selection is")
    print("  working; 0 means that key is not how the level is addressed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

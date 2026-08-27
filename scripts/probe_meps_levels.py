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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

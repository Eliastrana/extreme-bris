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

    print("=== 1. the pressure coordinate as xarray sees it\n")
    ds = xr.open_dataset(URL, engine="pydap")
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

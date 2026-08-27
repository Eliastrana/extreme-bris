#!/usr/bin/env python3
"""Try every plausible way of opening the MEPS URL, and report which works.

.dodsrc now carries the right CA and curl gets HTTP 200, yet netCDF still fails
with a bare I/O error. So the problem is inside the DAP client rather than TLS
or the network. The realistic candidates:

  * libnetcdf 4.9.x negotiating DAP4 against a THREDDS dodsC endpoint, which
    speaks DAP2 — forced with a #mode=dap2 fragment
  * the .ncml suffix confusing protocol detection
  * pydap instead, a pure-Python DAP client that goes through requests and so
    honours REQUESTS_CA_BUNDLE like the rest of the stack

Whichever works becomes the `options:` block in the MEPS recipe.

    ~/bris-data-env/bin/python scripts/probe_opendap.py
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

BASE = ("https://thredds.met.no/thredds/dodsC/meps25epsarchive/"
        "2025/03/31/meps_det_sfc_20250331T18Z")


def attempt(label: str, fn) -> bool:
    print(f"--- {label}")
    try:
        n = fn()
        print(f"    OK — {n} variables\n")
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).split("\n")[0][:150]
        print(f"    failed: {type(exc).__name__}: {msg}\n")
        return False


def main() -> int:
    import xarray as xr

    winners = []

    def nc4(url):
        import netCDF4
        ds = netCDF4.Dataset(url)
        n = len(ds.variables)
        ds.close()
        return n

    def xr_open(url, **kw):
        ds = xr.open_dataset(url, **kw)
        n = len(ds.variables)
        ds.close()
        return n

    cases = [
        ("netCDF4, .ncml as-is",        lambda: nc4(BASE + ".ncml")),
        ("netCDF4, no .ncml suffix",    lambda: nc4(BASE)),
        ("netCDF4, #mode=dap2",         lambda: nc4(BASE + ".ncml#mode=dap2")),
        ("netCDF4, dap2:// scheme",     lambda: nc4("dap2://" + BASE.split("://", 1)[1] + ".ncml")),
        ("xarray netcdf4 engine",       lambda: xr_open(BASE + ".ncml", engine="netcdf4")),
        ("xarray pydap engine",         lambda: xr_open(BASE + ".ncml", engine="pydap")),
        ("xarray pydap, no suffix",     lambda: xr_open(BASE, engine="pydap")),
    ]

    for label, fn in cases:
        if attempt(label, fn):
            winners.append(label)

    print("=" * 60)
    if winners:
        print("WORKS:")
        for w in winners:
            print(f"  {w}")
        print("\nUse the first one in the recipe. For a pydap variant that means")
        print("adding to the opendap source:\n")
        print("    options:")
        print("      engine: pydap")
        return 0

    print("Nothing worked. The remaining option is to stop streaming and")
    print("download the four NetCDF files over HTTPS first, then use the")
    print("`netcdf` source on local paths — slower, but curl already works.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

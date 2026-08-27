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
    import os
    import xarray as xr

    # pydap goes through requests. Without REQUESTS_CA_BUNDLE it uses certifi,
    # which does not know eX3's TLS-interception CA — so every https attempt
    # fails before it starts. Earlier runs of this probe had these unset.
    missing = [v for v in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE")
               if not os.environ.get(v)]
    if missing:
        print(f"WARNING: {', '.join(missing)} unset.", file=sys.stderr)
        print("  Run `source scripts/env.sh` first, or the pydap cases test", file=sys.stderr)
        print("  nothing but a missing certificate authority.\n", file=sys.stderr)
    else:
        print(f"TLS bundle: {os.environ['REQUESTS_CA_BUNDLE']}\n")

    winners = []

    def nc4(url):
        import netCDF4
        ds = netCDF4.Dataset(url)
        n = len(ds.variables)
        ds.close()
        return n

    def pydap_direct(url, protocol="dap2"):
        from pydap.client import open_url
        ds = open_url(url, protocol=protocol) if protocol else open_url(url)
        return len(list(ds.keys()))

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
        # pydap only preserves the https scheme when `protocol` is given
        # explicitly (handlers/dap.py, the branch around determine_protocol).
        # Without it the scheme is guessed, becomes http, and eX3's TLS
        # interception turns that into 421 Misdirected Request.
        ("pydap open_url, protocol=dap2", lambda: pydap_direct(BASE + ".ncml")),
        # determine_protocol turns a dap2:// scheme into https explicitly
        # (handlers/dap.py lines 151-155), which is the documented way to be
        # unambiguous about the protocol.
        ("pydap open_url, dap2:// scheme",
         lambda: pydap_direct("dap2://" + BASE.split("://", 1)[1] + ".ncml", protocol=None)),
        ("xarray pydap, protocol=dap2",  lambda: xr_open(BASE + ".ncml", engine="pydap",
                                                         protocol="dap2")),
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

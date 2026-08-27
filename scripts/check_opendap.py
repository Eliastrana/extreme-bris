#!/usr/bin/env python3
"""Work out why an OPeNDAP URL will not open.

The MEPS build fails with `OSError: [Errno -68] NetCDF: I/O failure` on a URL
that is public and, from outside eX3, works. Two candidates:

  1. the netCDF4 build has no DAP support compiled in
  2. TLS. eX3 re-signs HTTPS, and netCDF's DAP client uses its own curl
     configuration in ~/.dodsrc rather than the SSL_CERT_FILE and CURL_CA_BUNDLE
     environment variables the rest of the stack honours

    ~/bris-data-env/bin/python scripts/check_opendap.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

URL = ("https://thredds.met.no/thredds/dodsC/meps25epsarchive/"
       "2025/03/31/meps_det_sfc_20250331T18Z.ncml")


def main() -> int:
    print("=== 1. plain HTTPS to the same host (curl)\n")
    r = subprocess.run(["curl", "-sS", "-m", "25", "-o", "/dev/null",
                        "-w", "  HTTP %{http_code}\n", URL + ".dds"],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    curl_ok = " 200" in r.stdout

    print("=== 2. does netCDF4 have DAP support?\n")
    try:
        import netCDF4
        print(f"  netCDF4     {netCDF4.__version__}")
        print(f"  libnetcdf   {netCDF4.__netcdf4libversion__}")
        has_dap = getattr(netCDF4, "__has_nc_inq_path__", None)
        print(f"  hdf5        {netCDF4.__hdf5libversion__}")
        for flag in ("__has_nc_open_mem__", "__has_cdf5_format__"):
            print(f"  {flag:<24}{getattr(netCDF4, flag, '?')}")
    except ModuleNotFoundError:
        print("  netCDF4 not installed", file=sys.stderr)
        return 1

    print("\n=== 3. ~/.dodsrc\n")
    dodsrc = Path.home() / ".dodsrc"
    if dodsrc.exists():
        print(f"  exists:\n{dodsrc.read_text()}")
    else:
        print("  MISSING — netCDF's DAP client has no CA configured")

    print("=== 4. TLS environment the rest of the stack uses\n")
    for var in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        print(f"  {var:<20}{os.environ.get(var, 'unset')}")

    print("\n=== 5. open it\n")
    try:
        ds = netCDF4.Dataset(URL)
        print(f"  OK — {len(ds.variables)} variables")
        ds.close()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}\n")
        if curl_ok:
            print("  curl reaches the host but netCDF cannot, which points at")
            print("  TLS trust rather than the network or the URL.")
            print("  Fix: write ~/.dodsrc with the system CA bundle —")
            print("       source scripts/env.sh   (now does this)")
        else:
            print("  curl cannot reach it either — this is network, not netCDF.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

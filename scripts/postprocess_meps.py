#!/usr/bin/env python3
"""Apply the physical conversions anemoi's filters cannot, to a built dataset.

Every conversion filter in anemoi-datasets 0.5.24 reads
`metadata(namespace="mars")["param"]`, and xarray-sourced fields return an empty
mars namespace, so all of them raise KeyError on NetCDF or OPeNDAP input. Only
`rename` works. The recipe therefore renames columns to their target names while
they still hold the source quantity, and this converts the values in place.

Three conversions, each of which silently produces plausible-looking output if
skipped:

  winds     x_wind/y_wind are aligned with the Lambert grid; the model expects
            earth-relative u/v
  2d        holds relative humidity in percent; must become dewpoint in kelvin
  w         holds m/s; ECMWF w is Pa/s, which differs by -rho*g — an order of
            magnitude and a sign

    ~/bris-data-env/bin/python scripts/postprocess_meps.py <dataset> --dry-run

Idempotency is NOT checked: running it twice rotates twice. It writes a marker
into the dataset attributes and refuses to run again unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000)
WIND_PAIRS = [("10u", "10v")] + [(f"u_{l}", f"v_{l}") for l in LEVELS]
MARKER = "bris_postprocessed"

R_D = 287.05        # J/(kg K)
G = 9.80665         # m/s2


def rotation_angle(lat, lon, nx, ny):
    """Angle from grid north to true north, per point, in radians."""
    lat2 = np.deg2rad(np.asarray(lat).reshape(ny, nx))
    lon2 = np.deg2rad(np.asarray(lon).reshape(ny, nx))
    dlat = np.gradient(lat2, axis=0)
    dlon = np.gradient(lon2, axis=0)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    return np.arctan2(dlon * np.cos(lat2), dlat).reshape(-1)


def rh_to_dewpoint(rh_pct, t_kelvin):
    """Magnus formula. rh in percent, t in kelvin, result in kelvin."""
    a, b = 17.625, 243.04
    t_c = np.asarray(t_kelvin, dtype="float64") - 273.15
    rh = np.clip(np.asarray(rh_pct, dtype="float64"), 1e-3, 100.0)
    alpha = np.log(rh / 100.0) + (a * t_c) / (b + t_c)
    return (b * alpha) / (a - alpha) + 273.15


def wz_to_omega(wz, t_kelvin, pressure_pa):
    """Geometric vertical velocity (m/s) to omega (Pa/s): w = -rho g wz."""
    rho = pressure_pa / (R_D * np.asarray(t_kelvin, dtype="float64"))
    return -rho * G * np.asarray(wz, dtype="float64")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--nx", type=int, default=949)
    ap.add_argument("--ny", type=int, default=1069)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if already marked")
    args = ap.parse_args()

    import zarr

    z = zarr.open(str(args.dataset), mode="r" if args.dry_run else "r+")
    if z.attrs.get(MARKER) and not args.force:
        print(f"ERROR: already post-processed ({z.attrs[MARKER]}).", file=sys.stderr)
        print("Rotating or converting twice is worse than not at all. Use --force "
              "only if you know the previous run did not complete.", file=sys.stderr)
        return 1

    names = list(z.attrs["variables"])
    data = z["data"]
    idx = {n: i for i, n in enumerate(names)}
    nt = data.shape[0]

    # --- winds ---------------------------------------------------------------
    lat, lon = np.asarray(z["latitudes"]), np.asarray(z["longitudes"])
    if lat.size != args.nx * args.ny:
        print(f"ERROR: {lat.size:,} points is not {args.nx}x{args.ny}", file=sys.stderr)
        return 1
    ang = rotation_angle(lat, lon, args.nx, args.ny)
    print(f"rotation angle: {np.degrees(ang.min()):.2f} .. {np.degrees(ang.max()):.2f} deg")
    print("  a Lambert grid over the Nordics should span roughly -30 to +30;")
    print("  a near-zero range means the angle was not recovered.\n")
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    for un, vn in WIND_PAIRS:
        if un not in idx or vn not in idx:
            continue
        for t in range(nt):
            u = np.asarray(data[t, idx[un], 0, :], dtype="float64")
            v = np.asarray(data[t, idx[vn], 0, :], dtype="float64")
            ue, ve = u * cos_a - v * sin_a, u * sin_a + v * cos_a
            drift = float(np.abs(np.hypot(u, v) - np.hypot(ue, ve)).max())
            if not args.dry_run:
                data[t, idx[un], 0, :] = ue.astype(data.dtype)
                data[t, idx[vn], 0, :] = ve.astype(data.dtype)
        print(f"  rotated {un}/{vn}  (speed drift {drift:.2e}, must be ~0)")

    # --- dewpoint ------------------------------------------------------------
    print()
    if "2d" in idx and "2t" in idx:
        for t in range(nt):
            rh = data[t, idx["2d"], 0, :]
            t2 = data[t, idx["2t"], 0, :]
            td = rh_to_dewpoint(rh, t2)
            bad = int((td > np.asarray(t2, dtype="float64") + 0.5).sum())
            if not args.dry_run:
                data[t, idx["2d"], 0, :] = td.astype(data.dtype)
        print(f"  2d: relative humidity -> dewpoint  "
              f"({np.nanmin(td):.1f}..{np.nanmax(td):.1f} K, {bad} points above 2t)")
        if bad:
            print("     dewpoint above temperature is unphysical — check the input units")
    else:
        print("  2d or 2t missing; skipping dewpoint")

    # --- omega ---------------------------------------------------------------
    print()
    for lev in LEVELS:
        wn, tn = f"w_{lev}", f"t_{lev}"
        if wn not in idx or tn not in idx:
            continue
        for t in range(nt):
            wz = data[t, idx[wn], 0, :]
            tk = data[t, idx[tn], 0, :]
            om = wz_to_omega(wz, tk, lev * 100.0)
            if not args.dry_run:
                data[t, idx[wn], 0, :] = om.astype(data.dtype)
        print(f"  w_{lev}: m/s -> Pa/s  ({np.nanmin(om):.3f}..{np.nanmax(om):.3f})")

    if not args.dry_run:
        import datetime
        z.attrs[MARKER] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f"\nmarked as post-processed")
    else:
        print("\nDry run — nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

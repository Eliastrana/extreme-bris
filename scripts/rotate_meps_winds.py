#!/usr/bin/env python3
"""Rotate MEPS winds from grid-relative to earth-relative, in a built dataset.

MEPS stores `x_wind`/`y_wind` aligned with the Lambert projection axes, while
the model was trained on ECMWF's earth-relative `u`/`v`. Skipping this gives
wind fields wrong by an angle that grows with distance from the projection's
reference longitude — plausible everywhere, correct nowhere.

anemoi's own rotate_winds filter cannot do it here: it reads
`metadata(namespace="mars")["param"]`, and xarray-sourced fields return an empty
mars namespace, so it raises KeyError on any NetCDF or OPeNDAP input. The
transform is pointwise and depends only on grid geometry, so applying it after
the build is equivalent.

    ~/bris-data-env/bin/python scripts/rotate_meps_winds.py \
        ~/bris-data/meps-2p5km-2025-6h-v1.zarr --dry-run

UNVERIFIED. Check the angle field against a known case before trusting output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PAIRS = [("10u", "10v")] + [(f"u_{l}", f"v_{l}") for l in
                            (50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000)]


def rotation_angle(lat: np.ndarray, lon: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Angle between grid north and true north, per point, in radians.

    Derived from the grid itself rather than from a projection definition: the
    direction of increasing y is grid north, and its bearing relative to true
    north is what the wind components must be turned by.
    """
    lat2 = np.deg2rad(lat.reshape(ny, nx))
    lon2 = np.deg2rad(lon.reshape(ny, nx))

    # Difference along +y, one-sided at the edges.
    dlat = np.gradient(lat2, axis=0)
    dlon = np.gradient(lon2, axis=0)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi          # wrap the dateline

    return np.arctan2(dlon * np.cos(lat2), dlat).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--nx", type=int, default=949)
    ap.add_argument("--ny", type=int, default=1069)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the angle field and what would change, write nothing")
    args = ap.parse_args()

    import zarr

    z = zarr.open(str(args.dataset), mode="r" if args.dry_run else "r+")
    names = list(z.attrs["variables"])
    lat = np.asarray(z["latitudes"])
    lon = np.asarray(z["longitudes"])

    if lat.size != args.nx * args.ny:
        print(f"ERROR: {lat.size:,} points is not {args.nx} x {args.ny} = "
              f"{args.nx * args.ny:,}", file=sys.stderr)
        return 1

    ang = rotation_angle(lat, lon, args.nx, args.ny)
    print(f"rotation angle: {np.degrees(ang.min()):.2f} .. {np.degrees(ang.max()):.2f} deg "
          f"(mean {np.degrees(ang.mean()):.2f})")
    print("  a Lambert grid over the Nordics should span roughly -30 to +30 deg;")
    print("  a near-zero range means the angle was not recovered.\n")

    cos_a, sin_a = np.cos(ang), np.sin(ang)
    data = z["data"]
    changed = 0

    for uname, vname in PAIRS:
        if uname not in names or vname not in names:
            print(f"  skip {uname}/{vname}: not in dataset")
            continue
        iu, iv = names.index(uname), names.index(vname)
        for t in range(data.shape[0]):
            u = np.asarray(data[t, iu, 0, :], dtype="float64")
            v = np.asarray(data[t, iv, 0, :], dtype="float64")
            u_e = u * cos_a - v * sin_a
            v_e = u * sin_a + v * cos_a
            if not args.dry_run:
                data[t, iu, 0, :] = u_e.astype(data.dtype)
                data[t, iv, 0, :] = v_e.astype(data.dtype)
        speed_before = float(np.hypot(u, v).mean())
        speed_after = float(np.hypot(u_e, v_e).mean())
        print(f"  {uname}/{vname}: mean speed {speed_before:.3f} -> {speed_after:.3f} "
              f"(must be unchanged; rotation preserves magnitude)")
        changed += 1

    print(f"\n{changed} wind pair(s) {'would be' if args.dry_run else ''} rotated.")
    if args.dry_run:
        print("Dry run — nothing written. Drop --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Match every gauge to a grid point, once, and say which gauges are unusable.

    scripts/build_station_index.py
    scripts/build_station_index.py --max-dist-km 3

WHY NOW. Scoring three arms over a year needs, for each station, the grid point
that stands for it. That mapping depends only on the grid and the station
coordinates, both of which exist today, so it can be settled and checked before
any forecast does. It is also the step where a scoring run quietly loses half
its stations, and a number computed on half the gauges looks exactly like a
number computed on all of them.

THE EDGE TRIM MATTERS AND IS EASY TO MISS. The dataloader drops fifty rows and
columns from every side of the MEPS grid, so the model neither reads nor
predicts them. A gauge out there has a nearest grid point, at a plausible
distance, that the model never produces a value for. Matching against the full
grid and scoring against the trimmed one would silently drop those stations, or
worse, pair them with whatever the output file happens to hold at that index.
So this reports both, and marks which side of the trim each station falls on.

DISTANCE IS THE OTHER QUIET FAILURE. A gauge in a valley matched to a point on
a ridge two kilometres away is not measuring the same weather, and at 2.5 km
spacing the nearest point can be over a kilometre off even in the middle of the
domain. The distance is kept per station so a scorer can tighten the cut
without rebuilding anything, and the distribution is printed so the default is
a choice rather than an inheritance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("zarr", "numpy")

import numpy as np  # noqa: E402

EARTH_R = 6371.0088          # km
TRIM = 50                    # rows and columns the dataloader drops per side
NX, NY = 949, 1069


def great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine, broadcast over whatever shapes come in."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest(slat, slon, glat, glon, block: int = 16):
    """Index of and distance to the closest grid point, per station.

    Blocked over stations rather than built as one array: a million grid points
    against several hundred stations is billions of pairs, and the whole point
    of this script is to be runnable on a login node.
    """
    idx = np.empty(slat.size, dtype="int64")
    dist = np.empty(slat.size, dtype="float64")
    for i in range(0, slat.size, block):
        j = slice(i, min(i + block, slat.size))
        d = great_circle_km(slat[j, None], slon[j, None], glat[None, :], glon[None, :])
        idx[j] = d.argmin(axis=1)
        dist[j] = d.min(axis=1)
    return idx, dist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observations", type=Path,
                    default=Path.home() / "bris-runs" / "observations")
    ap.add_argument("--element", default="precipitation")
    ap.add_argument("--dataset", type=Path,
                    default=Path.home() / "bris-data" / "meps-2p5km-year1-6h-v1.zarr")
    ap.add_argument("--max-dist-km", type=float, default=5.0)
    ap.add_argument("--nx", type=int, default=NX)
    ap.add_argument("--ny", type=int, default=NY)
    ap.add_argument("--trim", type=int, default=TRIM)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    import zarr

    obs = args.observations / f"{args.element}.npz"
    if not obs.exists():
        raise SystemExit(f"no observations at {obs}; run fetch_observations.py first")
    with np.load(obs, allow_pickle=False) as f:
        sids, slat, slon = f["stations"], f["lat"], f["lon"]
        present = np.isfinite(f["values"]).mean(axis=1)
    print(f"=== {len(sids)} stations with {args.element}\n")

    z = zarr.open(str(args.dataset), mode="r")
    glat = np.asarray(z["latitudes"], dtype="float64")
    glon = np.asarray(z["longitudes"], dtype="float64")
    if glat.size != args.nx * args.ny:
        raise SystemExit(f"{glat.size:,} grid points is not {args.nx}x{args.ny}")

    idx, dist = nearest(slat, slon, glat, glon)

    # Where each match lands on the grid, and whether the trim keeps it.
    row, col = np.divmod(idx, args.nx)
    inside = ((row >= args.trim) & (row < args.ny - args.trim)
              & (col >= args.trim) & (col < args.nx - args.trim))

    print("--- distance to the nearest grid point")
    for q in (50, 75, 90, 95, 99):
        print(f"  {q}th percentile   {np.percentile(dist, q):6.2f} km")
    print(f"  worst            {dist.max():6.2f} km\n")

    near = dist <= args.max_dist_km
    usable = near & inside & (present > 0.5)
    print(f"--- how many survive each cut, of {len(sids)}")
    print(f"  within {args.max_dist_km:g} km of a grid point   {int(near.sum()):5d}")
    print(f"  inside the {args.trim}-point edge trim   {int(inside.sum()):5d}")
    print(f"  reporting more than half the hours  {int((present > 0.5).sum()):5d}")
    print(f"  all three                           {int(usable.sum()):5d}\n")

    lost = near & ~inside
    if lost.any():
        print(f"  {int(lost.sum())} station(s) are close to a grid point that the")
        print("  model never predicts, because the trim removes it. Scoring")
        print("  against the full grid would have kept them silently.\n")

    out = args.out or (args.observations / f"station_index_{args.element}.npz")
    np.savez_compressed(
        out,
        stations=sids, lat=slat, lon=slon,
        grid_index=idx, distance_km=dist,
        row=row, col=col, inside_trim=inside,
        present_fraction=present, usable=usable,
        nx=args.nx, ny=args.ny, trim=args.trim,
        dataset=str(args.dataset), max_dist_km=args.max_dist_km,
    )
    print(f"wrote {out}")
    print("\nEvery station is kept, with its distance and its flags, so a scorer "
          "can tighten\nany of these cuts without rebuilding the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

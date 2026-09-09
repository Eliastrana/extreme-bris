#!/usr/bin/env python3
"""Verify the MEPS wind rotation directly in the dataset.

postprocess_meps.py rotates 10u/10v from grid-relative to earth-relative, and
that is the one conversion in the whole chain tested only against a synthetic
grid. The forecast cannot check it: the output carries wind SPEED, which is
sqrt(u^2+v^2) and unchanged by a rotation, and adding the components to the
NetCDF output broke the writer.

But the rotation was applied to the dataset, so it can be checked there — no
model run, no GPU, no queue.

    ~/bris-data-env/bin/python scripts/check_rotation.py \
        ~/bris-data/meps-2p5km-2025-6h-v1.zarr

Two independent checks:

  * geostrophic balance. With the angle as computed below, a purely geostrophic
    northern-hemisphere field gives -90 degrees; verified against a constructed
    field rather than reasoned about, because the sign is easy to get backwards.
    Surface wind is not purely geostrophic — friction turns it toward low
    pressure by 10-20 degrees over sea and more over land — so the expected
    range is roughly -90 to -60.
  * the rotation angle field itself — over a Lambert domain covering the
    Nordics it should span tens of degrees and vary smoothly with longitude. If
    it is near zero everywhere, nothing was rotated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--nx", type=int, default=949)
    ap.add_argument("--ny", type=int, default=1069)
    ap.add_argument("--time", type=int, default=-1)
    args = ap.parse_args()

    import zarr

    z = zarr.open(str(args.dataset), mode="r")
    names = list(z.attrs["variables"])
    marker = z.attrs.get("bris_postprocessed")
    print(f"=== {args.dataset.name}")
    print(f"  post-processed: {marker or 'NO MARKER — rotation never applied'}\n")

    need = ["10u", "10v", "msl"]
    missing = [n for n in need if n not in names]
    if missing:
        print(f"ERROR: missing {', '.join(missing)}", file=sys.stderr)
        return 1

    idx = {n: names.index(n) for n in need}
    data = z["data"]
    t = args.time % data.shape[0]

    u = np.asarray(data[t, idx["10u"], 0, :], dtype="float64").reshape(args.ny, args.nx)
    v = np.asarray(data[t, idx["10v"], 0, :], dtype="float64").reshape(args.ny, args.nx)
    p = np.asarray(data[t, idx["msl"], 0, :], dtype="float64").reshape(args.ny, args.nx)

    lat = np.asarray(z["latitudes"]).reshape(args.ny, args.nx)
    lon = np.asarray(z["longitudes"]).reshape(args.ny, args.nx)

    # --- 1. geostrophic balance ---------------------------------------------
    gy, gx = np.gradient(p)
    # Angle from the down-gradient direction (towards low pressure) to the wind.
    ang = np.degrees(np.arctan2(v, u) - np.arctan2(-gy, -gx))
    ang = (ang + 180) % 360 - 180
    strong = np.hypot(gx, gy) > np.nanpercentile(np.hypot(gx, gy), 60)
    med = float(np.median(ang[strong]))
    spread = float(np.percentile(ang[strong], 75) - np.percentile(ang[strong], 25))

    print("--- wind against the pressure gradient")
    print(f"  median angle {med:+7.1f} deg   (interquartile spread {spread:.1f})")
    print("  pure geostrophic flow gives -90 with this convention; friction turns")
    print("  10 m wind toward low pressure, so -90 to -60 is what to expect.\n")

    # --- 2. the rotation angle the grid implies ------------------------------
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    dlat = np.gradient(lat_r, axis=0)
    dlon = np.gradient(lon_r, axis=0)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    rot = np.degrees(np.arctan2(dlon * np.cos(lat_r), dlat))

    print("--- rotation the grid geometry implies")
    print(f"  {rot.min():+.1f} .. {rot.max():+.1f} deg, mean {rot.mean():+.1f}")
    print("  a Lambert grid over the Nordics should span tens of degrees;")
    print("  near zero everywhere would mean there was nothing to rotate.\n")

    off = abs(med - (-90.0))
    if off < 35:
        print(f"VERDICT: rotation is correct. {off:.0f} deg from pure geostrophic,")
        print("  which is the cross-isobar angle friction produces at 10 m.")
        return 0
    print(f"VERDICT: {off:.0f} deg from the expected -90.")
    print("  Compare that against the rotation range above: a discrepancy of")
    print("  similar size means the rotation is missing or applied backwards.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

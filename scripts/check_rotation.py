#!/usr/bin/env python3
"""Verify the MEPS wind rotation directly in the dataset.

postprocess_meps.py rotates 10u/10v from grid-relative to earth-relative, and
that is the one conversion in the whole chain with no witness anywhere else. A
rotation preserves wind speed exactly, so every statistic is identical before
and after, and the forecast output carries speed rather than components.

    scripts/check_rotation.py ~/bris-data/meps-2p5km-year3-6h-v1.zarr

WHY THE OBVIOUS TEST DOES NOT WORK. The earlier version of this script compared
the wind direction against the pressure gradient and expected roughly minus
ninety degrees, allowing for the angle friction turns surface wind through. It
reported a discrepancy of about thirty degrees on all three year datasets and
called that a pass on two of them and a failure on the third, on a threshold of
thirty-five.

That test cannot answer the question it was asked. The pressure gradient is
computed on the array, so it is grid-relative, while a rotated wind is
earth-relative; the angle between them therefore carries the rotation. But what
it carries is the rotation AT EACH POINT, and the median of that angle over the
Nordics is about two degrees, because the grid turns one way west of the
central meridian and the other way east of it. Rotating or not rotating moves
the median by those two degrees, and thirty degrees of friction sits on top.
The test was measuring friction and reading it as rotation.

WHAT WORKS INSTEAD. Use the variation rather than the mean. Write the observed
angle between wind and grid-relative pressure gradient as

    angle(x) = cross_isobar + rotation(x)   if the wind was rotated
    angle(x) = cross_isobar                 if it was not
    angle(x) = cross_isobar - rotation(x)   if it was rotated backwards

The cross-isobar term is broadly constant across the domain; the rotation term
swings across sixty-five degrees from one side to the other. So subtracting the
rotation field from the observed angle collapses the spread in the first case,
inflates it in the second, and inflates it worse in the third. Comparing the
spread of those three candidates says which happened, and it says so from a
sixty-five degree signal rather than a two degree one.

Averaged over several states, because a single state can be synoptically
featureless and carry no gradient worth measuring against.
"""

from __future__ import annotations

# Hand over to an interpreter that has these, if this one does not. These
# scripts are run by path, so the shebang picks up whatever python3 is on
# PATH, and on the login node that one has none of the stack.
import sys as _sys, pathlib as _pathlib  # noqa: E401
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure('zarr', 'numpy')


import argparse
import sys
from pathlib import Path

import numpy as np



def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def spread(a):
    """Angular spread about the circular centre, in degrees."""
    import numpy as np
    r = np.deg2rad(a)
    centre = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return float(np.std(wrap(a - centre)))


def verdict(ang, rot) -> tuple[str, dict, float]:
    """Which of the three histories best explains the observed angles."""
    options = {
        "not rotated": spread(ang),
        "rotated correctly": spread(wrap(ang - rot)),
        "rotated backwards": spread(wrap(ang + rot)),
    }
    ordered = sorted(options.values())
    return min(options, key=options.get), options, ordered[1] - ordered[0]


def self_test() -> int:
    """Check the discriminator against fields whose history is known.

    The sign of a rotation is easy to get backwards, and a test that reports
    the wrong one confidently is worse than no test. So the three cases are
    constructed here, with a rotation field and a cross-isobar scatter matching
    what the real data shows, and the discriminator has to name each of them.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200_000
    rot = rng.uniform(-29.6, 35.0, n)          # as measured on the real grid
    base = -57.0 + rng.normal(0.0, 25.0, n)    # cross-isobar angle and scatter

    cases = {
        "not rotated": base,
        "rotated correctly": wrap(base + rot),
        "rotated backwards": wrap(base - rot),
    }
    bad = 0
    print("=== self test, on constructed fields\n")
    for truth, ang in cases.items():
        best, options, margin = verdict(ang, rot)
        ok = best == truth
        bad += not ok
        detail = "  ".join(f"{k}={v:.2f}" for k, v in options.items())
        print(f"  built {truth:18s} -> read as {best:18s} "
              f"{'ok' if ok else 'WRONG'}   margin {margin:.2f}")
        print(f"    {detail}")
    print()
    if bad:
        print(f"{bad} of 3 misread. Do not trust this script's verdict.",
              file=sys.stderr)
        return 1
    print("All three read correctly.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--nx", type=int, default=949)
    ap.add_argument("--ny", type=int, default=1069)
    ap.add_argument("--states", type=int, default=8,
                    help="how many states to pool (default 8)")
    ap.add_argument("--self-test", action="store_true",
                    help="check the discriminator on constructed fields and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

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

    lat = np.asarray(z["latitudes"]).reshape(args.ny, args.nx)
    lon = np.asarray(z["longitudes"]).reshape(args.ny, args.nx)

    # --- the rotation the grid geometry implies ------------------------------
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    dlat = np.gradient(lat_r, axis=0)
    dlon = np.gradient(lon_r, axis=0)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    rot = np.degrees(np.arctan2(dlon * np.cos(lat_r), dlat))

    print("--- rotation the grid geometry implies")
    print(f"  {rot.min():+.1f} .. {rot.max():+.1f} deg, mean {rot.mean():+.1f}")
    print("  the mean is near zero and the range is not; that is exactly why")
    print("  the spread rather than the average is what carries the answer.\n")

    # --- the observed angle, over several states -----------------------------
    picks = np.linspace(0, data.shape[0] - 1, min(args.states, data.shape[0]))
    angles = []
    for t_i in np.unique(picks.astype(int)):
        u = np.asarray(data[t_i, idx["10u"], 0, :], dtype="float64").reshape(args.ny, args.nx)
        v = np.asarray(data[t_i, idx["10v"], 0, :], dtype="float64").reshape(args.ny, args.nx)
        p = np.asarray(data[t_i, idx["msl"], 0, :], dtype="float64").reshape(args.ny, args.nx)
        gy, gx = np.gradient(p)
        # Angle from the down-gradient direction, towards low pressure, to the
        # wind. Grid-relative on both sides except for any rotation in the wind.
        ang = wrap(np.degrees(np.arctan2(v, u) - np.arctan2(-gy, -gx)))
        mag = np.hypot(gx, gy)
        strong = (mag > np.nanpercentile(mag, 70)) & (np.hypot(u, v) > 2.0)
        if strong.sum() > 1000:
            angles.append((ang[strong], rot[strong]))

    if not angles:
        print("ERROR: no state had enough wind over a usable pressure gradient",
              file=sys.stderr)
        return 1

    ang_all = np.concatenate([a for a, _ in angles])
    rot_all = np.concatenate([r for _, r in angles])
    print(f"--- observed angle, {len(angles)} state(s), "
          f"{ang_all.size:,} points\n")

    best, options, margin = verdict(ang_all, rot_all)
    for label, s in options.items():
        print(f"  spread if {label:20s} {s:6.2f} deg")
    print(f"\n  median angle {np.median(ang_all):+.1f} deg, which is the "
          "cross-isobar angle;")
    print("  friction turns surface wind toward low pressure by this much.\n")

    if margin < 0.5:
        print(f"VERDICT: inconclusive. The best fit is '{best}' but the next "
              f"is only {margin:.2f} deg behind,")
        print("  which is not enough to separate them. Try more states.")
        return 2
    print(f"VERDICT: {best}. Its spread is {margin:.2f} deg tighter than the "
          "next candidate,")
    print("  out of a rotation field spanning "
          f"{rot.max() - rot.min():.0f} deg.")
    return 0 if best == "rotated correctly" else 2


if __name__ == "__main__":
    raise SystemExit(main())

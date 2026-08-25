#!/usr/bin/env python3
"""Split the checkpoint's data nodes into the LAM and global parts.

dump_grid.py writes every data node as one flat array, but the cutout is two
grids concatenated: MEPS inside, N320 outside. Separating them gives the MEPS
domain extent and bounding box, which is what a dataset has to reproduce.

    python scripts/analyse_grid.py ~/bris-runs/grid.npz

Classification uses the fact that a reduced Gaussian grid puts many points on
each of a few hundred latitude rows, while a Lambert-conformal LAM gives a
nearly unique latitude per point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def find_split(lat: np.ndarray, purity: float = 1.0, decimals: int = 5,
               row_min_count: int = 10):
    """Index where the LAM block ends and the global block begins.

    A cutout concatenates two grids, so the parts are contiguous. Points are
    first marked as Gaussian-looking — sitting on a latitude row shared by many
    points — which a projected LAM produces only by coincidence. The boundary is
    then the smallest k whose tail is almost entirely such points, found by
    binary search on a cumulative sum. Requiring a share rather than purity
    tolerates the stray collisions that defeat pointwise classification.

    row_min_count is deliberately low: a reduced Gaussian grid has only about
    twenty points on its polar rows, and a higher threshold would misclassify
    them as LAM and push the boundary to the end of the array.

    Returns (k, order).
    """
    n = len(lat)
    key = np.round(lat, decimals)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    dense = (counts[inverse] >= row_min_count).astype(np.int64)

    def tail_share(k):
        return (dense[k:].sum() / (n - k)) if k < n else 1.0

    def head_share(k):
        return (dense[:k].sum() / k) if k > 0 else 1.0

    if tail_share(n - max(1000, n // 100)) >= purity:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if tail_share(mid) >= purity:
                hi = mid
            else:
                lo = mid + 1
        return lo, "lam_first"

    if head_share(max(1000, n // 100)) >= purity:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if head_share(mid) >= purity:
                lo = mid
            else:
                hi = mid - 1
        return lo, "global_first"

    return None, "unknown"


def factorise(n: int, lo: int = 200, hi: int = 4000):
    return [(a, n // a) for a in range(lo, min(hi, int(n ** 0.5)) + 1) if n % a == 0]


def report(name, lat, lon):
    print(f"--- {name}: {len(lat):,} points")
    print(f"      lat  {lat.min():8.3f} .. {lat.max():8.3f}")
    lon180 = ((lon + 180) % 360) - 180
    print(f"      lon  {lon180.min():8.3f} .. {lon180.max():8.3f}  (normalised to +/-180)")
    print(f"      unique latitudes: {len(np.unique(np.round(lat, 5))):,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--node-set", default="data")
    ap.add_argument("--purity", type=float, default=1.0,
                    help="share of Gaussian-looking points required in the tail")
    ap.add_argument("--split", type=int,
                    help="force the boundary index instead of detecting it")
    args = ap.parse_args()

    z = np.load(args.npz)
    try:
        lat = z[f"{args.node_set}_lat"]
        lon = z[f"{args.node_set}_lon"]
    except KeyError:
        print(f"ERROR: no '{args.node_set}' arrays. Present: {list(z.keys())}", file=sys.stderr)
        return 1

    print(f"=== {args.npz.name}: {len(lat):,} '{args.node_set}' nodes\n")

    if args.split is not None:
        k, order = args.split, "lam_first"
        print(f"using forced boundary {k:,}")
    else:
        k, order = find_split(lat, args.purity)
    if k is None:
        print("Could not find a clean boundary between the two grids.", file=sys.stderr)
        print("Neither half looks like a reduced Gaussian grid.", file=sys.stderr)
        return 2

    if order == "lam_first":
        lam_sl, glob_sl = slice(0, k), slice(k, None)
    else:
        glob_sl, lam_sl = slice(0, k), slice(k, None)
    print(f"boundary at index {k:,}  ({order.replace('_', ' ')})\n")

    n_lam, n_glob = len(lat[lam_sl]), len(lat[glob_sl])

    report("LAM (MEPS)", lat[lam_sl], lon[lam_sl])
    fac = factorise(n_lam)
    if fac:
        print(f"      factorisations: {', '.join(f'{a}x{b}' for a, b in fac[:8])}")
        if (849, 969) in fac or (969, 849) in fac:
            print("      -> 849 x 969 CONFIRMED (matches CRPSFFTLoss xdim/ydim)")
            print("      -> pre-trim extent 949 x 1069, with trim_edge: 50")
    else:
        print("      no clean factorisation — the split may be imperfect")
    print()

    report("global (N320)", lat[glob_sl], lon[glob_sl])
    print(f"      N320 full is 542,080; missing here: {542_080 - n_glob:,}")
    print("      (points removed where the LAM covers them)")
    print()

    print(f"total: {n_lam:,} + {n_glob:,} = {n_lam + n_glob:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

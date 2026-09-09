#!/usr/bin/env python3
"""Undo a wind rotation that was interrupted partway through a dataset.

    scripts/repair_partial_rotation.py <dataset> --dry-run
    scripts/repair_partial_rotation.py <dataset>

WHY THIS IS NEEDED. The first version of postprocess_meps.py rotated one wind
pair at a time, looping over every state for that pair before recording it as
done. A job killed mid-pair therefore leaves the early states of that pair
rotated, the later ones not, and nothing at all written down. Rotation is not
idempotent, so a later run over the same data would rotate those early states a
second time, and no statistic could reveal it afterwards: rotation preserves
wind speed exactly, which is what made the conversion invisible in the first
place.

HOW THE BOUNDARY IS FOUND. Not from the data, which cannot tell rotated from
unrotated, but from the filesystem. This dataset is chunked as one state by all
variables, so each state is a single file on disk, and each pass rewrites those
files in order. The interrupted pass therefore left states 0..k with a
modification time later than every state it never reached. The last state in
the dataset was written by the previous completed pass and never touched again,
so its time is the dividing line: every state modified after it belongs to the
interrupted pass.

That gives a single boundary rather than a scattered set, which is what makes
this recoverable at all.

WHAT IT DOES. It rotates those states back, by the inverse of the same
per-point angle, leaving the pair uniformly unconverted across the whole
dataset. The rewritten postprocess_meps.py then treats it like any other
pending variable. Undoing a few hundred states is far cheaper than finishing
thirteen hundred, and it needs no statistics work, because the pass that
follows recomputes them anyway.

RUN IT ONLY WHEN THE JOB IS DEAD. Reading modification times while something is
still writing gives an answer that is stale by the time it is used.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv

_venv.ensure("zarr", "numpy")

import numpy as np  # noqa: E402

from postprocess_meps import PROGRESS, WIND_PAIRS, rotation_angle  # noqa: E402


def chunk_times(dataset: Path, nt: int) -> np.ndarray:
    """Modification time of the file holding each state."""
    root = dataset / "data"
    times = np.full(nt, np.nan)
    for t in range(nt):
        # zarr v2 names a chunk by its index along every dimension; the last
        # three are always zero here because each chunk spans them entirely.
        for name in (f"{t}.0.0.0", f"{t}/0/0/0"):
            p = root / name
            if p.exists():
                times[t] = p.stat().st_mtime
                break
    missing = int(np.isnan(times).sum())
    if missing:
        raise SystemExit(
            f"{missing} of {nt} state files could not be found under {root}. "
            "The store is laid out differently than this assumes; list it by "
            "hand before going further."
        )
    return times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--nx", type=int, default=949)
    ap.add_argument("--ny", type=int, default=1069)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import zarr

    z = zarr.open(str(args.dataset), mode="r" if args.dry_run else "r+")
    names = list(z.attrs["variables"])
    idx = {n: i for i, n in enumerate(names)}
    data = z["data"]
    nt = data.shape[0]
    done = set(z.attrs.get(PROGRESS, []))

    # The interrupted pair is the first one not yet recorded as done.
    pair = next(((u, v) for u, v in WIND_PAIRS
                 if u in idx and v in idx and not (u in done and v in done)), None)
    if pair is None:
        print("Every wind pair is recorded as done. Nothing to repair.")
        return 0
    un, vn = pair
    print(f"=== {args.dataset.name}")
    print(f"  recorded done : {', '.join(sorted(done)) or 'nothing'}")
    print(f"  interrupted   : {un}/{vn}")

    times = chunk_times(args.dataset, nt)
    cutoff = times[-1]
    touched = np.flatnonzero(times > cutoff)
    if touched.size == 0:
        print("\n  No state was modified after the last one, so the interrupted\n"
              "  pass had not written anything yet. Nothing to undo.")
        return 0

    k = int(touched.max())
    if touched.size != k + 1:
        print(f"\nERROR: {touched.size} states were modified after the last one, "
              f"but they do not form the range 0..{k}.\n"
              "The passes did not run in order, so the boundary argument does "
              "not hold and this cannot safely guess which states to undo.",
              file=sys.stderr)
        return 1

    def when(ts: float) -> str:
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    print(f"\n  states 0..{k} were written by the interrupted pass "
          f"({k + 1} of {nt})")
    print(f"  state {k}     {when(times[k])}")
    print(f"  state {k + 1}     {when(times[k + 1])}   <- untouched by it")
    gap = times[k] - times[k + 1]
    print(f"  gap           {gap / 60:.1f} minutes")
    if gap < 60:
        print("\n  That gap is small enough to be ambiguous. Look at the times "
              "before letting this write.", file=sys.stderr)
        if not args.dry_run:
            return 1

    lat, lon = np.asarray(z["latitudes"]), np.asarray(z["longitudes"])
    ang = rotation_angle(lat, lon, args.nx, args.ny)
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    if args.dry_run:
        print(f"\nDry run. Would rotate {k + 1} state(s) of {un}/{vn} back.")
        return 0

    print(f"\n=== undoing the rotation on {k + 1} state(s)")
    started = datetime.datetime.now()
    for t in range(k + 1):
        block = data[t]
        ue = np.asarray(block[idx[un], 0, :], dtype="float64")
        ve = np.asarray(block[idx[vn], 0, :], dtype="float64")
        # Inverse of the rotation the other script applies. Orthogonal, so the
        # inverse is the transpose and wind speed is preserved either way.
        u = ue * cos_a + ve * sin_a
        v = -ue * sin_a + ve * cos_a
        block[idx[un], 0, :] = u.astype(block.dtype)
        block[idx[vn], 0, :] = v.astype(block.dtype)
        data[t] = block
        if t % 25 == 0 or t == k:
            per = (datetime.datetime.now() - started).total_seconds() / (t + 1)
            left = datetime.timedelta(seconds=int(per * (k - t)))
            print(f"  {t + 1}/{k + 1}  {per:.2f} s each  about {left} left")

    print(f"\n{un}/{vn} is now unrotated across all {nt} states. "
          "postprocess_meps.py will treat it as pending like any other variable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

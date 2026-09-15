#!/usr/bin/env python3
"""Look for broken gauges before anything is scored against them.

    scripts/check_gauges.py ~/bris-runs/observations/precipitation_daily.npz
    scripts/check_gauges.py ~/bris-runs/observations/precipitation.npz

WHY THIS EXISTS. Frost marks faulty values as good. In the hourly record there
were stations reporting rain every hour of the year, one stuck at 117 mm for
days, and single 40 to 100 mm winter hours with every gauge within 40 km dry.
All carried quality code 0. A score computed against those is a score of the
gauges, and in the tail, where the thesis lives, a handful of them dominate.

WHAT IT DOES NOT DO. It removes nothing. Stations are flagged on properties no
real gauge has, and large values with dry neighbours are listed for a person
to look at, because a genuine local downpour looks exactly like that too, and
those are the events this project is trying to forecast.

THE TESTS, per station:
  - identical run: the same value several reports in a row, either at 5 mm or
    more or for a long stretch at 1 mm or more. Steady drizzle does repeat
    1.0 mm for a few hours at a tenth-of-a-millimetre resolution, so a short
    run of small values proves nothing; 117 mm for days, or 1.8 mm for 68
    hours straight, is a stuck sensor.
  - wet share: the fraction of reports above zero, against a limit set by the
    time resolution, since most days rain somewhere on the west coast but most
    hours do not.
  - yearly total: scaled to a year from the reports present. Norway's wettest
    gauges are around five metres.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("numpy")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.stations import great_circle_km, load_observations  # noqa: E402

# Limits per report length in hours: (identical run at >= 5 mm, identical run at
# >= 1 mm, wet share, big value mm).
LIMITS = {1: (3, 12, 0.5, 40.0), 24: (3, 6, 0.9, 50.0)}
YEARLY_MM = 7000.0
NEIGHBOUR_KM = 40.0


def longest_identical_run(x: np.ndarray, floor: float) -> tuple[int, float]:
    best, value, run = 1, 0.0, 1
    for a, b in zip(x[:-1], x[1:]):
        if np.isfinite(b) and b >= floor and a == b:
            run += 1
            if run > best:
                best, value = run, float(b)
        else:
            run = 1
    return best, value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("observations", type=Path)
    ap.add_argument("--max-quality", type=int, default=4,
                    help="blank values Frost codes worse than this first (default 4)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    obs = load_observations(args.observations, max_quality=args.max_quality)
    v = obs["values"].astype("float64")
    times, st = obs["times"], obs["stations"]
    step_h = int(np.median(np.diff(times)) / np.timedelta64(1, "h"))
    if step_h not in LIMITS:
        raise SystemExit(f"reports are {step_h} h apart; limits exist for {sorted(LIMITS)}")
    big_run, long_run, wet_limit, big_mm = LIMITS[step_h]
    print(f"=== {args.observations.name}: {len(st)} gauges x {len(times)} reports "
          f"{step_h} h apart")
    if obs["quality_dropped"]:
        print(f"  blanked {obs['quality_dropped']:,} value(s) Frost codes worse than "
              f"{args.max_quality}, before any test")

    present = np.isfinite(v)
    n = present.sum(axis=1)
    wet = np.where(n > 0, ((v > 0) & present).sum(axis=1) / np.maximum(n, 1), np.nan)
    yearly = np.where(n > 0, np.nansum(v, axis=1) / np.maximum(n, 1) * 8766 / step_h, np.nan)

    flagged = {}
    for i, sid in enumerate(st):
        reasons = []
        for floor, limit in ((5.0, big_run), (1.0, long_run)):
            run, value = longest_identical_run(v[i], floor)
            if run >= limit:
                reasons.append(f"{run} identical reports of {value:g} mm")
                break
        if n[i] >= 30 and wet[i] > wet_limit:
            reasons.append(f"wet in {wet[i]:.0%} of reports")
        if n[i] >= 30 and yearly[i] > YEARLY_MM:
            reasons.append(f"{yearly[i]:,.0f} mm a year")
        if reasons:
            flagged[str(sid)] = reasons

    print(f"\n--- {len(flagged)} gauge(s) with properties no working gauge has")
    for sid, reasons in flagged.items():
        print(f"  {sid:9s} " + "; ".join(reasons))

    print(f"\n--- wet share, percentiles 50/90/99: "
          + " / ".join(f"{p:.2f}" for p in np.nanpercentile(wet, [50, 90, 99])))
    print(f"--- yearly total mm, percentiles 50/90/99: "
          + " / ".join(f"{p:,.0f}" for p in np.nanpercentile(yearly, [50, 90, 99])))

    # Large values nobody nearby saw, for a person to judge. Flagged gauges are
    # left out on both sides: as candidates and as the neighbours that vouch.
    good = np.array([str(s) not in flagged for s in st])
    lonely = []
    for i in np.where(good)[0]:
        rows = np.where(v[i] >= big_mm)[0]
        if not len(rows):
            continue
        d = great_circle_km(obs["lat"][i], obs["lon"][i], obs["lat"], obs["lon"])
        near = good & (d < NEIGHBOUR_KM) & (np.arange(len(st)) != i)
        for j in rows:
            window = v[near, max(j - 1, 0):j + 2]
            seen = float(np.nanmax(window)) if np.isfinite(window).any() else np.nan
            if not np.isfinite(seen) or seen < 0.2 * v[i, j]:
                lonely.append({"station": str(st[i]), "time": str(times[j]),
                               "mm": float(v[i, j]), "neighbours": int(near.sum()),
                               "neighbour_max_mm": None if np.isnan(seen) else seen})

    print(f"\n--- {len(lonely)} value(s) of {big_mm:g} mm or more that no gauge within "
          f"{NEIGHBOUR_KM:g} km came within a fifth of, one report either side")
    for r in sorted(lonely, key=lambda r: -r["mm"])[:40]:
        nb = "no neighbour reported" if r["neighbour_max_mm"] is None else \
            f"max {r['neighbour_max_mm']:.1f} mm at {r['neighbours']} gauge(s)"
        print(f"  {r['station']:9s} {r['time'][:13]}  {r['mm']:6.1f} mm   {nb}")

    if "quality" in np.load(args.observations).files:
        q = np.load(args.observations)["quality"]
        codes, counts = np.unique(q[present], return_counts=True)
        print("\n--- quality codes: " + ", ".join(f"{c}: {k:,}" for c, k in zip(codes, counts)))

    out = args.out or args.observations.with_name(args.observations.stem + "_check.json")
    limits = {"identical_run_5mm": big_run, "identical_run_1mm": long_run,
              "wet_share": wet_limit, "big_mm": big_mm, "yearly_mm": YEARLY_MM}
    out.write_text(json.dumps({"flagged": flagged, "lonely": lonely, "limits": limits},
                              indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""How rare is a precipitation threshold? Percentiles of the training target.

    scripts/precip_percentiles.py --out ~/bris-runs/tail-variants/percentiles.json

WHY. The next tail variant follows Wessel et al. (2026) and puts the threshold
at a moderately high percentile, about the 97.5th, instead of 20 mm per 6 h,
which on ordinary states almost no Nordic point reached. Choosing that number
needs the distribution of what the tail term actually sees: 6-hour MEPS
precipitation on the Nordic cutout, inside the training window.

WHAT IT READS. The MEPS year files, the same trim of 50 points as the cutout,
in millimetres as stored (the config rescales to metres afterwards, which
does not change a percentile). Every third day, all four states of it, so all
times of day are equally represented and the day's four states also give the
06-06 UTC daily sum the evaluation scores against. Each state is one 400 MB
chunk, so reading every state would mean about a terabyte; a third is plenty
for percentiles of a billion values.

WHAT IT REPORTS, for 6-hour and for daily values separately:
  percentiles   over every point and time, dry ones included, as in the paper
  wet share     above 0.1 mm
  exceedance    how often 5, 10, 20 and 50 mm are passed, and which
                percentile each threshold is
and the same for the gauges' daily values in the evaluation observations, so
a grid threshold can be put next to the gauge threshold it corresponds to.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("zarr", "numpy")

import numpy as np  # noqa: E402

DATA = Path.home() / "bris-data"
FILES = ["meps-2p5km-year3-6h-v1.zarr", "meps-2p5km-year2-6h-v1.zarr"]
SHAPE = (1069, 949)
TRIM = 50
# Anything under 0.1 mm, including the small negatives an accumulation
# difference can leave, is one dry bin. Rounded so 1, 5, 20 mm are exact edges.
EDGES = np.concatenate([[-np.inf, 0.1], np.round(np.arange(0.2, 400.0, 0.05), 2), [np.inf]])
PERCENTILES = [90.0, 95.0, 97.0, 97.5, 98.0, 99.0, 99.5, 99.9, 99.99]
THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 50.0]


def read_day(job: tuple[str, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Histogram counts of the four 6-hour fields and of their sum."""
    import zarr

    path, idx = job
    z = zarr.open(path, mode="r")
    tp = z.attrs["variables"].index("tp")
    fields = []
    for i in idx:
        f = np.asarray(z["data"][i, tp, 0, :], dtype="float64").reshape(SHAPE)
        fields.append(f[TRIM:-TRIM, TRIM:-TRIM])
    six = np.zeros(EDGES.size - 1, dtype="int64")
    for f in fields:
        six += np.histogram(f[np.isfinite(f)], bins=EDGES)[0]
    total = np.sum(fields, axis=0)
    daily = np.histogram(total[np.isfinite(total)], bins=EDGES)[0]
    return six, daily


def jobs(start: str, end: str, every: int) -> list[tuple[str, list[int]]]:
    """Every n-th day: the states valid 12, 18, 00 and 06, the 06-06 day."""
    import zarr

    out = []
    for name in FILES:
        z = zarr.open(str(DATA / name), mode="r")
        dates = np.asarray(z["dates"][:]).astype("datetime64[h]")
        where = {d: i for i, d in enumerate(dates)}
        day = np.datetime64(start, "D")
        while day <= np.datetime64(end, "D"):
            wanted = [day + np.timedelta64(h, "h") for h in (12, 18, 24, 30)]
            if all(w in where for w in wanted):
                out.append((str(DATA / name), [where[w] for w in wanted]))
            day += np.timedelta64(every, "D")
    # A day that straddles two files is skipped rather than stitched.
    return out


def summarise(counts: np.ndarray) -> dict:
    n = counts.sum()
    cum = np.cumsum(counts) / n
    upper = EDGES[1:]

    def pct(p):
        return float(upper[np.searchsorted(cum, p / 100.0)])

    def rate(t):
        # Bins are fine enough near every listed threshold to count exactly.
        return float(counts[EDGES[:-1] >= t].sum() / n)

    return {"values": int(n), "wet_share_over_0.1mm": float(counts[1:].sum() / n),
            "percentiles_mm": {f"{p:g}": pct(p) for p in PERCENTILES},
            "exceedance": {f"{t:g}": {"rate": rate(t), "percentile": 100.0 * (1.0 - rate(t))}
                           for t in THRESHOLDS}}


def gauges(path: Path, max_quality: int) -> dict:
    import evaluate_experiment as ev

    obs, _ = ev.load_screened_observations(path, max_quality)
    v = np.asarray(obs["values"], dtype="float64").ravel()
    v = v[np.isfinite(v)]
    counts = np.histogram(v, bins=EDGES)[0]
    return summarise(counts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-10-01")
    ap.add_argument("--end", default="2025-03-30")
    ap.add_argument("--every", type=int, default=3, help="use every n-th day")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--observations", type=Path, default=Path.home() / "bris-runs/observations")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    todo = jobs(args.start, args.end, args.every)
    print(f"{len(todo)} days, {4 * len(todo)} states", flush=True)
    six = np.zeros(EDGES.size - 1, dtype="int64")
    daily = np.zeros(EDGES.size - 1, dtype="int64")
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (s, d) in enumerate(pool.map(read_day, todo), 1):
            six += s
            daily += d
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} days", flush=True)

    report = {"window": [args.start, args.end], "every_nth_day": args.every, "days": len(todo),
              "grid": "MEPS Nordic cutout, trim 50", "six_hour": summarise(six),
              "daily_06_06": summarise(daily)}
    if (args.observations / "precipitation_daily.npz").exists():
        report["gauges_daily_evaluation_period"] = gauges(args.observations, 4)

    for name in ("six_hour", "daily_06_06", "gauges_daily_evaluation_period"):
        if name not in report:
            continue
        r = report[name]
        print(f"\n=== {name}: {r['values']:,} values, {r['wet_share_over_0.1mm']:.1%} wet")
        print("  percentiles: " + "  ".join(f"p{k} {v:.2f}" for k, v in r["percentiles_mm"].items()))
        print("  thresholds:  " + "  ".join(f"{k} mm = p{v['percentile']:.3f}" for k, v in r["exceedance"].items()))
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

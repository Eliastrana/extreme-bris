#!/usr/bin/env python3
"""Where and when the tail-term models help: the test-year gain split by region, season and wetness.

EXPLORATORY. The primary results are one number each over the whole test
year. This splits that number to ask whether the gain sits where heavy rain
matters (the west coast, autumn, the wettest gauges) or is spread thin. It
reads the cases the scoring already collected (scripts/_cases.py checks they
are the same station-days).

    scripts/gain_breakdown.py \\
        --cases ~/bris-runs/calibration-test/with-baseline-cases.npz \\
        --cases ~/bris-runs/a-vs-pavo/cases.npz \\
        --models baseline control2000 tailA2000 tailC2000 --comparator control2000 \\
        --observations ~/bris-runs/observations \\
        --out ~/bris-runs/gain-breakdown/gain-breakdown.json

THE SCORE is the one the primary results use: Brier score for more than the
threshold in the day at +30 h, after month-out isotonic calibration, fitted
on all gauges at once exactly as in scripts/hits_and_calibration.py. Each
subset is then scored with that pooled calibration; a subset is not
recalibrated on itself. Raw Brier is reported beside it. Differences against
the comparator carry the same seven-day block bootstrap.

THE SUBSETS, all defined without looking at the forecasts:
  region    from each gauge's position: Nord-Norge north of 65 N,
            Midt-Norge 63-65 N, Vestlandet south of 63 N and west of 7.5 E
            (the coast and fjords from Agder's west end to Møre), and
            Sør- og Østlandet for the rest.
  season    by calendar month of the valid day: winter DJF, spring MAM,
            summer JJA, autumn SON.
  wetness   each gauge's own mean daily precipitation over the test year,
            split at the quartiles: the wettest quarter of the gauges
            against the driest. That uses the observations, but only the
            gauge's average, never the day being scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402

import _cases  # noqa: E402
from hits_and_calibration import calibrated_cv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import paired_block_bootstrap  # noqa: E402

THRESHOLDS = [10.0, 20.0, 50.0]
SEASONS = {"vinter": (12, 1, 2), "vår": (3, 4, 5), "sommer": (6, 7, 8), "høst": (9, 10, 11)}


def region_of(lat: float, lon: float) -> str:
    if lat >= 65.0:
        return "Nord-Norge"
    if lat >= 63.0:
        return "Midt-Norge"
    if lon < 7.5:
        return "Vestlandet"
    return "Sør- og Østlandet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", action="append", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--comparator", required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.comparator not in args.models:
        raise SystemExit("the comparator has to be one of --models")

    c = _cases.load(args.cases, args.models)
    o, dates, station = c["obs"], c["dates"], c["station"]
    month = dates.astype("datetime64[M]").astype(int)
    cal_month = (dates.astype("datetime64[M]").astype(int) % 12) + 1

    with np.load(args.observations.expanduser() / "precipitation_daily.npz", allow_pickle=False) as d:
        where = {s: (float(la), float(lo)) for s, la, lo in zip(d["stations"], d["lat"], d["lon"])}
    names = c["stations"]
    missing = [s for s in names if s not in where]
    if missing:
        raise SystemExit(f"{len(missing)} gauges have no position, e.g. {missing[:3]}")
    station_region = np.array([region_of(*where[s]) for s in names])
    region = station_region[station]

    gauge_mean = np.bincount(station, weights=o, minlength=names.size) / np.maximum(
        np.bincount(station, minlength=names.size), 1)
    present = np.bincount(station, minlength=names.size) > 0
    q1, q3 = np.quantile(gauge_mean[present], [0.25, 0.75])
    wetness = np.where(gauge_mean[station] >= q3, "våteste fjerdedel",
                       np.where(gauge_mean[station] <= q1, "tørreste fjerdedel", "midten"))

    subsets = {"hele teståret": np.ones(o.size, bool)}
    for r in ("Vestlandet", "Sør- og Østlandet", "Midt-Norge", "Nord-Norge"):
        subsets[r] = region == r
    for s, months in SEASONS.items():
        subsets[s] = np.isin(cal_month, months)
    for w in ("våteste fjerdedel", "tørreste fjerdedel"):
        subsets[w] = wetness == w

    gauges_per_region = {r: int(((station_region == r) & present).sum()) for r in np.unique(station_region)}
    report = {"exploratory": True, "lead_hours": _cases.LEAD, "station_days": int(o.size),
              "comparator": args.comparator, "gauges_per_region": gauges_per_region,
              "wetness_quartiles_mm_per_day": [float(q1), float(q3)], "thresholds": {}}
    print(f"{o.size:,} station-days, comparator {args.comparator}; gauges per region: {gauges_per_region}")
    others = [l for l in args.models if l != args.comparator]

    for t in THRESHOLDS:
        event = (o > t).astype(float)
        cal_sq, raw_sq = {}, {}
        for label in args.models:
            p = (c["models"][label] > t).mean(axis=1)
            cal_sq[label] = (calibrated_cv(p, event, month) - event) ** 2
            raw_sq[label] = (p - event) ** 2
        rows = {}
        print(f"\n===== {t:g} mm: calibrated Brier minus {args.comparator}, per cent of its Brier [95 % interval]")
        print(f"  {'subset':20s} {'days':>8s} {'events':>7s}   " + "   ".join(f"{l:>26s}" for l in others))
        for name, sel in subsets.items():
            n_ev = int(event[sel].sum())
            row = {"station_days": int(sel.sum()), "events": n_ev, "models": {}}
            cells = []
            for label in others:
                entry = {}
                for kind, sq in (("calibrated", cal_sq), ("raw", raw_sq)):
                    base = float(sq[args.comparator][sel].mean())
                    if n_ev < 20:
                        entry[kind] = None
                        continue
                    bt = paired_block_bootstrap(sq[label][sel] - sq[args.comparator][sel], dates[sel],
                                                block_days=7, replicates=args.replicates)
                    entry[kind] = {"brier": float(sq[label][sel].mean()), "comparator_brier": base,
                                   "difference": bt["mean_difference"], "ci": [bt["ci_lower"], bt["ci_upper"]],
                                   "per_cent": 100 * bt["mean_difference"] / base if base else float("nan")}
                row["models"][label] = entry
                e = entry["calibrated"]
                if e is None:
                    cells.append(f"{'too few events':>26s}")
                else:
                    sure = "*" if e["ci"][1] < 0 or e["ci"][0] > 0 else " "
                    lo, hi = (100 * x / e["comparator_brier"] for x in e["ci"])
                    cells.append(f"{e['per_cent']:+6.1f}% [{lo:+5.1f}, {hi:+5.1f}]{sure}")
            rows[name] = row
            print(f"  {name:20s} {row['station_days']:8d} {n_ev:7d}   " + "   ".join(cells))
        report["thresholds"][f"{t:g}"] = rows
    print("\n  * the whole 95 % interval is on one side of zero")

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Heavy rain by each gauge's own climate, and calibrated Bris against MEPS.

EXPLORATORY, AND SAID SO. Written 2026-10-09, after the calibration test
(evaluation/calibration_test_plan.json) had been scored on the same test year
at 10, 20 and 50 mm. Nothing here was fixed before those results were seen.

TWO QUESTIONS.

1. Twenty millimetres is an ordinary wet day on the west coast and a rare one
   inland, so a fixed threshold mostly asks about the west coast. Here the
   event is a day above the gauge's own percentile (95, 99, 99.5) of its daily
   totals over the cases. The thresholds come from the observations alone, so
   they favour neither model, but they are in-sample in the observations: the
   test year defines what is rare at each gauge. A gauge needs MIN_DAYS cases
   to get a threshold at all.

2. How calibrated Bris stands against MEPS, the operational regional model,
   on the same cases. The MEPS forecast here is a single point forecast
   (~/bris-runs/meps/precipitation_daily.npz), so its probability is 0 or 1
   before calibration. That is a handicap against a four-member ensemble, and
   the comparison is reported with it, not as a verdict on MEPS's ensemble.

Probability, calibration, ROC area and the bootstrap are exactly those of
hits_and_calibration.py: share of members above the threshold, isotonic
calibration out of the calendar month, paired seven-day block bootstrap.

  python scripts/station_extremes.py \\
      --model "control2000=~/bris-runs/evaluation/control2000/nordic_*.nc" \\
      --model "tailC2000=~/bris-runs/evaluation/tailC2000/nordic_*.nc" \\
      --meps ~/bris-runs/meps/precipitation_daily.npz \\
      --observations ~/bris-runs/observations --start 2025-08-01 --end 2026-09-06 \\
      --cache ~/bris-runs/station-extremes/testyear-cases.npz \\
      --out ~/bris-runs/station-extremes/testyear.json
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

import hits_and_calibration as hc  # noqa: E402
import tail_calibration as tc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import paired_block_bootstrap  # noqa: E402

LEAD = 30
FIXED_MM = [10.0, 20.0, 50.0]
PERCENTILES = [95.0, 99.0, 99.5]
MIN_DAYS = 300
MEPS = "MEPS"


def station_thresholds(obs: np.ndarray, station: np.ndarray, q: float) -> np.ndarray:
    """Per case, its gauge's q-th percentile over the cases; NaN for gauges with too few."""
    thr = np.full(obs.shape, np.nan)
    for s in np.unique(station):
        mine = station == s
        if mine.sum() >= MIN_DAYS:
            thr[mine] = np.percentile(obs[mine], q)
    return thr


def score(p: np.ndarray, event: np.ndarray, block: np.ndarray, yes: np.ndarray) -> tuple[dict, np.ndarray]:
    cal = hc.calibrated_cv(p, event, block)
    hits = int((yes & (event == 1)).sum())
    n_event = int(event.sum())
    n_yes = int(yes.sum())
    return {
        "brier": float(((p - event) ** 2).mean()),
        "brier_calibrated_cv": float(((cal - event) ** 2).mean()),
        "roc_area": hc.roc_area(p, event),
        "hit_rate": hits / n_event if n_event else float("nan"),
        "false_alarm_share": (n_yes - hits) / n_yes if n_yes else float("nan"),
        "frequency_bias": n_yes / n_event if n_event else float("nan"),
    }, (cal - event) ** 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    ap.add_argument("--meps", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-09-06")
    ap.add_argument("--replicates", type=int, default=5000)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    specs = dict(item.split("=", 1) for item in args.model)
    labels = list(specs)
    cache = args.cache.expanduser()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cases = {k: data[k] for k in data.files}
    else:
        plan = {"required_models": labels, "leads_hours": [LEAD], "forecast_accumulation": "per_step",
                "period": {"start": args.start, "end": args.end, "cycle_hour_utc": 0},
                "observations": {"max_quality": 4, "max_distance_km": 5.0}}
        cases = tc.collect(plan, specs, args.observations.expanduser(), args.meps.expanduser())
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **cases)
    if f"station_{LEAD}" not in cases:
        raise SystemExit(f"{cache} predates station indexes; delete it and run again")

    o = cases[f"obs_{LEAD}"]
    dates = cases[f"dates_{LEAD}"].astype("datetime64[D]")
    station = cases[f"station_{LEAD}"]
    meps = cases[f"meps_{LEAD}"]
    block = dates.astype("datetime64[M]").astype(int)
    ens = {l: cases[f"model_{l}_{LEAD}"] for l in labels}
    m = ens[labels[0]].shape[1]
    forecasters = labels + [MEPS]
    pairs = [(labels[1], labels[0]), (labels[1], MEPS), (labels[0], MEPS)] if len(labels) == 2 else \
        [(l, MEPS) for l in labels]

    report = {"exploratory": True, "lead_hours": LEAD, "station_days": int(o.size), "members": int(m),
              "calibration": {"cross_validation": "month"}, "min_days_per_gauge": MIN_DAYS,
              "meps": "single point forecast, probability 0 or 1 before calibration", "events": {}}
    print(f"{o.size:,} station-days at {np.unique(station).size} gauges, {m} members, +{LEAD} h")

    definitions = [(f"{t:g} mm", np.full(o.shape, t)) for t in FIXED_MM] + \
                  [(f"gauge p{q:g}", station_thresholds(o, station, q)) for q in PERCENTILES]
    for name, thr in definitions:
        keep = np.isfinite(thr)
        ob, th, dt, bl = o[keep], thr[keep], dates[keep], block[keep]
        event = (ob > th).astype(float)
        res = {"events": int(event.sum()), "cases": int(keep.sum()),
               "threshold_mm": {"median": float(np.median(th)), "p10": float(np.percentile(th, 10)),
                                "p90": float(np.percentile(th, 90))},
               "forecasters": {}, "calibrated_brier_differences": {}}
        sq = {}
        for f in forecasters:
            if f == MEPS:
                p = (meps[keep] > th).astype(float)
                yes = p == 1
            else:
                above = ens[f][keep] > th[:, None]
                p = above.mean(axis=1)
                yes = above.sum(axis=1) >= int(np.ceil(0.5 * m))
            res["forecasters"][f], sq[f] = score(p, event, bl, yes)
        for a, b in pairs:
            bt = paired_block_bootstrap(sq[a] - sq[b], dt, block_days=7, replicates=args.replicates)
            res["calibrated_brier_differences"][f"{a} - {b}"] = {
                "mean": bt["mean_difference"], "ci_lower": bt["ci_lower"], "ci_upper": bt["ci_upper"],
                "relative": bt["mean_difference"] / res["forecasters"][b]["brier_calibrated_cv"]}
        report["events"][name] = res

        t = res["threshold_mm"]
        print(f"\n===== {name}: {res['events']:,} events in {res['cases']:,} cases "
              f"(threshold median {t['median']:.1f} mm, 10-90 % {t['p10']:.1f}-{t['p90']:.1f})")
        print(f"  {'':12s} {'Brier':>8s} {'calibrated':>10s} {'ROC':>6s} {'hit rate':>8s} {'f.al. share':>11s} {'bias':>5s}")
        for f in forecasters:
            r = res["forecasters"][f]
            print(f"  {f:12s} {r['brier']:8.5f} {r['brier_calibrated_cv']:10.5f} {r['roc_area']:6.3f} "
                  f"{r['hit_rate']:8.1%} {r['false_alarm_share']:11.1%} {r['frequency_bias']:5.2f}")
        for pair, d in res["calibrated_brier_differences"].items():
            print(f"    calibrated {pair}: {d['mean']:+.5f} [{d['ci_lower']:+.5f}, {d['ci_upper']:+.5f}]"
                  f"  ({d['relative']:+.1%})")

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

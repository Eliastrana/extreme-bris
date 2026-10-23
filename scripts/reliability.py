#!/usr/bin/env python3
"""Reliability, spread and rank histograms on the test year: why calibration is needed.

EXPLORATORY. No plan rests on this. The primary results say the tail-term
models beat the control arm after calibration; this shows what the
calibration is correcting. It reads the cases the scoring already collected
(scripts/_cases.py checks they are the same station-days), so every number
is on the cases the reported results were computed on.

    scripts/reliability.py \\
        --cases ~/bris-runs/calibration-test/with-baseline-cases.npz \\
        --cases ~/bris-runs/a-vs-pavo/cases.npz \\
        --models baseline control2000 tailA2000 tailC2000 \\
        --out ~/bris-runs/reliability/reliability.json

PER THRESHOLD (10, 20 and 50 mm in the day):
  reliability    the forecast probability (the share of the four members
                 over the threshold, so 0, 1/4, ... 1) against how often it
                 then rained that much, with a seven-day block bootstrap
                 interval. Raw, and after the same month-out isotonic
                 calibration the scoring uses, binned.
  Murphy         Brier = reliability - resolution + uncertainty, raw and
                 calibrated. Calibration can only shrink the first term;
                 the second is what a model knows, and calibration cannot
                 add to it.
PER MODEL, ON DAILY AMOUNTS:
  spread-skill   member spread against the error of the ensemble mean, as in
                 scripts/spread_skill.py: 1 is honest, below 1 overconfident.
                 Overall and in bins of the forecast amount.
  ranks          where the gauge falls among the sorted members.
MEPS is one deterministic run, so it appears in reliability with
probabilities 0 and 1 only, and not in the spread checks.
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
from spread_skill import AMOUNT_BINS, rank_histogram, stats  # noqa: E402

THRESHOLDS = [10.0, 20.0, 50.0]
CAL_BINS = [0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0000001]


def block_ids(dates: np.ndarray, days: int = 7) -> np.ndarray:
    return ((dates - dates.min()).astype(int) // days).astype(int)


def ratio_interval(num: np.ndarray, den: np.ndarray, blocks: np.ndarray, rng, replicates: int):
    """Block-bootstrap 95 % interval for sum(num) / sum(den)."""
    ids = np.unique(blocks)
    n = np.array([num[blocks == b].sum() for b in ids])
    d = np.array([den[blocks == b].sum() for b in ids])
    draws = rng.integers(0, ids.size, size=(replicates, ids.size))
    r = n[draws].sum(axis=1) / np.maximum(d[draws].sum(axis=1), 1e-12)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def reliability_table(p, event, bins, blocks, rng, replicates):
    rows = []
    for lo, hi in bins:
        sel = (p >= lo) & (p < hi) if hi is not None else (p == lo)
        n = int(sel.sum())
        if not n:
            continue
        lo_ci, hi_ci = ratio_interval(np.where(sel, event, 0.0), sel.astype(float), blocks, rng, replicates)
        rows.append({"bin": [lo, hi], "cases": n, "mean_forecast": float(p[sel].mean()),
                     "observed": float(event[sel].mean()), "observed_ci": [lo_ci, hi_ci]})
    return rows


def murphy(p, event, bins):
    base = event.mean()
    rel = res = 0.0
    for lo, hi in bins:
        sel = (p >= lo) & (p < hi) if hi is not None else (p == lo)
        if sel.any():
            w = sel.mean()
            rel += w * (p[sel].mean() - event[sel].mean()) ** 2
            res += w * (event[sel].mean() - base) ** 2
    return {"brier": float(((p - event) ** 2).mean()), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(base * (1 - base))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", action="append", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    c = _cases.load(args.cases, args.models)
    o, dates = c["obs"], c["dates"]
    blocks = block_ids(dates)
    month = dates.astype("datetime64[M]").astype(int)
    m = next(iter(c["models"].values())).shape[1]
    rng = np.random.default_rng(20261023)
    print(f"{o.size:,} station-days, {m} members, models: {', '.join(args.models)}, MEPS")

    raw_bins = [(k / m, None) for k in range(m + 1)]
    cal_bins = list(zip(CAL_BINS[:-1], CAL_BINS[1:]))
    report = {"exploratory": True, "lead_hours": _cases.LEAD, "station_days": int(o.size), "members": m,
              "thresholds": {}, "spread_skill": {}, "rank_histogram": {}}

    for t in THRESHOLDS:
        event = (o > t).astype(float)
        entry = {"events": int(event.sum()), "base_rate": float(event.mean()), "models": {}}
        print(f"\n===== {t:g} mm: {int(event.sum()):,} events, base rate {event.mean():.4f}")
        print(f"  {'model':12s} {'brier':>8s} {'REL':>9s} {'RES':>9s}   observed frequency at forecast 0, 1/4 .. 1")
        sources = dict(c["models"])
        for label, ens in sources.items():
            p = (ens > t).mean(axis=1)
            cal = calibrated_cv(p, event, month)
            raw_rows = reliability_table(p, event, raw_bins, blocks, rng, args.replicates)
            entry["models"][label] = {
                "raw": {"murphy": murphy(p, event, raw_bins), "reliability": raw_rows},
                "calibrated": {"murphy": murphy(cal, event, cal_bins),
                               "reliability": reliability_table(cal, event, cal_bins, blocks, rng, args.replicates)},
            }
            mu = entry["models"][label]["raw"]["murphy"]
            curve = " ".join(f"{r['observed']:.3f}" for r in raw_rows)
            print(f"  {label:12s} {mu['brier']:8.5f} {mu['reliability']:9.6f} {mu['resolution']:9.6f}   {curve}")
            muc = entry["models"][label]["calibrated"]["murphy"]
            print(f"  {'  calibrated':12s} {muc['brier']:8.5f} {muc['reliability']:9.6f} {muc['resolution']:9.6f}")
        if c["meps"] is not None:
            p = (c["meps"] > t).astype(float)
            entry["models"]["MEPS"] = {"raw": {"murphy": murphy(p, event, [(0.0, None), (1.0, None)]),
                                               "reliability": reliability_table(p, event, [(0.0, None), (1.0, None)],
                                                                                blocks, rng, args.replicates)}}
            mu = entry["models"]["MEPS"]["raw"]["murphy"]
            curve = " ".join(f"{r['observed']:.3f}" for r in entry["models"]["MEPS"]["raw"]["reliability"])
            print(f"  {'MEPS':12s} {mu['brier']:8.5f} {mu['reliability']:9.6f} {mu['resolution']:9.6f}   at 0 and 1: {curve}")
        report["thresholds"][f"{t:g}"] = entry

    print("\n===== spread against error of the ensemble mean, daily amounts (1 is honest, below 1 overconfident)")
    for label, ens in c["models"].items():
        s = stats(ens, o)
        by = []
        mean = ens.mean(axis=1)
        for lo, hi in AMOUNT_BINS:
            sel = (mean >= lo) & (mean < hi)
            if sel.sum() > 50:
                b = stats(ens[sel], o[sel])
                by.append({"lo": lo, "hi": None if np.isinf(hi) else hi, "cases": int(sel.sum()), **b})
        report["spread_skill"][label] = {**s, "by_forecast_amount": by}
        report["rank_histogram"][label] = rank_histogram(ens, o, rng)
        parts = " ".join("[{:g}-{}] {:.2f}".format(b["lo"], "" if b["hi"] is None else "{:g}".format(b["hi"]), b["ratio"])
                         for b in by)
        ranks = " ".join(f"{x:.3f}" for x in report["rank_histogram"][label])
        print(f"  {label:12s} ratio {s['ratio']:.2f}   by forecast amount: {parts}")
        print(f"  {'':12s} ranks {ranks}")

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

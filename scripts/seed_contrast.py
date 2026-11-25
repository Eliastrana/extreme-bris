#!/usr/bin/env python3
"""The primary endpoint of evaluation/seed_spectral_plan.json: the mean over seed pairs of tail minus control.

Each pair is a tail model and the control arm trained on the same seed, so
the two share sample order and ensemble noise and differ by the tail term
alone. For every station-day the calibrated Brier difference of each pair is
taken, and the pairs are averaged day by day before the bootstrap, so the
interval carries the spread between pairs as well as the spread between days:

    d(case) = mean over pairs of [ BS_cal(tail_i, case) - BS_cal(control_i, case) ]

Calibration is month-out isotonic regression per model, exactly as
scripts/hits_and_calibration.py --cv month does it (calibrated_cv is
imported from there). The cases are the cache that script writes, so every
model is on the same strict common station-days, MEPS included.

Also reported, as the plan lists them: each pair on its own at 10, 20 and
50 mm, the spread of the per-pair contrasts and of the tail models' own
calibrated Brier, how many pairs fall below zero, and raw Brier and
frequency bias for each model.

    python scripts/seed_contrast.py --cache ~/bris-runs/seed-spectral/testyear-cases.npz \\
        --pair tailCfft2000:controlfft2000 --pair tailCfft2000s2:controlfft2000s2 \\
        --pair tailCfft2000s3:controlfft2000s3 --out ~/bris-runs/seed-spectral/primary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("numpy", "xarray")

import numpy as np  # noqa: E402

from hits_and_calibration import calibrated_cv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import paired_block_bootstrap  # noqa: E402

LEAD = 30
THRESHOLDS = [10, 20, 50]
PRIMARY = 20


def interval(diff: np.ndarray, dates: np.ndarray, base: float, replicates: int) -> dict:
    bt = paired_block_bootstrap(diff, dates, block_days=7, replicates=replicates)
    return {"difference": bt["mean_difference"], "ci_lower": bt["ci_lower"], "ci_upper": bt["ci_upper"],
            "percent": 100 * bt["mean_difference"] / base,
            "percent_ci": [100 * bt["ci_lower"] / base, 100 * bt["ci_upper"] / base]}


def verdict(ci: dict) -> str:
    if ci["ci_upper"] < 0:
        return "positive"
    if ci["ci_lower"] > 0:
        return "negative"
    return "inconclusive"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--pair", action="append", required=True, metavar="TAIL:CONTROL")
    ap.add_argument("--replicates", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = [tuple(p.split(":", 1)) for p in args.pair]
    with np.load(args.cache.expanduser(), allow_pickle=False) as z:
        obs = z[f"obs_{LEAD}"].astype("float64")
        dates = z[f"dates_{LEAD}"].astype("datetime64[D]")
        labels = sorted({l for p in pairs for l in p})
        missing = [l for l in labels if f"model_{l}_{LEAD}" not in z.files]
        if missing:
            raise SystemExit(f"{args.cache} has no cases for {missing}")
        ens = {l: z[f"model_{l}_{LEAD}"] for l in labels}
    month = dates.astype("datetime64[M]").astype(int)
    print(f"{obs.size:,} station-days, {ens[labels[0]].shape[1]} members, +{LEAD} h, {len(pairs)} pairs")

    report = {"plan": "evaluation/seed_spectral_plan.json", "lead_hours": LEAD, "station_days": int(obs.size),
              "pairs": [f"{t} - {c}" for t, c in pairs], "replicates": args.replicates,
              "note": "differences are tail minus control in calibrated Brier; negative favours the tail model; "
                      "percent is of the mean calibrated Brier of the control arms in the pairs",
              "thresholds": {}}
    for t in THRESHOLDS:
        event = (obs > t).astype("float64")
        cal, models = {}, {}
        for l in labels:
            p = (ens[l] > t).mean(axis=1)
            c = calibrated_cv(p, event, month)
            cal[l] = (c - event) ** 2
            models[l] = {"brier_calibrated_cv": float(cal[l].mean()), "brier": float(((p - event) ** 2).mean()),
                         "frequency_bias": float(p.mean() / event.mean())}
        per_pair = {}
        for tail, ctl in pairs:
            ci = interval(cal[tail] - cal[ctl], dates, cal[ctl].mean(), args.replicates)
            per_pair[f"{tail} - {ctl}"] = ci
        base = float(np.mean([cal[c].mean() for _, c in pairs]))
        mean_diff = np.mean([cal[tl] - cal[c] for tl, c in pairs], axis=0)
        mean_ci = interval(mean_diff, dates, base, args.replicates)
        contrasts = np.array([d["difference"] for d in per_pair.values()])
        tails = np.array([models[tl]["brier_calibrated_cv"] for tl, _ in pairs])
        row = {"events": int(event.sum()), "models": models, "per_pair": per_pair, "mean_over_pairs": mean_ci,
               "pairs_below_zero": int((contrasts < 0).sum()),
               "spread_of_contrasts": {"sd": float(contrasts.std(ddof=1)) if contrasts.size > 1 else None,
                                       "range": float(contrasts.max() - contrasts.min())},
               "spread_of_tail_models": {"sd": float(tails.std(ddof=1)) if tails.size > 1 else None,
                                         "range": float(tails.max() - tails.min())}}
        if t == PRIMARY:
            row["primary_verdict"] = verdict(mean_ci)
            row["registered_expectation_met"] = bool(
                (contrasts < 0).all() and contrasts.std(ddof=1) < abs(contrasts.mean()))
        report["thresholds"][str(t)] = row

        print(f"\n===== {t} mm: {row['events']:,} station-days over")
        for l, m in models.items():
            print(f"  {l:18s} calibrated {m['brier_calibrated_cv']:.5f}  raw {m['brier']:.5f}  bias {m['frequency_bias']:.2f}")
        for name, d in per_pair.items():
            print(f"  {name:36s} {d['difference']:+.5f} [{d['ci_lower']:+.5f}, {d['ci_upper']:+.5f}]"
                  f"  {d['percent']:+.1f} % [{d['percent_ci'][0]:+.1f}, {d['percent_ci'][1]:+.1f}]")
        m = mean_ci
        print(f"  {'MEAN OVER PAIRS':36s} {m['difference']:+.5f} [{m['ci_lower']:+.5f}, {m['ci_upper']:+.5f}]"
              f"  {m['percent']:+.1f} % [{m['percent_ci'][0]:+.1f}, {m['percent_ci'][1]:+.1f}]")
        print(f"  pairs below zero {row['pairs_below_zero']} of {len(pairs)};"
              f" sd of contrasts {row['spread_of_contrasts']['sd']:.5f}; sd of tail models {row['spread_of_tail_models']['sd']:.5f}")
        if t == PRIMARY:
            print(f"  PRIMARY: {row['primary_verdict']}; registered expectation met: {row['registered_expectation_met']}")

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

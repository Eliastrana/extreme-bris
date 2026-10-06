#!/usr/bin/env python3
"""Where heavy-rain forecasts lose their points, and how much calibration recovers.

    scripts/hits_and_calibration.py \
      --model 'control2000=~/bris-runs/validation-forecasts/control2000/nordic_*.nc' \
      --model 'tailC2000=~/bris-runs/validation-forecasts/tailC2000/nordic_*.nc' \
      --model 'tailE2000=~/bris-runs/validation-forecasts/tailE2000/nordic_*.nc' \
      --comparator control2000 \
      --observations ~/bris-runs/observations-val \
      --out ~/bris-runs/round3/hits-calibration.json

EXPLORATORY, ON THE VALIDATION PERIOD. Written after variant E forecast heavy
rain about as often as it happens (occurrence ratio 0.92) and still scored 11
percent worse than the control arm on twCRPS at 20 mm. Two questions:

1. WHERE THE POINTS GO. Contingency tables at 10, 20 and 50 mm, with a "yes"
   forecast when at least k of the M members exceed (k = 2, the rule of the
   main evaluation, and k = 1). And twCRPS split by what happened: the part
   from gauge-days that did exceed the threshold (misses and intensity), and
   the part from those that did not, where any cost is forecast mass above
   the threshold (false alarms). The two parts add up to the twCRPS. Each
   part's difference from the comparator gets a paired seven-day block
   bootstrap.

2. HOW MUCH CALIBRATION RECOVERS. The ensemble probability of exceeding t
   takes M + 1 values. Each is mapped to the observed frequency by isotonic
   regression, cross-validated by week: every seven-day block is calibrated
   with a mapping fitted on all the other blocks. Brier score before and
   after. Calibration changes how much a model believes, not what it can tell
   apart; a gap that survives it is a gap in discrimination.

Same strict common cases as the screening: tail_calibration.collect.
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

import tail_calibration as tc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import paired_block_bootstrap, threshold_weighted_crps_cases  # noqa: E402

THRESHOLDS = [10.0, 20.0, 50.0]
YES_K = [2, 1]
LEAD = 30


def contingency(ens: np.ndarray, obs: np.ndarray, t: float, k: int) -> dict:
    yes = (ens > t).sum(axis=1) >= k
    event = obs > t
    hits = int((yes & event).sum())
    misses = int((~yes & event).sum())
    false_alarms = int((yes & ~event).sum())
    n_yes = hits + false_alarms
    n_event = hits + misses
    return {"hits": hits, "misses": misses, "false_alarms": false_alarms,
            "correct_negatives": int((~yes & ~event).sum()),
            "hit_rate": hits / n_event if n_event else float("nan"),
            "false_alarm_share": false_alarms / n_yes if n_yes else float("nan"),
            "frequency_bias": n_yes / n_event if n_event else float("nan")}


def isotonic(x: np.ndarray, y: np.ndarray) -> dict[float, float]:
    """Pool-adjacent-violators on the distinct values of x; returns value -> fit."""
    levels = np.unique(x)
    sums = [float(y[x == v].sum()) for v in levels]
    counts = [float((x == v).sum()) for v in levels]
    blocks = [[s, c, [v]] for s, c, v in zip(sums, counts, levels)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i] = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1],
                         blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return {float(v): b[0] / b[1] for b in blocks for v in b[2]}


def calibrated_cv(p: np.ndarray, event: np.ndarray, block: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    for b in np.unique(block):
        test = block == b
        mapping = isotonic(p[~test], event[~test])
        known = np.array(sorted(mapping))
        for v in np.unique(p[test]):
            # a value never seen in training takes the nearest seen value's fit
            nearest = known[np.argmin(np.abs(known - v))]
            out[test & (p == v)] = mapping[float(nearest)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    ap.add_argument("--comparator", required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2025-07-31")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    specs = dict(item.split("=", 1) for item in args.model)
    labels = list(specs)
    if args.comparator not in labels:
        raise SystemExit(f"comparator {args.comparator} is not among the models")
    plan = {"required_models": labels, "leads_hours": [LEAD], "forecast_accumulation": "per_step",
            "period": {"start": args.start, "end": args.end, "cycle_hour_utc": 0},
            "observations": {"max_quality": 4, "max_distance_km": 5.0}}
    cache = args.cache.expanduser() if args.cache else None
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cases = {k: data[k] for k in data.files}
    else:
        cases = tc.collect(plan, specs, args.observations.expanduser(), None)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **cases)

    o = cases[f"obs_{LEAD}"]
    dates = cases[f"dates_{LEAD}"].astype("datetime64[D]")
    block = ((dates - dates.min()).astype(int) // 7)
    ens = {l: cases[f"model_{l}_{LEAD}"] for l in labels}
    m = ens[labels[0]].shape[1]
    report = {"exploratory": True, "lead_hours": LEAD, "station_days": int(o.size), "members": int(m),
              "thresholds": {}}
    print(f"{o.size:,} station-days, {m} members, +{LEAD} h, comparator {args.comparator}")

    for t in THRESHOLDS:
        event = o > t
        res = {"events": int(event.sum()), "models": {}}
        tw = {l: threshold_weighted_crps_cases(ens[l], o, t) for l in labels}
        for l in labels:
            p = (ens[l] > t).mean(axis=1)
            cal = calibrated_cv(p, event.astype(float), block)
            res["models"][l] = {
                "contingency": {f"at_least_{k}_of_{m}": contingency(ens[l], o, t, k) for k in YES_K},
                "twcrps": float(tw[l].mean()),
                "twcrps_from_event_days": float(np.where(event, tw[l], 0.0).mean()),
                "twcrps_from_non_event_days": float(np.where(~event, tw[l], 0.0).mean()),
                "brier": float(((p - event) ** 2).mean()),
                "brier_calibrated_cv": float(((cal - event) ** 2).mean()),
                "calibration_map_all_data": isotonic(p, event.astype(float)),
            }
        for l in labels:
            if l == args.comparator:
                continue
            diffs = {}
            for name, mask in (("event_days", event), ("non_event_days", ~event)):
                d = np.where(mask, tw[l] - tw[args.comparator], 0.0)
                bt = paired_block_bootstrap(d, dates, block_days=7, replicates=args.replicates)
                diffs[name] = {"mean": bt["mean_difference"], "ci_lower": bt["ci_lower"], "ci_upper": bt["ci_upper"]}
            res["models"][l]["twcrps_parts_minus_comparator"] = diffs
        report["thresholds"][f"{t:g}"] = res

        print(f"\n===== {t:g} mm: {int(event.sum()):,} gauge-days over")
        print(f"  {'model':12s} {'hits':>5s} {'miss':>5s} {'f.al.':>6s} {'hit rate':>8s} {'f.al. share':>11s} {'bias':>5s}   (yes = 2 of {m}; 1 of {m} in brackets)")
        for l in labels:
            c2 = res["models"][l]["contingency"][f"at_least_2_of_{m}"]
            c1 = res["models"][l]["contingency"][f"at_least_1_of_{m}"]
            print(f"  {l:12s} {c2['hits']:5d} {c2['misses']:5d} {c2['false_alarms']:6d} {c2['hit_rate']:8.1%} "
                  f"{c2['false_alarm_share']:11.1%} {c2['frequency_bias']:5.2f}   "
                  f"[{c1['hit_rate']:.1%} / {c1['false_alarm_share']:.1%} / {c1['frequency_bias']:.2f}]")
        print(f"  twCRPS split:  {'total':>8s} {'event days':>11s} {'non-event days':>15s}")
        for l in labels:
            r = res["models"][l]
            print(f"  {l:12s} {r['twcrps']:8.4f} {r['twcrps_from_event_days']:11.4f} {r['twcrps_from_non_event_days']:15.4f}")
        for l in labels:
            if l == args.comparator:
                continue
            d = res["models"][l]["twcrps_parts_minus_comparator"]
            print(f"    {l} - {args.comparator}: event days {d['event_days']['mean']:+.4f} "
                  f"[{d['event_days']['ci_lower']:+.4f}, {d['event_days']['ci_upper']:+.4f}]   "
                  f"non-event days {d['non_event_days']['mean']:+.4f} "
                  f"[{d['non_event_days']['ci_lower']:+.4f}, {d['non_event_days']['ci_upper']:+.4f}]")
        print(f"  Brier, raw -> calibrated (week-out cross-validation):")
        for l in labels:
            r = res["models"][l]
            mp = ", ".join(f"{k:.2f}->{v:.3f}" for k, v in sorted(r["calibration_map_all_data"].items()))
            print(f"  {l:12s} {r['brier']:.5f} -> {r['brier_calibrated_cv']:.5f} "
                  f"({(r['brier_calibrated_cv'] / r['brier'] - 1):+.1%})   map: {mp}")

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the frozen round-2 screening plan to validation forecasts.

    scripts/screen_variants.py \
      --model 'control2000=~/bris-runs/validation-forecasts/control2000/nordic_*.nc' \
      --model 'tailA2000=~/bris-runs/validation-forecasts/tailA2000/nordic_*.nc' \
      --model 'tailB2000=~/bris-runs/validation-forecasts/tailB2000/nordic_*.nc' \
      --model 'tail2000=~/bris-runs/validation-forecasts/tail2000/nordic_*.nc' \
      --observations ~/bris-runs/observations-val \
      --out ~/bris-runs/tail-variants/screening.json

WHAT IT DECIDES, and only as evaluation/screening_round2_plan.json says, at
the plan's primary lead, every candidate against the comparator:

  gate    spread-skill ratio difference. Out if the point estimate is below
          the plan's floor or the whole interval is below zero.
  rank    fair twCRPS at 20 mm per day. Passes at or below zero.
  guard   fair CRPS over all cases. Out if more than the plan's tolerance worse.

A candidate that clears all three passes; the winner is the passing one with
the lowest twCRPS, ties going to the lower weight. The reference model, the
known failure, is put through the same rules so the gate is seen to work.
Everything else the plan lists is reported and decides nothing.

The plan's SHA-256 goes into the report, and the rules are read from constants
here that mirror the plan rather than parsed from its prose, so a changed plan
changes the SHA and a reader can check the two agree.

THE SAME CASES as every other score in this project: tail_calibration.collect,
which uses the evaluation's readers and strict common-case rule. There is no
MEPS for the validation period, so a zero field stands in for it in that rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402

import tail_calibration as tc  # noqa: E402
from spread_skill import AMOUNT_BINS, stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import (  # noqa: E402
    fair_crps_cases,
    paired_block_bootstrap,
    threshold_weighted_crps_cases,
)

# The plan's rules, as numbers. Change the plan and these together, never one.
GATE_FLOOR = -0.05
GUARD_TOLERANCE = 0.01
RANK_THRESHOLD = 20.0
TIE = 0.002
REPORTED_TWCRPS = [50.0]
CALIBRATION_THRESHOLDS = [10.0, 20.0, 50.0]
# Lower weight wins a tie; the plan's candidates, lowest dose first.
TIE_ORDER = ["tailB2000", "tailA2000"]


def ratio_bootstrap(ens_a, ens_b, obs, dates, replicates: int, seed: int) -> dict:
    """Spread-skill ratio difference with seven-day calendar blocks."""
    day0 = dates.min()
    block = (dates - day0).astype(int) // 7
    blocks = np.unique(block)
    index = {b: np.flatnonzero(block == b) for b in blocks}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        pick = np.concatenate([index[b] for b in rng.choice(blocks, size=blocks.size, replace=True)])
        draws.append(stats(ens_a[pick], obs[pick])["ratio"] - stats(ens_b[pick], obs[pick])["ratio"])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    point = stats(ens_a, obs)["ratio"] - stats(ens_b, obs)["ratio"]
    return {"difference": float(point), "ci_lower": float(lo), "ci_upper": float(hi), "blocks": int(blocks.size)}


def paired(case_a, case_b, dates, replicates: int) -> dict:
    out = paired_block_bootstrap(case_a - case_b, dates, block_days=7, replicates=replicates)
    out["first"] = float(np.mean(case_a))
    out["second"] = float(np.mean(case_b))
    return out


def score_lead(lead: int, cases: dict, labels: list[str], comparator: str, others: list[str],
               replicates: int, rng) -> dict:
    o = cases[f"obs_{lead}"]
    dates = cases[f"dates_{lead}"].astype("datetime64[D]")
    ens = {l: cases[f"model_{l}_{lead}"] for l in labels}
    out = {"cases": int(o.size), "models": {}, "versus_comparator": {}}
    for l in labels:
        s = stats(ens[l], o)
        s["fair_crps"] = float(fair_crps_cases(ens[l], o).mean())
        for t in [RANK_THRESHOLD] + REPORTED_TWCRPS:
            s[f"twcrps_{t:g}"] = float(threshold_weighted_crps_cases(ens[l], o, t).mean())
        mean = ens[l].mean(axis=1)
        s["ratio_by_forecast_amount"] = []
        for lo, hi in AMOUNT_BINS:
            sel = (mean >= lo) & (mean < hi)
            if sel.sum() >= 50:
                b = stats(ens[l][sel], o[sel])
                s["ratio_by_forecast_amount"].append({"lo": lo, "hi": None if np.isinf(hi) else hi,
                                                      "ratio": b["ratio"], "cases": b["cases"]})
        s["tail_calibration"] = {}
        for t in CALIBRATION_THRESHOLDS:
            pc = tc.per_case(ens[l], o, t, rng)
            summary = tc.tail_summary(pc["p"], pc["hit"], pc["cpit"], pc["none_above"])
            s["tail_calibration"][f"{t:g}"] = {k: summary[k] for k in
                                              ("observed", "O_t", "share_no_member_above_t", "CPIT_mean_k1")}
        out["models"][l] = s

    base = ens[comparator]
    crps_base = fair_crps_cases(base, o)
    for l in others:
        crps = fair_crps_cases(ens[l], o)
        v = {"spread_skill_ratio": ratio_bootstrap(ens[l], base, o, dates, replicates, 20260927),
             "fair_crps": paired(crps, crps_base, dates, replicates),
             "fair_crps_relative": float(crps.mean() / crps_base.mean() - 1.0)}
        for t in [RANK_THRESHOLD] + REPORTED_TWCRPS:
            v[f"twcrps_{t:g}"] = paired(threshold_weighted_crps_cases(ens[l], o, t),
                                        threshold_weighted_crps_cases(base, o, t), dates, replicates)
        out["versus_comparator"][l] = v
    return out


def decide(primary: dict, candidates: list[str], reference: list[str]) -> dict:
    verdicts = {}
    for l in candidates + reference:
        v = primary["versus_comparator"][l]
        r = v["spread_skill_ratio"]
        gate_ok = not (r["difference"] < GATE_FLOOR or r["ci_upper"] < 0.0)
        rank_ok = v[f"twcrps_{RANK_THRESHOLD:g}"]["mean_difference"] <= 0.0
        guard_ok = v["fair_crps_relative"] <= GUARD_TOLERANCE
        verdicts[l] = {"gate_spread": gate_ok, "ranking_metric": rank_ok, "guard_ordinary_weather": guard_ok,
                       "passes": gate_ok and rank_ok and guard_ok, "is_reference": l in reference}
    passing = [l for l in candidates if verdicts[l]["passes"]]
    winner = None
    if passing:
        tw = {l: primary["versus_comparator"][l][f"twcrps_{RANK_THRESHOLD:g}"]["first"] for l in passing}
        best = min(tw.values())
        tied = [l for l in passing if tw[l] - best <= TIE]
        winner = sorted(tied, key=lambda l: TIE_ORDER.index(l) if l in TIE_ORDER else len(TIE_ORDER))[0]
    return {"verdicts": verdicts, "passing": passing, "winner": winner}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=root / "evaluation" / "screening_round2_plan.json")
    ap.add_argument("--replicates", type=int, default=None, help="default: the plan's")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if "frozen_at" not in plan:
        raise SystemExit(f"{args.plan} is not frozen; freeze it before scoring anything")
    comparator = plan["comparator"]
    candidates = list(plan["candidates"])
    reference = list(plan.get("reference", {}))
    labels = [comparator] + candidates + reference
    specs = dict(item.split("=", 1) for item in args.model)
    if set(specs) != set(labels):
        raise SystemExit(f"need models {labels}, got {sorted(specs)}")
    replicates = args.replicates or int(plan["bootstrap"]["replicates"])

    collect_plan = dict(plan, required_models=labels)
    cache = args.cache.expanduser() if args.cache else None
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cases = {k: data[k] for k in data.files}
        print(f"read cached cases from {cache}")
    else:
        cases = tc.collect(collect_plan, specs, args.observations.expanduser(), None)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **cases)

    rng = np.random.default_rng(20260927)
    leads = [int(v) for v in plan["leads_hours"]]
    primary_lead = int(plan["primary_lead_hours"])
    report = {"plan": str(args.plan), "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
              "frozen_at": plan["frozen_at"], "leads": {}}
    for lead in leads:
        report["leads"][str(lead)] = score_lead(lead, cases, labels, comparator,
                                                candidates + reference, replicates, rng)
    primary = report["leads"][str(primary_lead)]
    days = np.unique(cases[f"dates_{primary_lead}"].astype("datetime64[D]")).size
    report["cycles_scored"] = int(days)
    report["decision"] = decide(primary, candidates, reference)

    print(f"plan {report['plan_sha256'][:12]}, frozen {plan['frozen_at']}, "
          f"{days} forecast days, {primary['cases']:,} station-days at +{primary_lead} h")
    print(f"\n{'model':12s} {'ratio':>6s} {'fCRPS':>7s} {'tw20':>7s} {'tw50':>7s} {'O_t 20':>7s} {'miss 20':>8s}")
    for l in labels:
        s = primary["models"][l]
        c = s["tail_calibration"]["20"]
        print(f"{l:12s} {s['ratio']:6.3f} {s['fair_crps']:7.4f} {s['twcrps_20']:7.4f} {s['twcrps_50']:7.4f} "
              f"{c['O_t']:7.2f} {c['share_no_member_above_t']:8.1%}")
    print(f"\nagainst {comparator} at +{primary_lead} h:")
    for l in candidates + reference:
        v = primary["versus_comparator"][l]
        r, w = v["spread_skill_ratio"], v["twcrps_20"]
        d = report["decision"]["verdicts"][l]
        tag = "reference" if d["is_reference"] else ("PASSES" if d["passes"] else "out")
        print(f"  {l:12s} ratio {r['difference']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] "
              f"{'ok' if d['gate_spread'] else 'FAIL'}   "
              f"tw20 {w['mean_difference']:+.4f} [{w['ci_lower']:+.4f}, {w['ci_upper']:+.4f}] "
              f"{'ok' if d['ranking_metric'] else 'FAIL'}   "
              f"CRPS {v['fair_crps_relative']:+.1%} {'ok' if d['guard_ordinary_weather'] else 'FAIL'}   -> {tag}")
    print(f"\nwinner: {report['decision']['winner'] or 'none; the plan says no full-length run'}")

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

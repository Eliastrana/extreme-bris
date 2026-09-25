#!/usr/bin/env python3
"""Tail calibration of the evaluation forecasts, after Wessel et al. (2026).

    scripts/tail_calibration.py \
      --model 'baseline=~/bris-runs/evaluation/baseline/nordic_*.nc' \
      --model 'control=~/bris-runs/evaluation/control/nordic_*.nc' \
      --model 'tail=~/bris-runs/evaluation/tail/nordic_*.nc' \
      --meps ~/bris-runs/meps/precipitation_daily.npz \
      --observations ~/bris-runs/observations \
      --cache ~/bris-runs/evaluation-score/cases.npz \
      --out ~/bris-runs/evaluation-score/tail-calibration.json

EXPLORATORY, AND SAID SO. Written after the frozen evaluation had returned its
answer, prompted by Wessel, Schillinger, Kwasniok and Allen, "Enforcing tail
calibration when training probabilistic forecast models", IJF 42 (2026). They
train with the same CRPS + gamma * twCRPS loss as the tail arm and find that
twCRPS itself barely moves, while tail calibration improves a lot. This asks
whether the tail arm is worse on their measure too, or only on twCRPS. It is
not part of evaluation_plan.json and cannot change the primary result.

THE SAME CASES as the evaluation: readers and the strict common-case rule are
imported from evaluate_experiment.py.

WHAT IT MEASURES, per model, lead and threshold t (their section 3 and 4.2):
  O_t      occurrence ratio: observed exceedances over the forecast expected
           number, sum over cases of the share of members above t. Above 1 the
           ensemble forecasts too few exceedances (tail too light), below 1 too
           many. For MEPS, which is one member, this is the frequency bias.
  CPIT     for cases where the gauge exceeded t, where the gauge falls in the
           forecast distribution above t: (F(y) - F(t)) / (1 - F(t)). Uniform
           if the excess distribution is calibrated; piled at 1 if the tail is
           too light. When no member exceeds t, F(t) = 1 and the value is 1,
           their convention: the event was entirely outside the forecast.
  Q_t(u)   O_t times the u-quantile of the CPIT values. Lies on the diagonal
           for a tail-calibrated forecast; above it the tail is too light.
  TMCB     mean |Q_t(u) - u| over the sorted CPIT values. Their tail
           miscalibration, lower is better. WITH FOUR MEMBERS, READ WITH CARE:
           on synthetic data a calibrated ensemble scores 0.24 and one 1.6
           times too wide 0.17, because the pile of ones and an O_t below 1
           cancel, the cancellation their appendix A describes. Reported, not
           relied on.
  CPIT_mean_k1, CPIT_MCB_k1
           the conditional PIT only where at least one member is above t.
           Exactly uniform for a calibrated ensemble, so mean 0.5 and MCB 0,
           with no floor. Above 0.5: gauges sit high among the members that
           do exceed t, the excess distribution is too light.
  share_no_member_above_t
           of the gauge exceedances, how many no member reached at all.
  MCB      the same for the ordinary PIT over all cases: overall calibration.

FOUR MEMBERS IS COARSE. PITs are rank-based: the gauge's rank among the
members plus a uniform draw, over M + 1, which is exactly uniform for a
calibrated ensemble (the step-function CDF of four members is not). The
conditional PIT is the same among the k members above t. When no member is
above t the value is 1, the paper's convention, and a calibrated ensemble of
four still puts a fair share there, so TMCB has a floor well above zero.
O_t has no such floor. Everything is comparable between models with the same
member count, which the three arms have; none of it is comparable with the
paper's 250-sample numbers. share_no_member_above_t is reported so the floor
can be read directly.

Uncertainty on differences between models comes from the evaluation's kind of
bootstrap: whole seven-day calendar blocks resampled.
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

import evaluate_experiment as ev  # noqa: E402

THRESHOLDS_MM = [10.0, 20.0, 50.0]
Q_GRID = [round(0.1 * k, 1) for k in range(1, 11)]


def collect(plan: dict, specs: dict, observations: Path, meps_path: Path | None) -> dict:
    """Every strict common case, per lead: obs, dates, MEPS and each model's members."""
    labels = list(plan["required_models"])
    op = plan["observations"]
    obs, _ = ev.load_screened_observations(observations, op["max_quality"])
    leads = [int(v) for v in plan["leads_hours"]]
    dist = float(op["max_distance_km"])
    indexes = {l: ev.index_model_files(specs[l]) for l in labels}
    readers = {l: ev.BrisReader(obs, dist, plan["forecast_accumulation"]) for l in labels}
    meps = ev.MepsReader(meps_path, obs, dist) if meps_path else None
    period = plan["period"]
    cycles = ev.expected_cycles(period["start"], period["end"], int(period["cycle_hour_utc"]))

    got = {lead: {"obs": [], "dates": [], "meps": [], "models": {l: [] for l in labels}} for lead in leads}
    for n, cycle in enumerate(cycles, 1):
        if not all(cycle in indexes[l] for l in labels) or (meps and not meps.has_cycle(cycle)):
            continue
        daily = {l: readers[l].read(indexes[l][cycle], leads)[0] for l in labels}
        meps_daily = meps.read(cycle, leads) if meps else np.zeros((len(leads), len(obs["stations"])))
        for li, lead in enumerate(leads):
            valid = np.datetime64(cycle) + np.timedelta64(lead, "h")
            truth = ev.observation_for_time(obs, valid)
            if truth is None:
                continue
            common = ev.strict_common_mask(truth, meps_daily[li], [daily[l][li] for l in labels])
            if not common.any():
                continue
            slot = got[lead]
            slot["obs"].append(truth[common])
            slot["dates"].append(np.full(int(common.sum()), valid.astype("datetime64[D]")))
            slot["meps"].append(meps_daily[li][common])
            for l in labels:
                slot["models"][l].append(daily[l][li][:, common].T)
        if n % 50 == 0:
            print(f"  {n}/{len(cycles)} cycles read", flush=True)

    flat = {}
    for lead in leads:
        slot = got[lead]
        flat[f"obs_{lead}"] = np.concatenate(slot["obs"])
        flat[f"dates_{lead}"] = np.concatenate(slot["dates"])
        flat[f"meps_{lead}"] = np.concatenate(slot["meps"])
        for l in labels:
            flat[f"model_{l}_{lead}"] = np.concatenate(slot["models"][l])
    return flat


def randomised_rank(ens: np.ndarray, x: np.ndarray, rng, above: float | None = None) -> np.ndarray:
    """Where x falls among the members, as a value in (0, 1).

    The rank of x among the members plus a uniform draw, over M + 1: exactly
    uniform when x and the members are exchangeable, which the empirical
    step-function CDF of four members is not. Ties, plentiful at zero, are
    spread across their slots. With `above`, only members above that value
    count, which gives the conditional PIT among the k members above t.
    """
    keep = np.ones(ens.shape, bool) if above is None else ens > above
    below = ((ens < x[:, None]) & keep).sum(axis=1)
    ties = ((ens == x[:, None]) & keep).sum(axis=1)
    k = keep.sum(axis=1)
    return (below + rng.random(x.size) * (ties + 1)) / (k + 1)


def per_case(ens: np.ndarray, obs: np.ndarray, t: float, rng) -> dict:
    """What every summary needs, computed once so the bootstrap only resamples."""
    exceed_prob = (ens > t).mean(axis=1)
    hit = obs > t
    none_above = (ens[hit] > t).sum(axis=1) == 0
    cpit = randomised_rank(ens[hit], obs[hit], rng, above=t)
    # No member above t: F(t) = 1 and the excess distribution is a point mass,
    # so the paper's convention puts the value at 1. It is where a too-light
    # tail shows most, and it is not uniform even for a calibrated ensemble
    # of four, since four members often all stay below a rare threshold.
    cpit[none_above] = 1.0
    return {"p": exceed_prob, "hit": hit, "cpit": cpit, "none_above": none_above,
            "above_all": (ens[hit] < obs[hit][:, None]).all(axis=1)}


def tail_summary(p: np.ndarray, hit: np.ndarray, cpit: np.ndarray, none: np.ndarray) -> dict:
    """p, hit over all cases; cpit, none over the cases that exceeded t."""
    n_t = int(hit.sum())
    expected = float(p.sum())
    nan = float("nan")
    if n_t == 0 or expected == 0.0:
        return {"observed": n_t, "expected": expected, "O_t": nan, "share_no_member_above_t": nan,
                "CPIT_mean_k1": nan, "CPIT_MCB_k1": nan, "CPIT_hist_k1": [nan] * 5,
                "TMCB": nan, "CPIT_MCB": nan, "Q_t": [nan] * len(Q_GRID)}
    o_t = n_t / expected
    z = np.sort(cpit)
    u = np.arange(1, n_t + 1) / n_t
    inside = cpit[~none]
    return {"observed": n_t, "expected": expected, "O_t": o_t,
            "share_no_member_above_t": float(none.mean()),
            # Only where at least one member is above t: here the CPIT is
            # exactly uniform for a calibrated ensemble, mean 0.5, no floor.
            "CPIT_mean_k1": float(inside.mean()) if inside.size else nan,
            "CPIT_MCB_k1": mcb(inside) if inside.size else nan,
            "CPIT_hist_k1": (np.histogram(inside, bins=5, range=(0, 1))[0] / max(inside.size, 1)).tolist(),
            "TMCB": float(np.mean(np.abs(o_t * z - u))),
            "CPIT_MCB": float(np.mean(np.abs(z - u))),
            "Q_t": [float(o_t * np.quantile(z, q)) for q in Q_GRID]}


def mcb(pit: np.ndarray) -> float:
    z = np.sort(pit)
    return float(np.mean(np.abs(z - np.arange(1, z.size + 1) / z.size)))


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    ap.add_argument("--meps", type=Path, default=None)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=root / "evaluation" / "evaluation_plan.json")
    ap.add_argument("--pairs", default="tail-control,control-baseline")
    ap.add_argument("--thresholds", default=",".join(f"{v:g}" for v in THRESHOLDS_MM))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--cache", type=Path, default=None,
                    help="npz of the collected cases; written on the first run, read after that")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    labels = list(plan["required_models"])
    specs = dict(item.split("=", 1) for item in args.model)
    if set(specs) != set(labels):
        raise SystemExit(f"need models {labels}, got {sorted(specs)}")
    leads = [int(v) for v in plan["leads_hours"]]
    thresholds = [float(v) for v in args.thresholds.split(",")]

    cache = args.cache.expanduser() if args.cache else None
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cases = {k: data[k] for k in data.files}
        print(f"read cached cases from {cache}")
    else:
        cases = collect(plan, specs, args.observations.expanduser(),
                        args.meps.expanduser() if args.meps else None)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **cases)
            print(f"cached cases in {cache}")

    rng = np.random.default_rng(20260925)
    pairs = [tuple(p.split("-")) for p in args.pairs.split(",") if p]
    report = {"exploratory": True, "reference": "Wessel et al., IJF 42 (2026) 1336-1356",
              "cases_from_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
              "q_grid": Q_GRID, "leads": {}}
    for lead in leads:
        o = cases[f"obs_{lead}"]
        dates = cases[f"dates_{lead}"].astype("datetime64[D]")
        ens = {l: cases[f"model_{l}_{lead}"] for l in labels}
        meps = cases[f"meps_{lead}"]
        m = ens[labels[0]].shape[1]
        out = {"cases": int(o.size), "members": int(m), "MCB": {}, "thresholds": {}}
        print(f"\n===== +{lead} h: {o.size:,} station-days, {m} members")
        for l in labels:
            out["MCB"][l] = mcb(randomised_rank(ens[l], o, rng))
        print("  overall MCB (lower is better): "
              + "  ".join(f"{l} {out['MCB'][l]:.4f}" for l in labels))

        day0 = dates.min()
        block = (dates - day0).astype(int) // 7
        blocks = np.unique(block)
        index = {b: np.flatnonzero(block == b) for b in blocks}
        picks = [np.concatenate([index[b] for b in rng.choice(blocks, size=blocks.size, replace=True)])
                 for _ in range(args.replicates)]

        for t in thresholds:
            res = {"models": {}, "differences": {}}
            pc = {l: per_case(ens[l], o, t, rng) for l in labels}
            for l in labels:
                s = tail_summary(pc[l]["p"], pc[l]["hit"], pc[l]["cpit"], pc[l]["none_above"])
                s["share_above_all_members"] = float(pc[l]["above_all"].mean()) if pc[l]["above_all"].size else float("nan")
                res["models"][l] = s
            meps_expected = float((meps > t).sum())
            res["models"]["meps"] = {"observed": int((o > t).sum()), "expected": meps_expected,
                                     "O_t": float((o > t).sum() / meps_expected) if meps_expected else float("nan")}

            # CPIT values sit on the cases that exceeded t; carry their positions.
            hit_pos = np.flatnonzero(o > t)
            cpit_at = {l: np.full(o.size, np.nan) for l in labels}
            none_at = {l: np.zeros(o.size, bool) for l in labels}
            for l in labels:
                cpit_at[l][hit_pos] = pc[l]["cpit"]
                none_at[l][hit_pos] = pc[l]["none_above"]
            names = ("O_t", "share_no_member_above_t", "CPIT_mean_k1", "TMCB")
            for first, second in pairs:
                draws = {name: [] for name in names}
                for pick in picks:
                    h = pc[first]["hit"][pick]
                    x = {l: tail_summary(pc[l]["p"][pick], h, cpit_at[l][pick][h], none_at[l][pick][h])
                         for l in (first, second)}
                    for name in names:
                        draws[name].append(x[first][name] - x[second][name])
                diff = {}
                for name in names:
                    point = res["models"][first][name] - res["models"][second][name]
                    lo, hi = np.nanpercentile(draws[name], [2.5, 97.5])
                    diff[name] = {"difference": point, "ci_lower": float(lo), "ci_upper": float(hi)}
                res["differences"][f"{first}-{second}"] = diff

            out["thresholds"][f"{t:g}"] = res
            print(f"\n  t = {t:g} mm: {res['models'][labels[0]]['observed']:,} observed exceedances")
            print(f"  {'model':9s} {'O_t':>6s} {'none>t':>7s} {'CPITk1':>7s} {'MCBk1':>6s} {'>all':>6s} {'TMCB':>6s}"
                  "   CPIT histogram where some member > t (0.2 each if calibrated)")
            for l in labels:
                s = res["models"][l]
                print(f"  {l:9s} {s['O_t']:6.2f} {s['share_no_member_above_t']:7.1%} {s['CPIT_mean_k1']:7.3f} "
                      f"{s['CPIT_MCB_k1']:6.3f} {s['share_above_all_members']:6.1%} {s['TMCB']:6.3f}   "
                      + " ".join(f"{v:.2f}" for v in s["CPIT_hist_k1"]))
            print(f"  {'meps':9s} {res['models']['meps']['O_t']:6.2f}   (one member: frequency bias only)")
            for pair, diff in res["differences"].items():
                print(f"    {pair}: " + "   ".join(
                    f"{name} {d['difference']:+.3f} [{d['ci_lower']:+.3f}, {d['ci_upper']:+.3f}]"
                    for name, d in diff.items()))
        report["leads"][str(lead)] = out

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

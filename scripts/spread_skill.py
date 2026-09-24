#!/usr/bin/env python3
"""Spread-skill check on the evaluation forecasts: is each ensemble honest?

    scripts/spread_skill.py \
      --model 'baseline=~/bris-runs/evaluation/baseline/nordic_*.nc' \
      --model 'control=~/bris-runs/evaluation/control/nordic_*.nc' \
      --model 'tail=~/bris-runs/evaluation/tail/nordic_*.nc' \
      --meps ~/bris-runs/meps/precipitation_daily.npz \
      --observations ~/bris-runs/observations \
      --out ~/bris-runs/evaluation-score/spread-skill.json

EXPLORATORY, AND SAID SO. This was written after the frozen evaluation had
returned its answer, to test one explanation of it: that the tail arm's four
members agree with each other too much. It cannot change the primary result,
and it is not part of evaluation_plan.json.

THE SAME CASES. The readers and the strict common-case rule are imported from
evaluate_experiment.py, so every number here is computed on exactly the
station-days the evaluation scored: all three arms, MEPS and a screened gauge
present, same leads, same windows.

WHAT IT MEASURES, per model and lead:
  spread   square root of the mean unbiased variance across the members
  error    RMSE of the ensemble mean against the gauge
  ratio    spread * sqrt((M+1)/M) / error. A calibrated M-member ensemble
           gives 1: the ensemble mean of M exchangeable draws misses the truth
           by sqrt((M+1)/M) times the member spread on average. Below 1 is
           overconfident, above 1 underconfident.
  ranks    where the gauge falls among the sorted members, M+1 positions.
           Flat is honest; heavy ends are overconfident. Ties, which rain has
           plenty of at zero, are broken at random.
  by amount  the same ratio in bins of the forecast ensemble mean. Binned on
           the forecast, never the observation, so the check does not become
           the forecaster's dilemma it is trying to read around.

Uncertainty on the tail-minus-control ratio comes from the same kind of
bootstrap the evaluation uses: whole seven-day calendar blocks resampled.
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

AMOUNT_BINS = [(0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)]


def stats(ens: np.ndarray, obs: np.ndarray) -> dict:
    """ens: cases x members."""
    m = ens.shape[1]
    var = ens.var(axis=1, ddof=1)
    mean = ens.mean(axis=1)
    spread = float(np.sqrt(var.mean()))
    error = float(np.sqrt(((mean - obs) ** 2).mean()))
    return {"cases": int(obs.size), "spread": spread, "rmse": error,
            "ratio": spread * np.sqrt((m + 1) / m) / error if error else float("nan")}


def rank_histogram(ens: np.ndarray, obs: np.ndarray, rng) -> list[float]:
    m = ens.shape[1]
    below = (ens < obs[:, None]).sum(axis=1)
    ties = (ens == obs[:, None]).sum(axis=1)
    rank = below + np.floor(rng.random(obs.size) * (ties + 1)).astype(int)
    counts = np.bincount(rank, minlength=m + 1)
    return (counts / counts.sum()).tolist()


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    ap.add_argument("--meps", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=root / "evaluation" / "evaluation_plan.json")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = json.loads(args.plan.read_bytes())
    labels = list(plan["required_models"])
    specs = dict(item.split("=", 1) for item in args.model)
    if set(specs) != set(labels):
        raise SystemExit(f"need models {labels}, got {sorted(specs)}")
    op = plan["observations"]
    obs, _ = ev.load_screened_observations(args.observations.expanduser(), op["max_quality"])
    leads = [int(v) for v in plan["leads_hours"]]
    dist = float(op["max_distance_km"])
    indexes = {l: ev.index_model_files(specs[l]) for l in labels}
    readers = {l: ev.BrisReader(obs, dist, plan["forecast_accumulation"]) for l in labels}
    meps = ev.MepsReader(args.meps.expanduser(), obs, dist)
    period = plan["period"]
    cycles = ev.expected_cycles(period["start"], period["end"], int(period["cycle_hour_utc"]))

    got = {lead: {"obs": [], "dates": [], "models": {l: [] for l in labels}} for lead in leads}
    for n, cycle in enumerate(cycles, 1):
        if not all(cycle in indexes[l] for l in labels) or not meps.has_cycle(cycle):
            continue
        daily = {l: readers[l].read(indexes[l][cycle], leads)[0] for l in labels}
        meps_daily = meps.read(cycle, leads)
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
            for l in labels:
                slot["models"][l].append(daily[l][li][:, common].T)
        if n % 50 == 0:
            print(f"  {n}/{len(cycles)} cycles read", flush=True)

    rng = np.random.default_rng(20260924)
    report = {"exploratory": True,
              "cases_from_plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "leads": {}}
    for lead in leads:
        slot = got[lead]
        o = np.concatenate(slot["obs"])
        dates = np.concatenate(slot["dates"])
        ens = {l: np.concatenate(slot["models"][l]) for l in labels}
        out = {"cases": int(o.size), "models": {}}
        print(f"\n===== +{lead} h: {o.size:,} station-days (same strict common cases as the evaluation)")
        print(f"{'model':9s} {'spread':>8s} {'rmse':>8s} {'ratio':>7s}   rank histogram (1/{ens[labels[0]].shape[1]+1} each if honest)")
        for l in labels:
            s = stats(ens[l], o)
            s["rank_histogram"] = rank_histogram(ens[l], o, rng)
            s["by_forecast_amount"] = []
            mean = ens[l].mean(axis=1)
            for lo, hi in AMOUNT_BINS:
                sel = (mean >= lo) & (mean < hi)
                if sel.sum() >= 50:
                    b = stats(ens[l][sel], o[sel])
                    b.update(lo=lo, hi=None if np.isinf(hi) else hi)
                    s["by_forecast_amount"].append(b)
            out["models"][l] = s
            print(f"{l:9s} {s['spread']:8.3f} {s['rmse']:8.3f} {s['ratio']:7.3f}   "
                  + " ".join(f"{v:.3f}" for v in s["rank_histogram"]))
        print("  ratio by forecast ensemble-mean amount (mm):")
        for l in labels:
            parts = []
            for b in out["models"][l]["by_forecast_amount"]:
                upper = "" if b["hi"] is None else f"{b['hi']:g}"
                parts.append(f"[{b['lo']:g}-{upper}] {b['ratio']:.2f} (n={b['cases']})")
            print(f"    {l:9s} " + "  ".join(parts))

        # tail - control ratio, seven-day calendar blocks
        day0 = dates.min()
        block = ((dates - day0).astype(int) // 7)
        blocks = np.unique(block)
        index = {b: np.flatnonzero(block == b) for b in blocks}
        diffs = []
        for _ in range(args.replicates):
            pick = np.concatenate([index[b] for b in rng.choice(blocks, size=blocks.size, replace=True)])
            diffs.append(stats(ens["tail"][pick], o[pick])["ratio"] - stats(ens["control"][pick], o[pick])["ratio"])
        lo_ci, hi_ci = np.percentile(diffs, [2.5, 97.5])
        d = out["models"]["tail"]["ratio"] - out["models"]["control"]["ratio"]
        out["tail_minus_control_ratio"] = {"difference": d, "ci_lower": float(lo_ci), "ci_upper": float(hi_ci),
                                           "blocks": int(blocks.size), "replicates": args.replicates}
        print(f"  tail - control ratio: {d:+.3f}  (95% block bootstrap {lo_ci:+.3f} to {hi_ci:+.3f}, {blocks.size} blocks)")
        report["leads"][str(lead)] = out

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

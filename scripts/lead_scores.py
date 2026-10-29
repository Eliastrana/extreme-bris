#!/usr/bin/env python3
"""Six-hour rain, lead by lead: does the heavy-rain gain hold from +6 to +36 h?

    scripts/lead_scores.py \\
      --model 'baseline=~/bris-runs/evaluation/baseline/nordic_*.nc' \\
      --model 'controlfft2000=~/bris-runs/evaluation/controlfft2000/nordic_*.nc' \\
      --model 'tailC2000=~/bris-runs/evaluation/tailC2000/nordic_*.nc' \\
      --comparator controlfft2000 --primary tailC2000 \\
      --meps-members ~/bris-runs/meps-ens \\
      --observations ~/bris-runs/observations \\
      --cache ~/bris-runs/leads/cases.npz --out ~/bris-runs/leads/report.json

THE PLAN is evaluation/lead_plan.json, frozen before this was first run.

WHY. The arms train with a rollout of one step and Bris with six. Every
heavy-rain score so far is one 24 h total ending at +30 h, which mixes the
steps ending at +12 to +30 h. Scored step by step, a gain that came from the
one step the loss rewards would show as largest early and shrinking.

TRUTH. The hourly gauges, summed over the six hours ending at each valid time,
complete windows only, gauges check_gauges.py flagged left out. The daily
climate gauges cannot be split into six-hour pieces, so this is the hourly
network, a different and noisier one.

CASES. A gauge and cycle is kept only if the observation and every model are
present at every lead, so all six leads are scored on the same gauge-cycles
and a difference between +12 and +36 h is a paired one. A second set adds the
fifteen MEPS members, present at every lead too; their files hold only the
daily-network gauges, so it has fewer stations.

SCORES, per lead and threshold: Brier score raw and after month-out isotonic
calibration (fitted per model, lead and threshold), skill against the
climatological base rate, ROC area, and fair CRPS of all rain. Differences
against the comparator get the paired seven-day block bootstrap, blocks of
cycle dates, so every lead of a cycle moves together.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402

import evaluate_experiment as ev  # noqa: E402
from hits_and_calibration import calibrated_cv, roc_area  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import fair_crps_cases, paired_block_bootstrap  # noqa: E402
from xbris.stations import accumulate, load_observations  # noqa: E402

LEADS = [6, 12, 18, 24, 30, 36]
WINDOW = 6
THRESHOLDS = [5.0, 10.0, 20.0]
MEPS_SETS = {"meps4": [0, 1, 2, 9], "meps15": list(range(15))}
MAX_DISTANCE_KM = 5.0
START, END = "2025-08-01", "2026-09-06"

_OBS: dict = {}


def hourly_truth(path: Path) -> tuple[dict, list[str]]:
    obs = load_observations(path / "precipitation.npz")
    flagged = json.loads((path / "precipitation_check.json").read_text())["flagged"]
    drop = np.isin(obs["stations"].astype(str), list(flagged))
    obs["values"] = np.array(obs["values"], dtype="float64", copy=True)
    obs["values"][drop] = np.nan
    return obs, sorted(str(s) for s in obs["stations"][drop])


def read_model(job: tuple[str, str, list[str]]) -> tuple[str, np.ndarray]:
    """One model's six-hour amounts at the gauges: cycle, lead, station, member."""
    label, spec, cycles = job
    index = ev.index_model_files(spec)
    reader = ev.BrisReader(_OBS, MAX_DISTANCE_KM, "per_step", window_hours=WINDOW)
    out = None
    for n, cycle in enumerate(cycles, 1):
        if cycle not in index:
            continue
        sums, _ = reader.read(index[cycle], LEADS)          # lead, member, station
        if out is None:
            out = np.full((len(cycles), len(LEADS), sums.shape[2], sums.shape[1]), np.nan, dtype="float32")
        out[n - 1] = sums.transpose(0, 2, 1)
        if n % 100 == 0:
            print(f"  {label}: {n}/{len(cycles)} cycles", flush=True)
    if out is None:
        raise SystemExit(f"{label}: no forecast file in the period")
    return label, out


def collect(specs: dict, observations: Path, members_dir: Path, start: str, end: str) -> dict:
    obs, flagged = hourly_truth(observations)
    _OBS.update(obs)
    cycles = ev.expected_cycles(start, end, 0)
    print(f"{len(cycles)} cycles, {len(obs['stations'])} hourly gauges, {len(flagged)} flagged and left out", flush=True)

    starts = np.array([np.datetime64(c) for c in cycles], dtype="datetime64[s]")
    valid = (starts[:, None] + np.array(LEADS, dtype="timedelta64[h]")[None, :]).astype("datetime64[s]")
    truth = accumulate(obs["values"], obs["times"], valid.ravel(), WINDOW)          # station, cycle*lead
    truth = truth.T.reshape(len(cycles), len(LEADS), -1)                            # cycle, lead, station

    with mp.get_context("fork").Pool(len(specs)) as pool:
        models = dict(pool.map(read_model, [(l, s, cycles) for l, s in specs.items()]))

    wanted = sorted({m for ms in MEPS_SETS.values() for m in ms})
    meps = {}
    for m in wanted:
        reader = ev.MepsReader(members_dir / f"mbr{m:03d}.npz", obs, MAX_DISTANCE_KM, window_hours=WINDOW)
        arr = np.full(truth.shape, np.nan, dtype="float32")
        for ci, cycle in enumerate(cycles):
            key = ev.cycle_key(np.datetime64(cycle))
            if reader.has_cycle(key):
                arr[ci] = reader.read(key, LEADS)
        meps[m] = arr

    # a gauge-cycle counts only if everything is present at every lead
    keep = np.isfinite(truth).all(axis=1)
    for arr in models.values():
        keep &= np.isfinite(arr).all(axis=(1, 3))
    with_meps = keep.copy()
    for arr in meps.values():
        with_meps &= np.isfinite(arr).all(axis=1)
    ci, si = np.nonzero(keep)
    print(f"{ci.size:,} gauge-cycles with every model at every lead; "
          f"{with_meps.sum():,} of them with all fifteen MEPS members", flush=True)

    cases = {"stations": np.asarray(obs["stations"]), "flagged": np.array(flagged),
             "leads": np.array(LEADS), "cycle": starts[ci].astype("datetime64[D]"), "station": si,
             "with_meps": with_meps[ci, si], "obs": truth[ci, :, si].T}      # obs: lead, case
    for label, arr in models.items():
        cases[f"model_{label}"] = arr[ci, :, si, :].transpose(1, 0, 2)    # lead, case, member
    for m, arr in meps.items():
        cases[f"meps_{m:03d}"] = arr[ci, :, si].T
    return cases


def ensembles(cases: dict, labels: list[str]) -> dict[str, np.ndarray]:
    ens = {l: cases[f"model_{l}"] for l in labels}
    for name, ms in MEPS_SETS.items():
        ens[name] = np.stack([cases[f"meps_{m:03d}"] for m in ms], axis=2)
    return ens


def interval(diff: np.ndarray, dates: np.ndarray, replicates: int, scale: float) -> dict:
    bt = paired_block_bootstrap(diff, dates, block_days=7, replicates=replicates)
    return {"mean": bt["mean_difference"], "ci_lower": bt["ci_lower"], "ci_upper": bt["ci_upper"],
            "percent": 100 * bt["mean_difference"] / scale,
            "percent_ci": [100 * bt["ci_lower"] / scale, 100 * bt["ci_upper"] / scale]}


def score_set(name: str, sel: np.ndarray, ens: dict, obs: np.ndarray, cycle: np.ndarray,
              labels: list[str], comparator: str, primary: str | None, replicates: int) -> dict:
    dates, block = cycle[sel], cycle[sel].astype("datetime64[M]").astype(int)
    out = {"cases_per_lead": int(sel.sum()), "comparator": comparator, "leads": {}, "contrast_36_minus_12": {}}
    per_case = {}  # threshold -> label -> lead index -> calibrated squared error
    print(f"\n########## {name}: {int(sel.sum()):,} gauge-cycles at each lead, comparator {comparator}")
    for li, lead in enumerate(LEADS):
        o = obs[li, sel].astype("float64")
        res = {"thresholds": {}, "fair_crps": {}}
        crps = {l: fair_crps_cases(ens[l][li, sel].astype("float64"), o) for l in labels}
        for l in labels:
            res["fair_crps"][l] = {"mean": float(crps[l].mean())}
            if l != comparator:
                res["fair_crps"][l]["minus_comparator"] = interval(
                    crps[l] - crps[comparator], dates, replicates, float(crps[comparator].mean()))
        for t in THRESHOLDS:
            event = (o > t).astype(float)
            base = event.mean()
            row = {"events": int(event.sum()), "base_rate": float(base), "models": {}}
            cal, raw = {}, {}
            for l in labels:
                p = (ens[l][li, sel] > t).mean(axis=1)
                c = calibrated_cv(p, event, block)
                cal[l], raw[l] = (c - event) ** 2, (p - event) ** 2
                per_case.setdefault(t, {}).setdefault(l, {})[li] = cal[l]
                row["models"][l] = {"brier": float(raw[l].mean()), "brier_calibrated": float(cal[l].mean()),
                                    "bss_calibrated": float(1 - cal[l].mean() / (base * (1 - base))),
                                    "roc_area": roc_area(p, event)}
            for l in labels:
                if l == comparator:
                    continue
                row["models"][l]["calibrated_minus_comparator"] = interval(
                    cal[l] - cal[comparator], dates, replicates, float(cal[comparator].mean()))
                row["models"][l]["raw_minus_comparator"] = interval(
                    raw[l] - raw[comparator], dates, replicates, float(raw[comparator].mean()))
            res["thresholds"][f"{t:g}"] = row
        out["leads"][str(lead)] = res

    i12, i36 = LEADS.index(12), LEADS.index(36)
    for t in THRESHOLDS:
        out["contrast_36_minus_12"][f"{t:g}"] = {}
        pc = per_case[t]
        for l in labels:
            if l == comparator:
                continue
            d = (pc[l][i36] - pc[comparator][i36]) - (pc[l][i12] - pc[comparator][i12])
            bt = paired_block_bootstrap(d, dates, block_days=7, replicates=replicates)
            out["contrast_36_minus_12"][f"{t:g}"][l] = {
                "mean": bt["mean_difference"], "ci_lower": bt["ci_lower"], "ci_upper": bt["ci_upper"]}

    for t in THRESHOLDS:
        print(f"\n===== {t:g} mm in 6 h, calibrated: skill (BSS), and % Brier against {comparator} [95 %]")
        print("  lead  events  " + "  ".join(f"{l:>14s}" for l in labels))
        for lead in LEADS:
            row = out["leads"][str(lead)]["thresholds"][f"{t:g}"]
            print(f"  +{lead:<3d} {row['events']:7d}  " + "  ".join(
                f"{row['models'][l]['bss_calibrated']:14.3f}" for l in labels))
        for l in labels:
            if l == comparator:
                continue
            cells = []
            for lead in LEADS:
                d = out["leads"][str(lead)]["thresholds"][f"{t:g}"]["models"][l]["calibrated_minus_comparator"]
                cells.append(f"+{lead}: {d['percent']:+.1f} [{d['percent_ci'][0]:+.1f}, {d['percent_ci'][1]:+.1f}]")
            print(f"    {l:14s} " + "   ".join(cells))
        for l, d in out["contrast_36_minus_12"][f"{t:g}"].items():
            print(f"    {l:14s} (+36) - (+12) difference: {d['mean']:+.6f} [{d['ci_lower']:+.6f}, {d['ci_upper']:+.6f}]")
    print(f"\n===== fair CRPS of all rain, % against {comparator} [95 %]")
    for l in labels:
        if l == comparator:
            continue
        cells = []
        for lead in LEADS:
            d = out["leads"][str(lead)]["fair_crps"][l]["minus_comparator"]
            cells.append(f"+{lead}: {d['percent']:+.1f} [{d['percent_ci'][0]:+.1f}, {d['percent_ci'][1]:+.1f}]")
        print(f"    {l:14s} " + "   ".join(cells))

    if primary and primary in labels and primary != comparator:
        d = out["leads"]["36"]["thresholds"]["10"]["models"][primary]["calibrated_minus_comparator"]
        c = out["contrast_36_minus_12"]["10"][primary]
        verdict = "positive" if d["ci_upper"] < 0 else "negative" if d["ci_lower"] > 0 else "inconclusive"
        trend = "shrinks" if c["ci_lower"] > 0 else "grows" if c["ci_upper"] < 0 else "no detectable change"
        out["primary"] = {"model": primary, "at_36h_10mm": d, "verdict": verdict,
                          "contrast_36_minus_12": c, "contrast_verdict": trend}
        print(f"\nPRIMARY {primary} - {comparator}, 10 mm in 6 h at +36 h, calibrated: "
              f"{d['percent']:+.1f} % [{d['percent_ci'][0]:+.1f}, {d['percent_ci'][1]:+.1f}] -> {verdict}")
        print(f"KEY SECONDARY, +36 h against +12 h: {c['mean']:+.6f} [{c['ci_lower']:+.6f}, {c['ci_upper']:+.6f}] -> {trend}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    ap.add_argument("--comparator", required=True)
    ap.add_argument("--primary", default=None)
    ap.add_argument("--meps-members", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END, help="the plan's period; shorter only for a smoke test")
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
        missing = [l for l in labels if f"model_{l}" not in cases]
        if missing:
            raise SystemExit(f"{cache} has no {missing}; use another --cache")
    else:
        cases = collect(specs, args.observations.expanduser(), args.meps_members.expanduser(),
                        args.start, args.end)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **cases)
        print(f"wrote {cache}", flush=True)

    ens = ensembles(cases, labels)
    obs, cycle = cases["obs"], cases["cycle"].astype("datetime64[D]")
    everything = np.ones(cycle.size, dtype=bool)
    report = {"plan": "evaluation/lead_plan.json", "leads_hours": LEADS, "window_hours": WINDOW,
              "flagged_gauges_left_out": cases["flagged"].tolist(),
              "models": score_set("models", everything, ens, obs, cycle, labels,
                                  args.comparator, args.primary, args.replicates),
              "with_meps": score_set("with_meps", cases["with_meps"].astype(bool), ens, obs, cycle,
                                     ["meps4"] + labels + ["meps15"],
                                     "meps4", None, args.replicates)}
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

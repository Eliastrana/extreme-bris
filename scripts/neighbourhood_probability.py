#!/usr/bin/env python3
"""Neighbourhood probabilities: does a forecast of rain nearby beat one at the gauge?

    scripts/neighbourhood_probability.py \
      --model 'control3000=~/bris-runs/validation-forecasts/control3000/nordic_*.nc' \
      --model 'tail3000=~/bris-runs/validation-forecasts/tail3000/nordic_*.nc' \
      --observations ~/bris-runs/observations-val \
      --start 2025-04-01 --end 2025-07-31 \
      --out ~/bris-runs/neighbourhood/validation.json

EXPLORATORY, ON THE VALIDATION PERIOD. The question is whether placement
errors are part of why heavy-rain forecasts score badly at gauges: a shower
forecast 8 km from where it fell counts once as a miss and once as a false
alarm. If so, the probability of heavy rain within some distance of the gauge
should score better than the probability at the gauge's own grid cell. The
radius that works best here is what a later test on the test year would use;
choosing it on the test year would be choosing on the answer.

WHAT IT COMPUTES, per forecast day and threshold t:
  the 24-hour total ending 30 h after the 00 UTC start (06-06 UTC, the gauge
  day), for every grid cell and member; the share of members above t in each
  cell; and, for each radius r, the average of that share over every cell
  within r of a cell (a disk, edges handled by counting only cells inside the
  domain). r = 0 is the ordinary ensemble probability at the gauge's cell.

SCORES, on the evaluation's strict common cases (every model finite at the
gauge cell, gauge finite and screened):
  Brier      mean of (p - o)^2, o = 1 if the gauge exceeded t. Lower is better.
  fair Brier at r = 0 only: Brier - p(1 - p)/(M - 1), the score an infinite
             ensemble with the same distribution would get. A neighbourhood
             averages many more values than M members, and some of what it
             gains over the raw r = 0 Brier is that alone; the fair score is
             the honest yardstick for what is left.
  ROC area   how well p separates gauge-days that exceeded t from those that
             did not, whatever the calibration. 0.5 is no skill, 1 perfect.
  reliability  observed frequency in bins of p.
Differences in Brier from r = 0 carry a paired seven-day block bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy", "scipy")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.ndimage import convolve  # noqa: E402

import evaluate_experiment as ev  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import daily_sums, deaccumulate, paired_block_bootstrap  # noqa: E402

CELL_KM = 2.5
RADII_KM = [0.0, 2.5, 5.0, 10.0, 15.0, 25.0]
THRESHOLDS = [10.0, 20.0, 50.0]
LEAD = 30
BINS = [0.0, 1e-9, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 + 1e-9]


def disk(radius_cells: int) -> np.ndarray:
    r = radius_cells
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2 <= r ** 2 + 1e-9).astype("float64")


def daily_field(path: Path, accumulation: str) -> np.ndarray:
    """member x y x x: the 24 h total ending at +LEAD h, every grid cell."""
    with xr.open_dataset(path) as ds:
        da = ds[ev.PRECIPITATION]
        member = ev.member_dimension(da)
        for dim in list(da.dims):
            if dim not in {"time", "y", "x"} | ({member} if member else set()):
                da = da.isel({dim: 0}, drop=True)
        if member is None:
            member = "ensemble_member"
            da = da.expand_dims({member: [0]}, axis=1)
        field = np.asarray(da.transpose("time", member, "y", "x").values, dtype="float64")
        times = np.asarray(ds["time"].values).astype("datetime64[s]")
    t, m, ny, nx = field.shape
    leads = ((times - times[0]) / np.timedelta64(1, "h")).astype(int)
    steps, _ = deaccumulate(field.reshape(t, m, ny * nx), accumulation)
    return daily_sums(steps, leads, [LEAD])[0].reshape(m, ny, nx)


def roc_area(p: np.ndarray, o: np.ndarray) -> float:
    pos, neg = p[o == 1], p[o == 0]
    if not pos.size or not neg.size:
        return float("nan")
    ranks = np.empty(p.size)
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    i = 0
    while i < sp.size:  # average ranks over ties
        j = i
        while j + 1 < sp.size and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rpos = ranks[o == 1].sum()
    return float((rpos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def reliability(p: np.ndarray, o: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            rows.append({"lo": round(lo, 3), "hi": round(min(hi, 1.0), 3), "cases": int(sel.sum()),
                         "mean_p": float(p[sel].mean()), "observed": float(o[sel].mean())})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-quality", type=int, default=4)
    ap.add_argument("--max-distance-km", type=float, default=5.0)
    ap.add_argument("--accumulation", default="per_step")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    specs = dict(item.split("=", 1) for item in args.model)
    labels = list(specs)
    obs, _ = ev.load_screened_observations(args.observations.expanduser(), args.max_quality)
    indexes = {l: ev.index_model_files(specs[l]) for l in labels}
    cycles = ev.expected_cycles(args.start, args.end, 0)
    radii = [int(round(r / CELL_KM)) for r in RADII_KM]
    kernels = {rc: disk(rc) for rc in radii if rc > 0}

    reader = ev.BrisReader(obs, args.max_distance_km, args.accumulation)
    got = {"obs": [], "dates": [], "p": {l: {t: {r: [] for r in RADII_KM} for t in THRESHOLDS} for l in labels},
           "members": None}
    n_days = 0
    for cycle in cycles:
        if not all(cycle in indexes[l] for l in labels):
            continue
        valid = np.datetime64(cycle) + np.timedelta64(LEAD, "h")
        truth = ev.observation_for_time(obs, valid)
        if truth is None:
            continue
        fields = {}
        for l in labels:
            path = indexes[l][cycle]
            if reader._keep is None:
                with xr.open_dataset(path) as ds:
                    reader._match_grid(ds, path)
            fields[l] = daily_field(path, args.accumulation)
        keep = np.flatnonzero(reader._keep)
        row, col = reader._row, reader._col
        o_all = truth[keep]
        # strict common cases: gauge finite, every member of every model finite at its cell
        common = np.isfinite(o_all)
        for l in labels:
            common &= np.isfinite(fields[l][:, row, col]).all(axis=0)
        if not common.any():
            continue
        n_days += 1
        got["obs"].append(o_all[common])
        got["dates"].append(np.full(int(common.sum()), valid.astype("datetime64[D]")))
        for l in labels:
            f = fields[l]
            got["members"] = f.shape[0]
            inside = np.isfinite(f).all(axis=0).astype("float64")
            for t in THRESHOLDS:
                share = np.where(inside > 0, (np.nan_to_num(f, nan=0.0) > t).mean(axis=0), 0.0)
                for r_km, rc in zip(RADII_KM, radii):
                    if rc == 0:
                        nb = share
                    else:
                        num = convolve(share, kernels[rc], mode="constant", cval=0.0)
                        den = convolve(inside, kernels[rc], mode="constant", cval=0.0)
                        nb = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                    got["p"][l][t][r_km].append(nb[row, col][common])
        print(f"  {cycle[:10]}: {int(common.sum())} gauges", flush=True)

    o = np.concatenate(got["obs"])
    dates = np.concatenate(got["dates"])
    m = got["members"]
    report = {"exploratory": True, "lead_hours": LEAD, "days": n_days, "station_days": int(o.size),
              "members": m, "radii_km": RADII_KM, "models": {}}
    print(f"\n{n_days} forecast days, {o.size:,} station-days, {m} members, +{LEAD} h")
    for l in labels:
        report["models"][l] = {}
        for t in THRESHOLDS:
            event = (o > t).astype("float64")
            base = None
            rows = {}
            for r_km in RADII_KM:
                p = np.concatenate(got["p"][l][t][r_km])
                bs_cases = (p - event) ** 2
                row = {"brier": float(bs_cases.mean()), "roc_area": roc_area(p, event),
                       "mean_p": float(p.mean()), "event_rate": float(event.mean()),
                       "reliability": reliability(p, event)}
                if r_km == 0.0:
                    base = bs_cases
                    row["fair_brier"] = float((bs_cases - p * (1 - p) / (m - 1)).mean())
                else:
                    bt = paired_block_bootstrap(bs_cases - base, dates, block_days=7, replicates=args.replicates)
                    row["brier_minus_point"] = {"mean": bt["mean_difference"], "ci_lower": bt["ci_lower"],
                                                "ci_upper": bt["ci_upper"]}
                rows[f"{r_km:g}"] = row
            report["models"][l][f"{t:g}"] = {"events": int(event.sum()), "radii": rows}
            print(f"\n{l}, {t:g} mm: {int(event.sum()):,} gauge-days over, rate {event.mean():.4f}")
            print(f"  {'radius':>8s} {'Brier':>8s} {'vs point':>24s} {'ROC area':>9s} {'mean p':>7s}")
            for r_km in RADII_KM:
                row = rows[f"{r_km:g}"]
                if r_km == 0.0:
                    extra = f"(fair {row['fair_brier']:.5f})"
                else:
                    d = row["brier_minus_point"]
                    extra = f"{d['mean']:+.5f} [{d['ci_lower']:+.5f}, {d['ci_upper']:+.5f}]"
                print(f"  {r_km:6.1f}km {row['brier']:8.5f} {extra:>24s} {row['roc_area']:9.3f} {row['mean_p']:7.4f}")

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

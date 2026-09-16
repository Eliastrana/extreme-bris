#!/usr/bin/env python3
"""Score ensemble forecasts member by member against one 06-06 UTC gauge day.

    scripts/score_members.py --end 2025-07-09T06 \
        --observations ~/bris-runs/observations-val \
        baseline=~/bris-runs/cpu-smoke/20250708T00Z-baseline-m4/nordic_*.nc \
        control=~/bris-runs/cpu-smoke/20250708T00Z-control-m4/nordic_*.nc

WHAT IT IS FOR. One member of a CRPS-trained model is one draw, and for rain
two draws of the same model can differ as much as two models do. Before a
difference between models means anything, it has to be larger than the spread
between members of the same model. So every member is scored on its own, and
the models are compared on the range of their members, on the ensemble mean,
and on CRPS, which judges the ensemble as a whole.

THE SAME RULES AS score_forecast.py. Gauges are matched to the nearest grid
point within 5 km, the day is the 24 hours ending at --end, values Frost codes
worse than 4 are blanked, and gauges flagged by check_gauges.py are left out.
Both models are scored on exactly the gauges that have a value for that day.
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
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.stations import load_grid, load_observations, nearest  # noqa: E402

VAR = "precipitation_amount"
THRESHOLDS = (10.0, 20.0, 50.0)


def day_sums(path: Path, obs: dict, end: np.datetime64, max_km: float):
    """24 h sums per member at the gauges, and which gauges matched."""
    with xr.open_dataset(path) as ds:
        glat, glon, shape = load_grid(path)
        idx, dist = nearest(obs["lat"], obs["lon"], glat, glon)
        keep = dist <= max_km
        row, col = np.divmod(idx[keep], shape[1])
        times = ds["time"].values.astype("datetime64[s]")
        da = ds[VAR]
        other = [d for d in da.dims if d not in ("time", "y", "x") and "ensemble" not in d]
        da = da.squeeze(other, drop=True)
        member_dim = next((d for d in da.dims if "ensemble" in d), None)
        if member_dim is None:
            da = da.expand_dims("ensemble_member", axis=1)
            member_dim = "ensemble_member"
        da = da.transpose("time", member_dim, "y", "x")
        step = np.asarray(da.values, dtype="float64")[:, :, row, col]   # time, member, station
    step_h = int((times[1] - times[0]) / np.timedelta64(1, "h"))
    k = int(np.where(times == end)[0][0])
    n = 24 // step_h
    if k - n + 1 < 1:
        raise SystemExit(f"{path.name}: the day ending {end} reaches back to the initial state")
    # bris writes amounts per step; a cumulative file would rise monotonically.
    if np.nanmean(np.diff(np.nan_to_num(step), axis=0) >= -1e-6) > 0.98:
        step = np.diff(step, axis=0, prepend=np.zeros_like(step[:1]))
    return step[k - n + 1:k + 1].sum(axis=0), keep                     # member, station


def scores(fc: np.ndarray, ob: np.ndarray) -> dict:
    out = {"n": int(ob.size), "forecast_mean": float(fc.mean()), "bias": float(fc.mean() - ob.mean()),
           "mae": float(np.abs(fc - ob).mean())}
    wet = ob >= 20.0
    out["mean_on_20mm_days"] = float(fc[wet].mean()) if wet.any() else None
    out["mean_on_other_days"] = float(fc[~wet].mean()) if (~wet).any() else None
    for t in THRESHOLDS:
        o, f = ob >= t, fc >= t
        out[f"hit_rate_{t:g}"] = float((o & f).sum() / o.sum()) if o.any() else None
        out[f"false_alarms_{t:g}"] = int((f & ~o).sum())
    return out


def crps(ens: np.ndarray, ob: np.ndarray) -> float:
    """Fair CRPS of an m-member ensemble, averaged over gauges."""
    m = ens.shape[0]
    skill = np.abs(ens - ob[None]).mean(axis=0)
    spread = np.abs(ens[:, None] - ens[None, :]).sum(axis=(0, 1)) / (2 * m * (m - 1))
    return float((skill - spread).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forecasts", nargs="+", help="label=path to a nordic NetCDF file")
    ap.add_argument("--end", required=True, help="end of the 06-06 UTC day, e.g. 2025-07-09T06")
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--max-dist-km", type=float, default=5.0)
    ap.add_argument("--max-quality", type=int, default=4)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    obs = load_observations(args.observations / "precipitation_daily.npz", max_quality=args.max_quality)
    values = np.array(obs["values"], dtype="float64")
    screen = args.observations / "precipitation_daily_check.json"
    if screen.exists():
        flagged = json.loads(screen.read_text())["flagged"]
        values[np.isin(obs["stations"], list(flagged))] = np.nan
        print(f"  screened out {len(flagged)} gauge(s) flagged in {screen.name}")
    end = np.datetime64(args.end).astype("datetime64[s]")
    j = np.where(obs["times"] == end)[0]
    if not len(j):
        raise SystemExit(f"no gauge day ends at {end}")
    truth = values[:, int(j[0])]

    runs = {}
    for item in args.forecasts:
        label, path = item.split("=", 1)
        runs[label] = day_sums(Path(path).expanduser(), obs, end, args.max_dist_km)

    # One gauge set for every model: matched in all, and reporting that day.
    common = np.isfinite(truth)
    for _, keep in runs.values():
        common &= keep
    ob = truth[common]
    print(f"=== 24 h to {end}: {int(common.sum())} gauges, observed mean {ob.mean():.2f} mm, "
          f"{int((ob >= 20).sum())} at 20 mm or more, {int((ob >= 50).sum())} at 50 or more, "
          f"max {ob.max():.1f}\n")

    report = {"end": str(end), "gauges": int(common.sum()), "models": {}}
    head = (f"{'model':14s} {'member':>7s} {'MAE':>6s} {'bias':>6s} {'on 20mm':>8s} {'on other':>9s}"
            f" {'hit10':>6s} {'hit20':>6s} {'hit50':>6s} {'false20':>8s}")
    print(head)
    for label, (sums, keep) in runs.items():
        ens = sums[:, common[keep]]                      # member, gauge
        rows = {}
        for mi in range(ens.shape[0]):
            rows[f"m{mi}"] = scores(ens[mi], ob)
        rows["mean"] = scores(ens.mean(axis=0), ob)
        for name, s in rows.items():
            fmt = lambda v, w=6, p=2: f"{v:{w}.{p}f}" if v is not None else f"{'-':>{w}s}"
            print(f"{label:14s} {name:>7s} {fmt(s['mae'])} {fmt(s['bias'])} {fmt(s['mean_on_20mm_days'], 8)}"
                  f" {fmt(s['mean_on_other_days'], 9)} {fmt(s['hit_rate_10'])} {fmt(s['hit_rate_20'])}"
                  f" {fmt(s['hit_rate_50'])} {s['false_alarms_20']:8d}")
        c = crps(ens, ob) if ens.shape[0] > 1 else None
        spread = [rows[f"m{mi}"]["hit_rate_50"] for mi in range(ens.shape[0])]
        spread = [s for s in spread if s is not None]
        print(f"{label:14s} {'CRPS':>7s} {c:.3f}" if c is not None else f"{label:14s} CRPS needs 2+ members")
        if spread:
            print(f"{label:14s} {'hit50':>7s} range over members {min(spread):.2f} .. {max(spread):.2f}")
        print()
        report["models"][label] = {"members": rows, "crps": c}

    out = args.out or Path.home() / "bris-runs" / "score" / f"members-{str(end)[:13]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score forecasts at gauges, splitting the tail from the ordinary days.

    scripts/score_forecast.py ~/bris-runs/20251004T00Z-od-120h/nordic_*.nc
    scripts/score_forecast.py --label bris-tail ~/bris-runs/tail/nordic_*.nc
    scripts/score_forecast.py --label meps ~/bris-runs/meps/precipitation.npz

TWO SHAPES OF FORECAST, ONE SCORE. The arms produce gridded NetCDF, one file
per run, and the station values are read out of the grid. MEPS is fetched at
the gauge sites directly, because pulling a year of full fields to use a
thousand points of them would be absurd. Both end up as the same thing, a value
per station per valid time, and everything after that is shared. It has to be:
a control scored by a different code path is not a control.

WHAT THIS IS FOR. The experiment ends in a comparison: baseline, control arm,
treatment arm, with MEPS alongside as something whose skill is known. That
comparison needs one number for the tail and one for everything else, computed
the same way for every arm, or the arms are not comparable no matter how
carefully they were trained.

WHY BOTH HALVES. Sharpening the tail almost always costs something in the
middle. An arm that goes from finding none of the heavy-rain stations to
finding most of them has succeeded only if it did not also turn the 341 quiet
days of the year wet. Reporting only the tail score would hide exactly the
failure the thesis is about avoiding, in the opposite direction.

THE TRAP THIS AVOIDS. Scoring only the days an extreme happened, or only the
stations that recorded one, is the forecaster's dilemma: it rewards crying
wolf, because a model that always predicts 100 mm looks perfect when judged
only on days that got 100 mm. Every threshold statistic here therefore counts
false alarms on the days nothing happened, over the same stations, from the
same files.

WHAT IT READS. Hourly gauge values cached by fetch_observations.py, summed over
the forecast's own accumulation window rather than a fixed one. The grid comes
from the forecast file itself, not from the training data: the two differ by
the fifty-point edge trim, and an index from one applied to the other lands on
real numbers belonging to somewhere else.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.stations import (accumulate, load_grid, load_observations,  # noqa: E402
                            nearest)

# mm in the accumulation window. The lowest is the wet/dry line; the highest
# brackets the event that motivated this work, where 24 gauges passed 75 mm in
# a day and the model produced none.
THRESHOLDS = (0.5, 5.0, 10.0, 20.0, 50.0)

# Forecast variable per element, and what turns it into the gauge's unit.
VARIABLES = {
    "precipitation": ("precipitation_amount", lambda a: a),
    "temperature": ("air_temperature_2m", lambda a: a - 273.15),
    "wind": ("wind_speed_10m", lambda a: a),
}


def deaccumulate(series: np.ndarray) -> tuple[np.ndarray, str]:
    """Per-step amounts, whichever convention the file uses.

    A cumulative field and a per-step field look identical at a glance and give
    answers that differ by a factor of the lead time. Monotonicity decides, and
    the decision is returned so it appears in the output rather than being
    assumed silently.
    """
    finite = np.where(np.isfinite(series), series, 0.0)
    rising = np.diff(finite, axis=1) >= -1e-6
    if rising.mean() > 0.98:
        out = np.diff(series, axis=1, prepend=np.zeros((series.shape[0], 1)))
        return np.maximum(out, 0.0), "cumulative"
    return series, "per step"


def contingency(fc, ob, threshold):
    """Hits, misses and false alarms at one threshold, over every valid pair."""
    good = np.isfinite(fc) & np.isfinite(ob)
    f, o = fc[good] >= threshold, ob[good] >= threshold
    hits = int((f & o).sum())
    misses = int((~f & o).sum())
    false = int((f & ~o).sum())
    correct = int((~f & ~o).sum())
    return {
        "observed": hits + misses,
        "forecast": hits + false,
        "hits": hits, "misses": misses,
        "false_alarms": false, "correct_negatives": correct,
        "hit_rate": hits / (hits + misses) if hits + misses else None,
        "false_alarm_ratio": false / (hits + false) if hits + false else None,
    }


def score(fc, ob) -> dict:
    good = np.isfinite(fc) & np.isfinite(ob)
    f, o = fc[good], ob[good]
    if f.size == 0:
        return {"pairs": 0}
    return {
        "pairs": int(f.size),
        "observed_mean": float(o.mean()),
        "forecast_mean": float(f.mean()),
        "bias": float((f - o).mean()),
        "mae": float(np.abs(f - o).mean()),
        "rmse": float(np.sqrt(((f - o) ** 2).mean())),
        "observed_p99": float(np.percentile(o, 99)),
        "forecast_p99": float(np.percentile(f, 99)),
    }


def from_grid(path: Path, obs: dict, var: str, convert, max_dist_km: float):
    """Station series read out of a gridded forecast file."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if var not in ds:
            print(f"  {path.name}: no {var}, skipping")
            return None
        glat, glon, shape = load_grid(path)
        idx, dist = nearest(obs["lat"], obs["lon"], glat, glon)
        keep = dist <= max_dist_km
        row, col = np.divmod(idx[keep], shape[1])

        times = ds["time"].values.astype("datetime64[s]")
        field = np.asarray(ds[var].squeeze().values, dtype="float64")
        if field.ndim != 3:
            print(f"  {path.name}: {var} is {field.ndim}-D after squeeze, "
                  "expected time by y by x; skipping")
            return None
        series = convert(field[:, row, col]).T

    step_h = int((times[1] - times[0]) / np.timedelta64(1, "h")) if len(times) > 1 else 6
    return [(series, times, step_h, keep, None)]


def from_points(path: Path, obs: dict, element: str):
    """Station series read from a point forecast file, one entry per cycle.

    The stations are realigned by name rather than assumed to be in the same
    order. They are written by the same fetch that reads the gauge cache, so
    they should match, but a silent misalignment here would pair every station
    with somebody else's weather and still produce a number.
    """
    with np.load(path, allow_pickle=False) as f:
        values = f["values"]
        stations = f["stations"]
        cycles = np.array([np.datetime64(c) for c in f["cycles"]],
                          dtype="datetime64[s]")
        leads = f["leads"].astype(int)
        accumulation = str(f["accumulation"]) if "accumulation" in f else ""

    where = {s: i for i, s in enumerate(stations)}
    order = np.array([where.get(s, -1) for s in obs["stations"]])
    keep = order >= 0
    if not keep.any():
        print(f"  {path.name}: no station in it matches the gauge cache")
        return None
    if keep.sum() < len(obs["stations"]):
        print(f"  {path.name}: {int((~keep).sum())} gauge(s) absent from it")

    step_h = int(leads[1] - leads[0]) if len(leads) > 1 else 6
    out = []
    for ci, cycle in enumerate(cycles):
        series = values[order[keep], ci, :].astype("float64")
        if not np.isfinite(series).any():
            continue
        times = cycle + leads.astype("timedelta64[h]").astype("timedelta64[s]")
        out.append((series, times, step_h, keep, accumulation))
    print(f"  {path.name}: {int(keep.sum())} gauges, {len(out)} cycles kept of "
          f"{len(cycles)}, {len(leads)} leads {step_h}h apart"
          + (f", {accumulation}" if accumulation else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forecasts", nargs="+", type=Path)
    ap.add_argument("--element", default="precipitation", choices=sorted(VARIABLES))
    ap.add_argument("--observations", type=Path,
                    default=Path.home() / "bris-runs" / "observations")
    ap.add_argument("--label", default=None, help="name this arm in the output")
    ap.add_argument("--max-dist-km", type=float, default=5.0)
    ap.add_argument("--tail-mm", type=float, default=20.0,
                    help="a valid time counts as tail if any gauge passed this")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    import xarray as xr

    obs = load_observations(args.observations / f"{args.element}.npz")
    var, convert = VARIABLES[args.element]
    label = args.label or args.forecasts[0].parent.name
    print(f"=== {label}: {len(args.forecasts)} forecast file(s), {args.element}\n")

    fc_all, ob_all, lead_all, all_times = [], [], [], []
    convention = None

    for path in args.forecasts:
        if path.suffix == ".npz":
            blocks = from_points(path, obs, args.element)
        else:
            blocks = from_grid(path, obs, var, convert, args.max_dist_km)
        if not blocks:
            continue

        for series, times, step_h, keep, note in blocks:
            if args.element == "precipitation" and note is None:
                # A gridded file may hold amounts per step or since the run
                # began. A point file says which it holds and has already been
                # differenced, so leave it alone.
                series, convention = deaccumulate(series)
            elif note:
                convention = note
            truth = accumulate(obs["values"][keep], obs["times"], times,
                               step_h if args.element == "precipitation" else 1)

            # Step zero of a forecast is its own initial state, not a forecast.
            fc_all.append(series[:, 1:])
            ob_all.append(truth[:, 1:])
            lead_all.append(np.tile(np.arange(1, len(times)) * step_h,
                                    (series.shape[0], 1)))
            all_times.append(times)
        if blocks and blocks[0][4] is None:
            print(f"  {path.name}: {len(blocks)} run(s)"
                  + (f", {convention}" if convention else ""))

    if not fc_all:
        print("\nNothing scored.", file=sys.stderr)
        return 1

    fc = np.concatenate([a.ravel() for a in fc_all])
    ob = np.concatenate([a.ravel() for a in ob_all])
    lead = np.concatenate([a.ravel() for a in lead_all])
    all_times = np.concatenate(all_times)

    # A valid time is a tail case if any gauge passed the threshold THEN. The
    # split is made on the observations alone, never on the forecast, or the
    # comparison would ask each arm about a different set of days.
    tail_pairs = ob >= args.tail_mm
    n_truth = int(np.isfinite(ob).sum())
    print(f"\n  {n_truth:,} station-hours with truth, "
          f"{int(tail_pairs.sum()):,} of them at or above {args.tail_mm:g}")

    if n_truth == 0:
        # Almost always the cache not covering these dates rather than anything
        # wrong with the forecast, and a bare zero does not say which.
        have = (obs["times"].min(), obs["times"].max())
        print(f"\n  The gauge cache covers {have[0]} .. {have[1]}.")
        print(f"  These forecasts are valid over "
              f"{np.min(all_times)} .. {np.max(all_times)}.")
        if np.max(all_times) < have[0] or np.min(all_times) > have[1]:
            print("  Those do not overlap, which is the whole explanation.")
            print("  Extend the cache:  scripts/fetch_observations.py "
                  f"--start {str(np.min(all_times))[:10]} "
                  f"--end {str(np.max(all_times))[:10]}")
        else:
            print("  They do overlap, so the gauges reported nothing usable in "
                  "that window.\n  A partial accumulation window counts as "
                  "missing here, by design.")
        return 1

    report = {
        "label": label,
        "element": args.element,
        "accumulation": convention,
        "tail_mm": args.tail_mm,
        "files": [str(p) for p in args.forecasts],
        "all": score(fc, ob),
        "tail": score(fc[tail_pairs], ob[tail_pairs]),
        "ordinary": score(fc[~tail_pairs], ob[~tail_pairs]),
        "thresholds": {str(t): contingency(fc, ob, t) for t in THRESHOLDS},
        "by_lead_hours": {str(int(h)): score(fc[lead == h], ob[lead == h])
                          for h in np.unique(lead)},
    }

    def line(name, s):
        if not s.get("pairs"):
            print(f"  {name:10s} nothing to score")
            return
        print(f"  {name:10s} n={s['pairs']:7,d}  observed {s['observed_mean']:7.3f}"
              f"  forecast {s['forecast_mean']:7.3f}  bias {s['bias']:+7.3f}"
              f"  mae {s['mae']:6.3f}")

    print(f"\n--- {args.element}, in {obs['unit']}")
    for name in ("all", "tail", "ordinary"):
        line(name, report[name])

    print(f"\n--- thresholds, over every station-hour")
    print(f"  {'mm':>6s} {'observed':>9s} {'forecast':>9s} {'hits':>6s} "
          f"{'misses':>7s} {'false':>6s} {'hit rate':>9s}")
    for t in THRESHOLDS:
        c = report["thresholds"][str(t)]
        hr = f"{c['hit_rate']:.3f}" if c["hit_rate"] is not None else "-"
        print(f"  {t:6.1f} {c['observed']:9d} {c['forecast']:9d} {c['hits']:6d} "
              f"{c['misses']:7d} {c['false_alarms']:6d} {hr:>9s}")

    out = args.out or Path("results") / "score" / f"{label}-{args.element}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    print("\nCompare arms by running this once per arm and reading the tail and\n"
          "ordinary lines together. Neither means anything on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

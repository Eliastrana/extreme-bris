#!/usr/bin/env python3
"""Pull one MEPS ensemble member at the daily gauges, for every day, as the 00 UTC forecast would see it.

    scripts/fetch_meps_members.py --member 9 --lag 0 --start 2025-08-01 --end 2026-09-07 \\
        --out ~/bris-runs/meps-ens

WHY. Every comparison so far set the models against one deterministic MEPS run
(meps_det_sfc, scripts/fetch_meps_forecasts.py). MET issues forecasts from the
MEPS ensemble. This fetches the members so the models can be set against that.

HOW THE ARCHIVE HOLDS THE ENSEMBLE. Members run hourly in a three-hour cycle,
five at a time: at 00, 03, ... UTC members 0, 1, 2, 9 and 12; an hour earlier
5, 6, 8, 11 and 14; two hours earlier 3, 4, 7, 10 and 13. The fifteen that
make up the ensemble valid from 00 UTC on day D are therefore the five of
D 00Z, lag 0, and ten lagged ones from D-1 23Z and 22Z, lag 1 and 2. Each
file runs 66 hours, so the lagged members still reach the +30 h day.

WHAT IS STORED is exactly the layout of fetch_meps_forecasts.py's output, so
evaluate_experiment.MepsReader reads it unchanged: six-hour amounts per gauge,
per day, per lead, with cycles labelled D 00Z and leads counted from D 00Z
whatever the member's own start. A lead's amount is the accumulation at that
valid time minus the one six hours before; for a lagged member the first slot
(lead 0) would need a time before its run began and is left missing, and it
is not used by the +30 h day (leads 12 to 30). A day whose file is missing or
short is missing, not filled.

One member per process, so a Slurm array runs the fifteen side by side.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy", "pydap")

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore", message=".*unable to determine the DAP protocol.*")
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

from bris_tls import ensure_ca_bundle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.stations import load_observations, nearest  # noqa: E402

ARCHIVE = ("https://thredds.met.no/thredds/dodsC/meps25epsarchive/"
           "{t:%Y}/{t:%m}/{t:%d}/{t:%H}/meps_mbr{m:03d}_sfc_{t:%Y%m%d}T{t:%H}Z.ncml")
MARGIN = 3
VARIABLE = "precipitation_amount_acc"


def open_member(cycle: dt.datetime, member: int):
    import xarray as xr
    return xr.open_dataset(ARCHIVE.format(t=cycle, m=member), engine="pydap")


class Missing(Exception):
    """The day's file is not there, or too short: final, saved as missing."""


def retrying(fetch, tries: int = 7):
    """Run fetch(), backing off on rate limits and network errors.

    thredds answers 429 Too Many Requests when many of these run at once, and
    an answer like that says nothing about the data. Only a 404 (no such file)
    or a short run is final. Anything else that outlasts the retries returns
    None, so the day is left unsaved and a rerun tries it again.
    """
    for attempt in range(tries):
        try:
            return fetch()
        except Missing:
            raise
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "404" in text:
                raise Missing(text[:90]) from exc
            if attempt == tries - 1:
                print(f"    gave up after {tries} tries: {type(exc).__name__}: {text[:80]}")
                return None
            time.sleep(min(20 * 2 ** attempt, 600) * (0.75 + 0.5 * random.random()))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", type=int, required=True)
    ap.add_argument("--lag", type=int, required=True, help="hours before D 00Z this member's cycle starts: 0, 1 or 2")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--every", type=int, default=1, help="every Nth day")
    ap.add_argument("--max-lead", type=int, default=36)
    ap.add_argument("--observations", type=Path, default=Path.home() / "bris-runs" / "observations" / "precipitation_daily.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="", help="added to the output name, e.g. the period")
    args = ap.parse_args()

    ensure_ca_bundle()
    days = []
    d = dt.datetime.fromisoformat(args.start)
    while d <= dt.datetime.fromisoformat(args.end):
        days.append(d)
        d += dt.timedelta(days=args.every)
    leads = list(range(0, args.max_lead + 1, 6))
    print(f"=== member {args.member}, lag {args.lag} h, {len(days)} days, leads {leads}")

    obs = load_observations(args.observations)
    def read_grid(when):
        with open_member(when, args.member) as ds:
            return (np.asarray(ds["latitude"].values, dtype="float64"),
                    np.asarray(ds["longitude"].values, dtype="float64"))

    for probe in days:
        try:
            got = retrying(lambda: read_grid(probe - dt.timedelta(hours=args.lag)))
        except Missing:
            got = None
        if got is not None:
            lat, lon = got
            break
        print(f"  no grid from {probe:%Y-%m-%d}")
    else:
        raise SystemExit("no file of this member opens in the whole period")
    idx, dist = nearest(obs["lat"], obs["lon"], lat.ravel(), lon.ravel())
    row, col = np.divmod(idx, lat.shape[1])
    r0, r1 = max(row.min() - MARGIN, 0), min(row.max() + MARGIN + 1, lat.shape[0])
    c0, c1 = max(col.min() - MARGIN, 0), min(col.max() + MARGIN + 1, lat.shape[1])
    rr, cc = row - r0, col - c0
    print(f"  {len(idx)} gauges, worst {dist.max():.2f} km from a grid point, box {r1 - r0} by {c1 - c0}")

    cache = args.out / "cache" / f"mbr{args.member:03d}"
    cache.mkdir(parents=True, exist_ok=True)
    grid = np.full((len(obs["stations"]), len(days), len(leads)), np.nan, dtype="float32")
    missing = 0
    started = dt.datetime.now()
    for di, day in enumerate(days):
        part = cache / f"{day:%Y%m%d}.npy"
        if part.exists():
            grid[:, di, :] = np.load(part)
            continue
        cycle = day - dt.timedelta(hours=args.lag)

        def fetch_day():
            with open_member(cycle, args.member) as ds:
                times = np.asarray(ds["time"].values).astype("datetime64[h]")
                # accumulations at D00Z-6h .. D00Z+max_lead, those inside the run
                valid = np.datetime64(day, "h") + np.array([-6] + leads, dtype="timedelta64[h]")
                pos = np.searchsorted(times, valid)
                inside = (pos < len(times)) & (times[np.minimum(pos, len(times) - 1)] == valid)
                if not inside[1:].all():
                    raise Missing(f"run ends {times[-1]}, before +{leads[-1]} h from {day:%Y-%m-%d}")
                take = pos[inside]
                # a strided slice when the steps are evenly spaced, so the server sends only those
                steps = (slice(int(take[0]), int(take[-1]) + 1, 6) if np.all(np.diff(take) == 6)
                         else take)
                acc = np.asarray(ds[VARIABLE].isel(time=steps, y=slice(r0, r1), x=slice(c0, c1))
                                 .squeeze(drop=True).values, dtype="float64")[:, rr, cc]
                full = np.full((len(valid), len(rr)), np.nan)
                full[inside] = acc
                if args.lag == 0:
                    full[0] = full[1]          # run starts at D00Z: nothing fell before lead 0
                amounts = np.maximum(np.diff(full, axis=0), 0.0)   # windows ending at each lead
                return amounts.T.astype("float32")

        try:
            block = retrying(fetch_day)
        except Missing as exc:
            missing += 1
            print(f"  {day:%Y-%m-%d} not in the archive: {exc}")
            block = np.full((len(obs["stations"]), len(leads)), np.nan, dtype="float32")
        if block is None:
            continue                      # transient; left unsaved so a rerun tries again
        np.save(part, block)
        grid[:, di, :] = block
        if (di + 1) % 20 == 0:
            per = (dt.datetime.now() - started).total_seconds() / (di + 1)
            print(f"  {di + 1}/{len(days)}  {per:.1f}s each")

    name = f"mbr{args.member:03d}{args.tag}.npz"
    np.savez_compressed(
        args.out / name, values=grid, stations=obs["stations"], lat=obs["lat"], lon=obs["lon"],
        distance_km=dist, cycles=np.array([f"{d:%Y-%m-%dT00:00:00}" for d in days]),
        leads=np.array(leads), element="precipitation_daily", member=args.member, lag_hours=args.lag,
        accumulation="differenced between valid times; leads from D 00Z",
    )
    print(f"  {np.isfinite(grid[:, :, 2:]).all(axis=(0, 2)).mean():.1%} of days complete, "
          f"{missing} unavailable; wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

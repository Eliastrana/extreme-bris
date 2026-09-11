#!/usr/bin/env python3
"""Pull MEPS forecasts at gauge sites, to stand beside the arms in the table.

    scripts/fetch_meps_forecasts.py --start 2025-08-01 --end 2026-09-07
    scripts/fetch_meps_forecasts.py --elements precipitation,temperature,wind

WHY A CONTROL AT ALL. Without one, a number saying Bris finds 38 percent of
observed rainfall in the tail is uninterpretable. It could be a failing of this
model, or it could be what any model on a 2.5 km grid scores against point
gauges, where a grid box is an average and a gauge is a pinhole. MEPS answers
that, because it runs on the same grid over the same country and its skill is
known. On 4 October it matched the observed 99th percentile to within a
millimetre while Bris found 62 percent of the mean, and it is that contrast,
not the raw figure, that says something about Bris.

IT MUST BE A FORECAST, NOT AN ANALYSIS. The MEPS analysis is already sitting in
the training datasets and costs nothing to read. It is also not a forecast: an
analysis has seen the observations it is being scored against. Putting it in
the same column as a 24 hour forecast would flatter it enormously and the
comparison would say nothing. So this reads the archived forecast cycles, at
the same lead times the arms will be scored at.

WHY IT IS SMALL. Only the gauge sites are wanted, so this reads a box around
the stations rather than the whole domain, and only the lead times asked for.
That turns tens of gigabytes into single figures.

ACCUMULATION. MEPS precipitation accumulates from the start of the run, which
is why the dataset recipe reads step 6 of the previous cycle rather than step 0
of its own. Here the whole run is in hand, so consecutive steps are differenced
to give the amount in each window, and the differencing is reported rather than
assumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402

from bris_tls import ensure_ca_bundle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.stations import load_observations, nearest  # noqa: E402

ARCHIVE = ("https://thredds.met.no/thredds/dodsC/meps25epsarchive/"
           "{t:%Y}/{t:%m}/{t:%d}/meps_det_sfc_{t:%Y%m%d}T{t:%H}Z.ncml")

# forecast element -> (variable in the file, how to turn it into the gauge unit)
ELEMENTS = {
    "precipitation": (("precipitation_amount_acc",), lambda a: a),
    "temperature": (("air_temperature_2m",), lambda a: a - 273.15),
    "wind": (("x_wind_10m", "y_wind_10m"), lambda a: a),
}

MARGIN = 3          # grid points of slack around the station bounding box


# Which xarray backend can talk OPeNDAP here. The dataset recipes use pydap,
# but that lives in the build environment; this one has netcdf4, whose library
# speaks DAP directly. Decided once, from what actually opens, rather than
# named in advance and wrong in one environment or the other.
_ENGINE: str | None = None


def open_cycle(when: dt.datetime):
    global _ENGINE
    import xarray as xr

    url = ARCHIVE.format(t=when)
    if _ENGINE:
        return xr.open_dataset(url, engine=_ENGINE)

    tried = {}
    for engine in ("netcdf4", "pydap"):
        try:
            ds = xr.open_dataset(url, engine=engine)
        except Exception as exc:  # noqa: BLE001
            tried[engine] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        _ENGINE = engine
        print(f"  reading the archive with the {engine} engine")
        return ds

    raise SystemExit(
        "no xarray engine here can open the archive over OPeNDAP:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in tried.items())
    )


def station_cells(obs, when: dt.datetime):
    """Row and column of each gauge, from the forecast file's own coordinates.

    Taken from thredds rather than from the training zarr. The two should be
    the same grid, and if they have drifted, reading indices from one file and
    values from another lands on real numbers belonging to somewhere else.
    """
    with open_cycle(when) as ds:
        lat = np.asarray(ds["latitude"].values, dtype="float64")
        lon = np.asarray(ds["longitude"].values, dtype="float64")
    if lat.ndim != 2:
        raise SystemExit(f"latitude is {lat.ndim}-D in the archive; expected 2-D")
    idx, dist = nearest(obs["lat"], obs["lon"], lat.ravel(), lon.ravel())
    row, col = np.divmod(idx, lat.shape[1])
    print(f"  matched {len(idx)} gauges, worst {dist.max():.2f} km from a grid point")
    return row, col, dist, lat.shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-09-07")
    ap.add_argument("--cycle-hour", type=int, default=0,
                    help="which daily cycle to take (default 00Z)")
    ap.add_argument("--max-lead", type=int, default=60,
                    help="hours ahead to keep (default 60)")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--elements", default="precipitation")
    ap.add_argument("--observations", type=Path,
                    default=Path.home() / "bris-runs" / "observations")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path.home() / "bris-runs" / "meps")
    args = ap.parse_args()

    wanted = [e.strip() for e in args.elements.split(",") if e.strip()]
    unknown = [e for e in wanted if e not in ELEMENTS]
    if unknown:
        raise SystemExit(f"unknown element(s): {', '.join(unknown)}")

    ensure_ca_bundle()
    import xarray as xr

    start = dt.datetime.fromisoformat(args.start).replace(hour=args.cycle_hour)
    end = dt.datetime.fromisoformat(args.end)
    cycles = []
    t = start
    while t <= end:
        cycles.append(t)
        t += dt.timedelta(days=1)
    leads = list(range(0, args.max_lead + 1, args.step))
    print(f"=== {len(cycles)} cycles at {args.cycle_hour:02d}Z, "
          f"leads {leads[0]} to {leads[-1]}h every {args.step}h\n")

    obs = load_observations(args.observations / f"{wanted[0]}.npz")
    print("=== grid")
    row, col, dist, shape = station_cells(obs, cycles[0])

    # One box around every gauge, read whole and subset locally. Reading a
    # thousand scattered points over OPeNDAP is a thousand requests.
    r0, r1 = max(row.min() - MARGIN, 0), min(row.max() + MARGIN + 1, shape[0])
    c0, c1 = max(col.min() - MARGIN, 0), min(col.max() + MARGIN + 1, shape[1])
    rr, cc = row - r0, col - c0
    print(f"  box {r1 - r0} by {c1 - c0} of {shape[0]} by {shape[1]}, "
          f"{(r1 - r0) * (c1 - c0) / (shape[0] * shape[1]):.0%} of the domain\n")

    cache = args.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        variables, convert = ELEMENTS[name]
        grid = np.full((len(obs["stations"]), len(cycles), len(leads)),
                       np.nan, dtype="float32")
        started = dt.datetime.now()
        missing = 0

        for ci, when in enumerate(cycles):
            part = cache / f"{name}-{when:%Y%m%dT%H}.npy"
            if part.exists():
                grid[:, ci, :] = np.load(part)
            else:
                try:
                    with open_cycle(when) as ds:
                        stack = []
                        for var in variables:
                            da = ds[var].isel(
                                y=slice(r0, r1), x=slice(c0, c1))
                            # Some fields carry a singleton height dimension.
                            da = da.squeeze(drop=True)
                            values = np.asarray(da.values, dtype="float64")
                            want = [h // args.step for h in leads]
                            values = values[want]
                            stack.append(values[:, rr, cc])
                        if len(stack) == 2:      # wind components
                            block = np.hypot(stack[0], stack[1])
                        else:
                            block = stack[0]
                        if name == "precipitation":
                            # Accumulated from the start of the run.
                            block = np.diff(block, axis=0,
                                            prepend=np.zeros((1, block.shape[1])))
                            block = np.maximum(block, 0.0)
                        block = convert(block).T.astype("float32")
                except Exception as exc:  # noqa: BLE001
                    missing += 1
                    print(f"  {when:%Y-%m-%d} unavailable: "
                          f"{type(exc).__name__}: {str(exc)[:90]}")
                    block = np.full((len(obs["stations"]), len(leads)), np.nan,
                                    dtype="float32")
                np.save(part, block)
                grid[:, ci, :] = block

            if (ci + 1) % 10 == 0 or ci == len(cycles) - 1:
                per = (dt.datetime.now() - started).total_seconds() / (ci + 1)
                left = dt.timedelta(seconds=int(per * (len(cycles) - ci - 1)))
                print(f"  {name}: {ci + 1}/{len(cycles)}  {per:.1f}s each  "
                      f"about {left} left")

        out = args.out / f"{name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, values=grid,
            stations=obs["stations"], lat=obs["lat"], lon=obs["lon"],
            distance_km=dist,
            cycles=np.array([f"{c:%Y-%m-%dT%H:%M:%S}" for c in cycles]),
            leads=np.array(leads), element=name,
            accumulation="differenced from run start" if name == "precipitation"
            else "instantaneous",
        )
        filled = float(np.isfinite(grid).mean())
        print(f"\n  {name}: {grid.shape[0]} gauges x {grid.shape[1]} cycles "
              f"x {grid.shape[2]} leads, {filled:.1%} present, "
              f"{missing} cycle(s) unavailable")
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

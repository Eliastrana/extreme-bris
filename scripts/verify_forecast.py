#!/usr/bin/env python3
"""Score a Bris forecast against ERA5, and plot how the error grows.

    ~/bris-env/.venv/bin/python scripts/verify_forecast.py \
        --forecast ~/bris-runs/20250401T00Z/global_*.nc \
        --truth    ~/bris-runs/truth/era5-truth-20250401T00Z.nc \
        -o results/verify

Uses the GLOBAL output file, not the Nordic one. That is deliberate: bris
already interpolates the global domain to 0.25 degrees, which is the grid ERA5
single-levels comes on, so the comparison needs no regridding and invents no
resolution. Scoring the 2.5 km Nordic file against 31 km truth would mostly
measure ERA5's smoothness and report it as model error.

Two things make the numbers mean something rather than just exist:

  PERSISTENCE. The forecast's own t=0 state, held constant, scored the same way.
  An RMSE with nothing to compare it to is a number, not a result. Beating
  persistence is the minimum bar for a forecast having done anything at all,
  and at short lead times it is a surprisingly hard bar.

  THE CUTOUT MASK. The global file has no data under the LAM domain - the
  cutout drops 5,480 points there and the interpolation to 0.25 degrees fills
  the hole with artefacts. Scoring over them would be scoring noise, so
  Scandinavia is excluded from the global numbers by default.

WHAT THIS CAN AND CANNOT SHOW. The forecast was initialised from ERA5, and is
verified here against ERA5. That is self-consistent, and it answers "did the
model carry this state forward sensibly". It is NOT Bris's skill: the model was
trained on the operational analysis, and a number from this run does not belong
next to MET's published scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# forecast variable -> (era5 variable(s), scale, label, unit)
PAIRS = {
    "air_temperature_2m":        (("t2m",),         1.0,   "2 m temperature",   "K"),
    "air_pressure_at_sea_level": (("msl",),         0.01,  "MSLP",              "hPa"),
    "wind_speed_10m":            (("u10", "v10"),   1.0,   "10 m wind speed",   "m/s"),
}

LAM_BOX = (50.0, 75.0, -12.0, 45.0)   # lat_min, lat_max, lon_min, lon_max


def pick(ds, names):
    for n in names:
        if n in ds:
            return ds[n]
    return None


def normalise(ds):
    """bris and CDS disagree on coordinate names; settle on lat/lon/time."""
    ren = {}
    for cand, target in (("valid_time", "time"), ("lat", "latitude"), ("lon", "longitude")):
        if cand in ds.dims or cand in ds.coords:
            ren[cand] = target
    ds = ds.rename(ren) if ren else ds
    for drop in ("number", "expver", "realization", "ensemble_member", "height", "height0"):
        if drop in ds.dims and ds.sizes[drop] == 1:
            ds = ds.isel({drop: 0}, drop=True)
    if "longitude" in ds.coords:
        lon = ds["longitude"]
        if float(lon.max()) > 180.0:
            ds = ds.assign_coords(longitude=(((lon + 180) % 360) - 180)).sortby("longitude")
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    return ds


def wind_speed(ds, names, np):
    if len(names) == 1:
        return pick(ds, names)
    u, v = pick(ds, (names[0],)), pick(ds, (names[1],))
    if u is None or v is None:
        return None
    return np.sqrt(u ** 2 + v ** 2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, type=Path, help="global_*.nc from bris")
    ap.add_argument("--truth", required=True, type=Path, help="ERA5 from fetch_verification.py")
    ap.add_argument("-o", "--out", type=Path, default=Path("results/verify"))
    ap.add_argument("--step-hours", type=int, default=6)
    ap.add_argument("--no-mask", action="store_true",
                    help="do NOT exclude the LAM cutout region (scores artefacts)")
    ap.add_argument("--region", choices=["global", "europe"], default="global")
    args = ap.parse_args()

    try:
        import numpy as np
        import xarray as xr
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc.name} missing - run inside the Bris environment.", file=sys.stderr)
        return 1

    fc = normalise(xr.open_dataset(args.forecast))
    tr = normalise(xr.open_dataset(args.truth))
    args.out.mkdir(parents=True, exist_ok=True)

    if "time" not in fc.dims:
        print("ERROR: forecast has no time dimension.", file=sys.stderr)
        return 1

    # Truth onto the forecast grid. Both are 0.25 deg, so on a matching grid
    # this is an alignment and nearest-neighbour is exact. Only fall back to
    # interpolation - which pulls in scipy - if the grids really do differ.
    tol = 0.02
    aligned = tr.reindex(latitude=fc["latitude"], longitude=fc["longitude"],
                         method="nearest", tolerance=tol)
    probe = aligned[list(aligned.data_vars)[0]]
    if bool(probe.isnull().all()):
        print(f"grids do not match within {tol} deg - interpolating instead")
        tr = tr.interp(latitude=fc["latitude"], longitude=fc["longitude"])
    else:
        print("grids match - aligned by nearest neighbour, no interpolation")
        tr = aligned

    nt = fc.sizes["time"]
    leads = [i * args.step_hours for i in range(nt)]
    print(f"forecast : {args.forecast.name}   {nt} steps, +0h .. +{leads[-1]}h")
    print(f"truth    : {args.truth.name}\n")

    weights = np.cos(np.deg2rad(fc["latitude"]))     # area weighting, or the poles dominate

    mask = xr.ones_like(fc["latitude"] * fc["longitude"], dtype=bool)
    if not args.no_mask:
        la0, la1, lo0, lo1 = LAM_BOX
        inside = ((fc["latitude"] >= la0) & (fc["latitude"] <= la1) &
                  (fc["longitude"] >= lo0) & (fc["longitude"] <= lo1))
        mask = mask & ~inside
        print(f"excluding the LAM cutout region: {la0}..{la1}N, {lo0}..{lo1}E")
    if args.region == "europe":
        box = ((fc["latitude"] >= 35) & (fc["latitude"] <= 72) &
               (fc["longitude"] >= -25) & (fc["longitude"] <= 45))
        mask = mask & box
        print("restricting to Europe")
    print()

    results = {}
    for fname, (tnames, scale, label, unit) in PAIRS.items():
        f = pick(fc, (fname,))
        t = wind_speed(tr, tnames, np)
        if f is None or t is None:
            print(f"  skipping {label}: not in both files")
            continue
        f = f.squeeze() * scale
        t = t.squeeze() * scale

        rmse, bias, pers = [], [], []
        base = f.isel(time=0)                        # the persistence forecast
        for i in range(nt):
            fi, ti = f.isel(time=i), t.isel(time=i)
            d = (fi - ti).where(mask)
            p = (base - ti).where(mask)
            w = weights.broadcast_like(d).where(d.notnull())
            rmse.append(float(np.sqrt(((d ** 2) * w).sum() / w.sum())))
            bias.append(float((d * w).sum() / w.sum()))
            wp = weights.broadcast_like(p).where(p.notnull())
            pers.append(float(np.sqrt(((p ** 2) * wp).sum() / wp.sum())))
        results[fname] = dict(label=label, unit=unit, rmse=rmse, bias=bias,
                              pers=pers, f=f, t=t)

        print(f"{label}  [{unit}]")
        print(f"  {'lead':>6} {'RMSE':>9} {'bias':>9} {'persist':>9}   verdict")
        for i, lead in enumerate(leads):
            better = "" if i == 0 else ("beats persistence" if rmse[i] < pers[i]
                                        else "WORSE than persistence")
            print(f"  {'+' + str(lead) + 'h':>6} {rmse[i]:9.3f} {bias[i]:9.3f} "
                  f"{pers[i]:9.3f}   {better}")
        print()

    if not results:
        print("Nothing scored - no variable pairs matched.", file=sys.stderr)
        return 2

    # --- error growth ---------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (_, r) in zip(axes[0], results.items()):
        ax.plot(leads, r["rmse"], "o-", label="Bris", lw=2)
        ax.plot(leads, r["pers"], "s--", label="persistence", lw=1.5, alpha=.7)
        ax.set_xlabel("lead time [h]")
        ax.set_ylabel(f"RMSE [{r['unit']}]")
        ax.set_title(r["label"])
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle(f"Error growth vs ERA5 — {args.forecast.stem}", y=1.02)
    fig.tight_layout()
    p1 = args.out / f"error_growth_{args.forecast.stem}.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- maps at the longest lead --------------------------------------------
    last = nt - 1
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.6 * n), squeeze=False)
    for row, (_, r) in enumerate(results.items()):
        fi = r["f"].isel(time=last)
        ti = r["t"].isel(time=last)
        # Masked the same way the scores are, so the map shows what was measured
        # rather than a difference field dominated by the cutout artefacts.
        di = (fi - ti).where(mask)
        lim = float(np.nanpercentile(np.abs(di.values), 99))
        for col, (da, title, kw) in enumerate((
                (fi, "Bris", {}), (ti, "ERA5", {}),
                (di, "Bris - ERA5", dict(cmap="RdBu_r", vmin=-lim, vmax=lim)))):
            ax = axes[row][col]
            vals = da.values
            if col < 2:
                lo, hi = np.nanpercentile(np.concatenate([fi.values.ravel(),
                                                          ti.values.ravel()]), [1, 99])
                kw = dict(vmin=lo, vmax=hi)
            m = ax.pcolormesh(da["longitude"], da["latitude"], vals, shading="auto", **kw)
            fig.colorbar(m, ax=ax, shrink=.8, label=r["unit"])
            ax.set_title(f"{r['label']} — {title}  (+{leads[last]}h)", fontsize=10)
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
    fig.tight_layout()
    p2 = args.out / f"maps_{args.forecast.stem}.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print("\nRead the error growth plot first. A forecast that tracks persistence")
    print("at every lead has not forecast anything - it has returned its input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

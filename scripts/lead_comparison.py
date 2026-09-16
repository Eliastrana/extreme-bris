#!/usr/bin/env python3
"""Compare ensemble forecasts with the MEPS analysis, one lead time at a time.

    scripts/lead_comparison.py --zarr ~/bris-data/meps-2p5km-year2-6h-v1.zarr \
        baseline=~/bris-runs/cpu-smoke/20250708T00Z-baseline-m4/nordic_*.nc \
        control=~/bris-runs/cpu-smoke/20250708T00Z-control-m4/nordic_*.nc

WHAT IT IS FOR. The arms train with a rollout of one step, so the loss only
ever rewards the six hours ahead. A model can get better at that and worse at
thirty hours, and a validation loss computed at six hours would never show it.
Scoring each lead time separately does: if fine-tuning helps at +6 h and the
advantage shrinks or turns as the lead grows, the rollout is the likely reason.

AGAINST WHAT. The MEPS analysis at the same valid time, over the Nordic grid
the model writes, the same truth the model trains on. Rain is the six-hour
amount ending at the valid time, which is what each forecast step holds. Wind
is the speed from the two components, so their rotation does not matter.

WHAT IT REPORTS, per field and lead time: fair CRPS of the ensemble (the
training score, lower is better), the range of absolute error over members,
and for rain the grid points forecast at 10 mm or more in six hours where the
analysis had less. Error ranges that do not overlap mean the difference is
larger than what separates two members of the same model.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy", "zarr")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import zarr  # noqa: E402

TRIM = 50          # the dataloader's trim_edge; the forecast grid is the analysis grid minus this
FIELDS = [
    # label, forecast variable, analysis variable(s), scale to report units, unit
    ("rain", "precipitation_amount", ("tp",), 1.0, "mm/6h"),
    ("t2m", "air_temperature_2m", ("2t",), 1.0, "K"),
    ("mslp", "air_pressure_at_sea_level", ("msl",), 0.01, "hPa"),
    ("wind10", "wind_speed_10m", ("10u", "10v"), 1.0, "m/s"),
]


def crps(ens: np.ndarray, ob: np.ndarray) -> float:
    m = ens.shape[0]
    skill = np.abs(ens - ob[None]).mean(axis=0)
    spread = np.abs(ens[:, None] - ens[None, :]).sum(axis=(0, 1)) / (2 * m * (m - 1))
    return float((skill - spread).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forecasts", nargs="+", help="label=path (a glob is fine) to a nordic NetCDF file")
    ap.add_argument("--zarr", type=Path, required=True, help="MEPS dataset holding the analysis")
    ap.add_argument("--points", type=int, default=200_000, help="grid points sampled per field")
    args = ap.parse_args()

    runs = {}
    for item in args.forecasts:
        label, pattern = item.split("=", 1)
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if len(matches) != 1:
            raise SystemExit(f"{label}: {len(matches)} files match {pattern}")
        runs[label] = xr.open_dataset(matches[0])

    first = next(iter(runs.values()))
    times = first["time"].values.astype("datetime64[s]")
    ny, nx = first.sizes["y"], first.sizes["x"]

    z = zarr.open(str(args.zarr.expanduser()), mode="r")
    names = z.attrs["variables"]
    zdates = z["dates"][:].astype("datetime64[s]")
    NY, NX = ny + 2 * TRIM, nx + 2 * TRIM
    zlat = np.asarray(z["latitudes"][:]).reshape(NY, NX)[TRIM:-TRIM, TRIM:-TRIM]
    off = float(np.abs(zlat - first["latitude"].values).max())
    if off > 1e-4:
        raise SystemExit(f"forecast grid is not the trimmed analysis grid (latitudes differ by {off})")

    rng = np.random.default_rng(0)
    sample = rng.choice(ny * nx, size=min(args.points, ny * nx), replace=False)

    def analysis(var_names, t):
        i = int(np.where(zdates == t)[0][0])
        parts = [np.asarray(z["data"][i, names.index(v), 0, :], dtype="float64")
                 .reshape(NY, NX)[TRIM:-TRIM, TRIM:-TRIM].ravel() for v in var_names]
        return np.hypot(*parts) if len(parts) == 2 else parts[0]

    def members(ds, var, k):
        da = ds[var].isel(time=k)
        extra = [d for d in da.dims if d not in ("y", "x") and "ensemble" not in d]
        da = da.squeeze(extra, drop=True)
        mdim = next((d for d in da.dims if "ensemble" in d), None)
        if mdim is None:
            da = da.expand_dims("ensemble_member")
            mdim = "ensemble_member"
        return np.asarray(da.transpose(mdim, "y", "x").values, dtype="float64").reshape(da.sizes[mdim], -1)

    labels = list(runs)
    for field, var, zvars, scale, unit in FIELDS:
        if any(var not in ds for ds in runs.values()):
            print(f"--- {field}: not in every file, skipped\n")
            continue
        print(f"--- {field} ({unit}), against the MEPS analysis at the valid time")
        head = f"{'lead':>5s}" + "".join(f" {l + ' CRPS':>15s}" for l in labels)
        head += f" {'change':>8s}" + "".join(f" {l + ' MAE range':>22s}" for l in labels)
        if field == "rain":
            head += "".join(f" {l + ' false10':>17s}" for l in labels)
        print(head)
        for k in range(1, len(times)):
            ob_full = analysis(zvars, times[k]) * scale
            ob = ob_full[sample]
            row, crpss = f"{int((times[k] - times[0]) / np.timedelta64(1, 'h')):4d}h", []
            ranges, falses = [], []
            for l in labels:
                ens_full = members(runs[l], var, k) * scale
                ens = ens_full[:, sample]
                c = crps(ens, ob) if ens.shape[0] > 1 else float("nan")
                crpss.append(c)
                maes = np.abs(ens - ob[None]).mean(axis=1)
                ranges.append(f"{maes.min():.3f}..{maes.max():.3f}")
                if field == "rain":
                    f = ((ens_full >= 10) & (ob_full[None] < 10)).sum(axis=1)
                    falses.append(f"{f.min()}..{f.max()}")
            row += "".join(f" {c:15.4f}" for c in crpss)
            change = (crpss[-1] - crpss[0]) / crpss[0] * 100 if len(crpss) > 1 and crpss[0] else float("nan")
            row += f" {change:+7.1f}%" + "".join(f" {r:>22s}" for r in ranges)
            if field == "rain":
                row += "".join(f" {f:>17s}" for f in falses)
            print(row)
        print()
    print(f"change = ({labels[-1]} - {labels[0]}) / {labels[0]}, CRPS; negative means {labels[-1]} is better")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

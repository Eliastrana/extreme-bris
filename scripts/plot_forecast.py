#!/usr/bin/env python3
"""Plot the forecast fields, and the LAM/global seam in particular.

The numeric checks confirm the fields are in physical ranges. They cannot see
the one thing most likely to be wrong after assembling a cutout from datasets
built by hand: a visible discontinuity where the MEPS domain meets the global
field, or wind that is rotated by a constant angle.

    ~/bris-env/.venv/bin/python scripts/plot_forecast.py ~/bris-runs/smoke/*.nc

Writes PNGs beside the NetCDF files. Runs on the login node — plotting is CPU
work and needs no allocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

FIELDS = [
    ("air_temperature_2m", "2 m temperature", "K", "RdYlBu_r"),
    ("air_pressure_at_sea_level", "Mean sea-level pressure", "Pa", "viridis"),
    ("wind_speed_10m", "10 m wind speed", "m/s", "YlGnBu"),
    ("precipitation_amount", "Precipitation", "mm", "Blues"),
]


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__, file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    for path in paths:
        ds = xr.open_dataset(path)
        # Last step: precipitation is undefined at lead time 0.
        t = ds.sizes.get("time", 1) - 1
        present = [f for f in FIELDS if f[0] in ds.data_vars]
        if not present:
            print(f"{path.name}: none of the expected fields", file=sys.stderr)
            continue

        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        for ax, (name, label, unit, cmap) in zip(axes.ravel(), present):
            da = ds[name].isel(time=t).squeeze()
            arr = np.asarray(da.values, dtype="float64")
            finite = np.isfinite(arr)
            if not finite.any():
                ax.set_title(f"{label} — all NaN")
                ax.axis("off")
                continue

            # Percentile limits: a single extreme cell otherwise flattens
            # everything else to one colour.
            lo, hi = np.nanpercentile(arr[finite], [1, 99])
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                           interpolation="nearest", aspect="auto")
            fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
            ax.set_title(f"{label}\n{np.nanmin(arr):.4g} .. {np.nanmax(arr):.4g} {unit}")
            ax.set_xticks([]); ax.set_yticks([])

        domain = "Nordic (MEPS domain)" if "nordic" in path.name else "Global (0.25 deg)"
        fig.suptitle(f"{domain} — step {t}\n{path.name}", fontsize=11)
        out = path.with_suffix(".png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")

        # Wind direction against the pressure gradient. Speed alone cannot
        # show a rotation error, since the magnitude is unchanged by one.
        if {"10u", "10v"} <= set(ds.data_vars) and "air_pressure_at_sea_level" in ds.data_vars:
            u = np.asarray(ds["10u"].isel(time=t).squeeze().values, dtype="float64")
            v = np.asarray(ds["10v"].isel(time=t).squeeze().values, dtype="float64")
            p_ = np.asarray(ds["air_pressure_at_sea_level"].isel(time=t).squeeze().values,
                            dtype="float64")
            gy, gx = np.gradient(p_)
            # In the northern hemisphere geostrophic flow runs anticlockwise
            # around low pressure, so wind should sit close to 90 degrees from
            # the pressure gradient. A systematic offset from that is the
            # signature of an unrotated or wrongly rotated wind field.
            ang = np.degrees(np.arctan2(v, u) - np.arctan2(-gy, -gx))
            ang = (ang + 180) % 360 - 180
            ok = np.isfinite(ang) & np.isfinite(p_)
            if ok.any():
                med = float(np.median(ang[ok]))
                print(f"  wind vs pressure gradient: median {med:+.1f} deg "
                      f"(expect roughly +90 in the northern hemisphere)")

    print()
    print("What to look for:")
    print("  * a visible edge in the global plot where the MEPS domain sits —")
    print("    the cutout should blend, not show a rectangle over Scandinavia")
    print("  * wind that follows the pressure field. A constant rotation offset")
    print("    is the signature of the grid-relative winds not being rotated,")
    print("    and it looks entirely plausible until compared with the isobars")
    print("  * temperature that follows orography and latitude rather than the")
    print("    grid axes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

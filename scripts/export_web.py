#!/usr/bin/env python3
"""Turn a Bris NetCDF field into web-ready map assets.

    ~/bris-env/.venv/bin/python scripts/export_web.py \
        --forecast ~/bris-runs/20250401T00Z-120h/nordic_*.nc \
        --var air_temperature_2m --stride 2 -o web/nordic

Writes one PNG per timestep plus a manifest JSON, small enough to commit and
serve as a Mapbox `image` source. The NetCDF itself never reaches the browser -
it is 283 MB, it is HDF5 underneath, and no JS library reads that reliably.

THE PROJECTION, WHICH IS THE WHOLE PROBLEM
------------------------------------------
Mapbox draws in Web Mercator. The MEPS grid is Lambert conformal, delivered as
2D latitude/longitude arrays. An image source is placed by its four corners and
interpolated LINEARLY IN MERCATOR SPACE between them, so the raster handed over
must already be uniform in Mercator y - not uniform in latitude.

Getting this wrong does not look wrong. An equirectangular raster dropped onto a
Mercator map is stretched north-south by a factor that grows with latitude;
Scandinavia lands a few hundred kilometres off and the picture still reads as a
plausible weather field. That is the failure worth guarding against, so the
manifest records the projection explicitly.

Resampling is a forward scatter: every source cell is binned into the target
pixel it falls in, and the pixels are averaged. No scipy, no interpolation
inventing values between grid cells. The Lambert domain is a rotated
quadrilateral, so the corners of the lon/lat bounding box have no data - those
are written transparent rather than filled.

The colour scale is computed across EVERY exported step, not per step. Scaling
each frame to its own range makes a time slider flicker in a way that reads as
weather.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# display conversions: stored unit -> what a reader expects
CONVERT = {
    "air_temperature_2m":        (lambda a: a - 273.15, "°C", "RdYlBu_r"),
    "air_pressure_at_sea_level": (lambda a: a / 100.0,  "hPa", "viridis"),
    "wind_speed_10m":            (lambda a: a,          "m/s", "YlGnBu"),
    "precipitation_amount":      (lambda a: a,          "mm",  "Blues"),
}


def mercator_y(lat_deg, np):
    """Web Mercator y, clamped short of the poles where it diverges."""
    lat = np.clip(lat_deg, -85.05112878, 85.05112878)
    r = np.deg2rad(lat)
    return np.log(np.tan(np.pi / 4 + r / 2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, type=Path, help="nordic_*.nc")
    ap.add_argument("--var", default="air_temperature_2m")
    ap.add_argument("--stride", type=int, default=1,
                    help="export every Nth timestep (default 1)")
    ap.add_argument("--width", type=int, default=0,
                    help="output raster width; default follows the source grid, "
                         "because a target finer than the source turns forward "
                         "scatter into a sieve")
    ap.add_argument("-o", "--out", type=Path, default=Path("web/nordic"))
    args = ap.parse_args()

    try:
        import numpy as np
        import xarray as xr
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import cm, colors
        from matplotlib.image import imsave
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc.name} missing - run inside the Bris environment.",
              file=sys.stderr)
        return 1

    ds = xr.open_dataset(args.forecast)
    if args.var not in ds:
        print(f"ERROR: {args.var} not in {args.forecast.name}", file=sys.stderr)
        print(f"  have: {', '.join(list(ds.data_vars)[:12])}", file=sys.stderr)
        return 1

    convert, unit, cmap_name = CONVERT.get(args.var, (lambda a: a, "", "viridis"))
    lat = np.asarray(ds["latitude"].values, dtype="float64").ravel()
    lon = np.asarray(ds["longitude"].values, dtype="float64").ravel()
    times = [np.datetime64(t, "s").astype(object) for t in ds["time"].values]
    steps = list(range(0, len(times), args.stride))

    # --- target raster, uniform in Mercator ----------------------------------
    lon0, lon1 = float(lon.min()), float(lon.max())
    lat0, lat1 = float(lat.min()), float(lat.max())
    y0, y1 = float(mercator_y(lat0, np)), float(mercator_y(lat1, np))
    # A target finer than the source leaves holes: forward scatter fills the
    # pixel a source cell lands in and nothing between. Follow the source grid
    # unless told otherwise.
    src_nx = ds[args.var].shape[-1]
    W = args.width if args.width > 0 else min(1400, max(300, src_nx))
    # keep pixels square in Mercator space
    H = max(1, int(round(W * (y1 - y0) / np.deg2rad(lon1 - lon0))))

    px = ((lon - lon0) / (lon1 - lon0) * (W - 1)).round().astype("int64")
    py = ((mercator_y(lat, np) - y0) / (y1 - y0) * (H - 1)).round().astype("int64")
    py = (H - 1) - py                      # image rows run north to south
    np.clip(px, 0, W - 1, out=px)
    np.clip(py, 0, H - 1, out=py)
    flat = py * W + px

    print(f"field    : {args.var}  [{unit}]")
    print(f"source   : {lat.size:,} points, Lambert 2D lat/lon")
    print(f"target   : {W} x {H} Web Mercator")
    print(f"bounds   : {lat0:.3f}..{lat1:.3f} N, {lon0:.3f}..{lon1:.3f} E")
    print(f"steps    : {len(steps)} of {len(times)} (stride {args.stride})\n")

    # --- one pass to fix the colour scale across every exported step ---------
    frames = []
    for k in steps:
        vals = convert(np.asarray(ds[args.var].isel(time=k).squeeze().values,
                                  dtype="float64").ravel())
        total = np.zeros(W * H)
        count = np.zeros(W * H)
        good = np.isfinite(vals)
        np.add.at(total, flat[good], vals[good])
        np.add.at(count, flat[good], 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            img = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        img = img.reshape(H, W)
        # One pass closing single-pixel holes from the scatter. Only pixels with
        # at least three filled neighbours are filled, so this smooths the sieve
        # without painting over the domain edge or inventing a region.
        hole = ~np.isfinite(img)
        if hole.any():
            pad = np.pad(np.nan_to_num(img, nan=0.0), 1)
            cnt = np.pad(np.isfinite(img).astype(float), 1)
            tot = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
            num = (cnt[:-2, 1:-1] + cnt[2:, 1:-1] + cnt[1:-1, :-2] + cnt[1:-1, 2:])
            fill = hole & (num >= 3)
            img = np.where(fill, tot / np.maximum(num, 1), img)
        frames.append(img)

    cover0 = float(np.isfinite(frames[0]).mean())
    if cover0 < 0.55:
        print(f"WARNING: only {cover0 * 100:.0f}% of the raster has data. The "
              f"target is finer than\n         the source grid - lower --width.")
    stack = np.concatenate([f[np.isfinite(f)].ravel() for f in frames])
    vmin, vmax = (float(np.percentile(stack, 1)), float(np.percentile(stack, 99)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-9:
        # A constant field would divide by zero in the normalisation and every
        # pixel would come out the same colour, which reads as a working export.
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
        pad = max(abs(hi) * 1e-3, 0.5)
        vmin, vmax = lo - pad, hi + pad
        print(f"WARNING: the 1st and 99th percentiles are equal - the field is "
              f"near-constant.\n         Widened to {vmin:.3f}..{vmax:.3f} so the "
              f"image is not a single flat colour.")
    print(f"colour   : {cmap_name}, {vmin:.2f} .. {vmax:.2f} {unit} "
          f"(1st/99th percentile over all exported steps)\n")

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name) if hasattr(cm, "get_cmap") else matplotlib.colormaps[cmap_name]

    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for n, (k, img) in enumerate(zip(steps, frames)):
        rgba = cmap(norm(img))
        rgba[..., 3] = np.where(np.isfinite(img), 1.0, 0.0)   # holes stay clear
        name = f"{args.var}_{n:02d}.png"
        imsave(args.out / name, rgba)
        cover = float(np.isfinite(img).mean())
        entries.append({"step": k, "lead_hours": k * 6,
                        "valid": times[k].strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "image": name})
        print(f"  +{k * 6:3d}h  {name}  ({cover * 100:.0f}% covered, "
              f"{(args.out / name).stat().st_size / 1024:.0f} KB)")

    swatches = [{"value": round(float(vmin + (vmax - vmin) * f), 2),
                 "color": colors.to_hex(cmap(norm(vmin + (vmax - vmin) * f)))}
                for f in (0, .25, .5, .75, 1)]

    manifest = {
        "variable": args.var,
        "unit": unit,
        "projection": "EPSG:3857",
        "note": ("Raster is uniform in Web Mercator y, not in latitude. Place it "
                 "as a Mapbox image source using `coordinates` below, which are "
                 "the corners in [lon, lat] order: TL, TR, BR, BL."),
        "coordinates": [[lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]],
        "bounds": {"west": lon0, "east": lon1, "south": lat0, "north": lat1},
        "width": W, "height": H,
        "vmin": round(vmin, 3), "vmax": round(vmax, 3),
        "colormap": cmap_name,
        "legend": swatches,
        "initialised": times[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": args.forecast.name,
        "caveat": ("Initialised from ERA5, not the operational analysis Bris was "
                   "trained on. Not a skill estimate."),
        "frames": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_kb = sum((args.out / e["image"]).stat().st_size for e in entries) / 1024
    print(f"\nwrote {args.out}/manifest.json")
    print(f"total {total_kb:.0f} KB across {len(entries)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

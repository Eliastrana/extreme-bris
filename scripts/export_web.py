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

# How strongly a value should paint, separate from its colour. A field like
# temperature covers the whole domain with a meaningful number, so every
# finite pixel is opaque. Precipitation does not: two thirds of pixels are
# exactly 0 mm, and drawing 0 mm at full opacity paints the driest cell the
# same strength as a wet one - the map reads as a white haze sitting over the
# whole domain instead of showing the basemap where nothing is falling. Ramp
# opacity in below half a millimetre instead, so dry areas are transparent
# and only actual rain covers the map.
ALPHA = {
    # numpy is imported inside main(), not at module scope, so this leans on
    # the ndarray's own .clip() rather than calling np.clip() by name.
    "precipitation_amount": lambda a: (a / 0.5).clip(0.0, 1.0),
}


def default_alpha(a):
    return 1.0


def mercator_y(lat_deg, np):
    """Web Mercator y, clamped short of the poles where it diverges."""
    lat = np.clip(lat_deg, -85.05112878, 85.05112878)
    r = np.deg2rad(lat)
    return np.log(np.tan(np.pi / 4 + r / 2))


def covered_by(lower, upper, np):
    """Which pixels of `lower` sit underneath `upper`'s domain, geographically.

    Each pixel of the lower raster is turned back into lon/lat and looked up
    in the upper raster - the same inverse sampling the global field already
    uses, run on the domain mask instead of the values.

    The upper domain is the union over timesteps rather than any single one.
    A dry cell is 0.0, not NaN, so for most fields one step would do; but
    precipitation has no value at all at analysis time, and taking the mask
    from that step alone would say the LAM covers nothing.
    """
    W, H = lower["W"], lower["H"]
    y0 = mercator_y(lower["lat0"], np)
    y1 = mercator_y(lower["lat1"], np)
    lon = lower["lon0"] + (np.arange(W) / max(W - 1, 1)) * (lower["lon1"] - lower["lon0"])
    yy = y0 + ((H - 1 - np.arange(H)) / max(H - 1, 1)) * (y1 - y0)
    lat = np.degrees(2 * np.arctan(np.exp(yy)) - np.pi / 2)

    dom = np.zeros((upper["H"], upper["W"]), dtype=bool)
    for g in upper["grids"]:
        dom |= np.isfinite(g)

    uy0 = mercator_y(upper["lat0"], np)
    uy1 = mercator_y(upper["lat1"], np)
    ux = (lon[None, :] - upper["lon0"]) / (upper["lon1"] - upper["lon0"]) * (upper["W"] - 1)
    uy = (mercator_y(lat, np)[:, None] - uy0) / (uy1 - uy0) * (upper["H"] - 1)
    uy = (upper["H"] - 1) - uy

    xi = np.rint(ux).astype("int64")
    yi = np.rint(uy).astype("int64")
    inside = ((xi >= 0) & (xi < upper["W"])) & ((yi >= 0) & (yi < upper["H"]))
    return inside & dom[np.clip(yi, 0, upper["H"] - 1), np.clip(xi, 0, upper["W"] - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, type=Path, nargs="+",
                    help="one or more NetCDF files. Give the global one first "
                         "and the LAM second: draw order follows this order, and "
                         "the colour scale is computed across all of them.")
    ap.add_argument("--bbox", default=None,
                    metavar="W,S,E,N",
                    help="crop to west,south,east,north. Worth using on the "
                         "global file: a full-globe Mercator raster spends most "
                         "of its pixels on ocean the map never shows. Pass it "
                         "with an equals sign when west is negative - "
                         "--bbox=-60,25,60,85 - or argparse reads the leading "
                         "minus as another flag.")
    ap.add_argument("--var", default="air_temperature_2m")
    ap.add_argument("--stride", type=int, default=1,
                    help="export every Nth timestep (default 1)")
    ap.add_argument("--width", type=int, default=0,
                    help="output raster width; default follows the source grid, "
                         "because a target finer than the source turns forward "
                         "scatter into a sieve")
    # Was hardcoded to the ERA5 wording, which silently became a false claim
    # the moment the same script exported an operational-analysis run.
    ap.add_argument("--caveat",
                    default="Initialised from ERA5, not the operational analysis "
                            "Bris was trained on. Not a skill estimate.",
                    help="what the manifest should say about this run's "
                         "provenance. Change it when the inputs change")
    ap.add_argument("--vmin", type=float, default=None,
                    help="pin the low end of the colour scale instead of "
                         "taking the 1st percentile. Use when exporting a "
                         "second run that has to be comparable with the first")
    ap.add_argument("--vmax", type=float, default=None,
                    help="pin the high end; see --vmin")
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
    convert, unit, cmap_name = CONVERT.get(args.var, (lambda a: a, "", "viridis"))
    alpha_of = ALPHA.get(args.var, default_alpha)

    bbox = None
    if args.bbox:
        w, s_, e, n = (float(x) for x in args.bbox.split(","))
        bbox = (w, s_, e, n)

    # --- resample each input, sharing nothing yet but the geometry ----------
    layers = []
    for path in args.forecast:
        ds = xr.open_dataset(path)
        if args.var not in ds:
            print(f"ERROR: {args.var} not in {path.name}", file=sys.stderr)
            print(f"  have: {', '.join(list(ds.data_vars)[:12])}", file=sys.stderr)
            return 1

        lat_v = np.asarray(ds["latitude"].values, dtype="float64")
        lon_v = np.asarray(ds["longitude"].values, dtype="float64")
        # The LAM file carries 2D Lambert coordinates; the global one is a
        # regular lat/lon grid with 1D axes. Mesh the second so both are
        # per-cell and the rest of the code needs no special case.
        regular = lat_v.ndim == 1 and lon_v.ndim == 1
        axes = (lat_v.copy(), lon_v.copy()) if regular else None
        if regular:
            lat_v, lon_v = np.meshgrid(lat_v, lon_v, indexing="ij")
        lat, lon = lat_v.ravel(), lon_v.ravel()

        keep = np.ones(lat.shape, dtype=bool)
        if bbox:
            w, s_, e, n = bbox
            keep = (lon >= w) & (lon <= e) & (lat >= s_) & (lat <= n)
            if not keep.any():
                print(f"ERROR: --bbox excludes all of {path.name}", file=sys.stderr)
                return 1

        lon0, lon1 = float(lon[keep].min()), float(lon[keep].max())
        lat0, lat1 = float(lat[keep].min()), float(lat[keep].max())
        y0, y1 = float(mercator_y(lat0, np)), float(mercator_y(lat1, np))
        # mercator_y clamps short of the poles, where the projection diverges.
        # The corners written to the manifest must be the latitudes the raster
        # ACTUALLY spans, not the data's raw extremes: a global field reaching
        # 90 degrees is drawn to 85.05, and reporting 90 would have Mapbox
        # stretch the image between the wrong parallels and shift everything.
        lat0 = float(np.degrees(2 * np.arctan(np.exp(y0)) - np.pi / 2))
        lat1 = float(np.degrees(2 * np.arctan(np.exp(y1)) - np.pi / 2))

        # Do not ask for more pixels than there is data. Mercator stretches
        # rows apart towards the poles, so a width that looks fine from the
        # source's column count leaves horizontal gaps at high latitude - the
        # global field is the one this bites. Cap so the target holds roughly
        # one source cell per pixel, given the aspect the projection forces.
        aspect = (y1 - y0) / np.deg2rad(lon1 - lon0)
        if regular:
            # Inverse sampling has no gaps by construction, so resolution is
            # only a question of file size.
            w_default = min(1200, max(400, ds[args.var].shape[-1]))
        else:
            # Forward scatter must not be asked for more pixels than it has
            # cells to fill them with.
            n_src = int(keep.sum())
            w_default = max(240, min(1400, int(np.sqrt(max(n_src, 1) / max(aspect, 1e-6)))))
        W = args.width if args.width > 0 else w_default
        H = max(1, int(round(W * aspect)))

        px = ((lon - lon0) / (lon1 - lon0) * (W - 1)).round().astype("int64")
        py = ((mercator_y(lat, np) - y0) / (y1 - y0) * (H - 1)).round().astype("int64")
        py = (H - 1) - py
        np.clip(px, 0, W - 1, out=px)
        np.clip(py, 0, H - 1, out=py)
        flat = py * W + px

        times = [np.datetime64(t, "s").astype(object) for t in ds["time"].values]
        steps = list(range(0, len(times), args.stride))
        name = "nordic" if "nordic" in path.name else (
            "global" if "global" in path.name else path.stem)

        print(f"--- {name}: {path.name}")
        print(f"    {lat.size:,} cells -> {W} x {H} Mercator")
        print(f"    {lat0:.2f}..{lat1:.2f} N, {lon0:.2f}..{lon1:.2f} E, "
              f"{len(steps)} steps")

        # Two resampling paths, because one algorithm cannot serve both.
        #
        # A REGULAR lat/lon source is sampled INVERSELY: every target pixel
        # computes its own lat/lon and looks the source up. Forward scatter
        # fails badly here - Mercator packs target rows into ever smaller
        # latitude increments towards the pole, so above roughly 70N most rows
        # fall between two source rows and stay empty. The raster comes out
        # barred, and no amount of hole filling repairs whole missing rows.
        # The arithmetic is unforgiving: at 85N a 0.25 degree source supports
        # about fifty Mercator rows before gaps appear.
        #
        # The CURVILINEAR Lambert source keeps forward scatter, which is right
        # there: the target is coarser than the source, there is no analytic
        # inverse, and every target pixel receives several source cells.
        inv = None
        if regular:
            src_lat, src_lon = axes
            yy = y0 + ((H - 1 - np.arange(H)) / max(H - 1, 1)) * (y1 - y0)
            tgt_lat = np.degrees(2 * np.arctan(np.exp(yy)) - np.pi / 2)
            tgt_lon = lon0 + (np.arange(W) / max(W - 1, 1)) * (lon1 - lon0)
            j_idx = np.abs(src_lat[None, :] - tgt_lat[:, None]).argmin(axis=1)
            i_idx = np.abs(src_lon[None, :] - tgt_lon[:, None]).argmin(axis=1)
            inv = (j_idx, i_idx)

        grids = []
        for k in steps:
            raw = np.asarray(ds[args.var].isel(time=k).squeeze().values,
                             dtype="float64")
            if inv is not None:
                img = convert(raw[np.ix_(inv[0], inv[1])])
                if bbox:
                    outside = ((tgt_lon[None, :] < bbox[0]) |
                               (tgt_lon[None, :] > bbox[2]) |
                               (tgt_lat[:, None] < bbox[1]) |
                               (tgt_lat[:, None] > bbox[3]))
                    img = np.where(outside, np.nan, img)
                grids.append(img)
                continue
            vals = convert(raw.ravel())
            total = np.zeros(W * H)
            count = np.zeros(W * H)
            good = np.isfinite(vals) & keep
            np.add.at(total, flat[good], vals[good])
            np.add.at(count, flat[good], 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                img = np.where(count > 0, total / np.maximum(count, 1), np.nan)
            img = img.reshape(H, W)
            # One pass closing single-pixel holes from the scatter. Only pixels
            # with at least three filled neighbours, so the domain edge is not
            # painted over and no region is invented.
            # Close the gaps the scatter leaves. Two passes at two neighbours,
            # not one pass at three: Mercator stretches rows apart towards the
            # poles, so the gaps arrive as whole empty ROWS. A pixel in an empty
            # row has exactly two filled neighbours - the one above and the one
            # below - so a three-neighbour rule can never close a stripe, which
            # is why the global field came out barred.
            for _ in range(2):
                hole = ~np.isfinite(img)
                if not hole.any():
                    break
                pad = np.pad(np.nan_to_num(img, nan=0.0), 1)
                cnt = np.pad(np.isfinite(img).astype(float), 1)
                tot = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
                num = (cnt[:-2, 1:-1] + cnt[2:, 1:-1] + cnt[1:-1, :-2] + cnt[1:-1, 2:])
                # Two is enough to interpolate between; one would smear the
                # domain edge outwards into empty space.
                img = np.where(hole & (num >= 2), tot / np.maximum(num, 1), img)
            grids.append(img)

        cover = float(np.isfinite(grids[0]).mean())
        if cover < 0.35:
            print(f"    WARNING: only {cover * 100:.0f}% of the raster has data. "
                  f"That is either a\n             genuine gap in the field - the "
                  f"cutout leaves one - or the target is\n             finer than "
                  f"the source. Compare against the domain shape before trusting it.")
        layers.append({"name": name, "grids": grids, "steps": steps, "times": times,
                       "W": W, "H": H, "lon0": lon0, "lon1": lon1,
                       "lat0": lat0, "lat1": lat1})

    # Time axes have to agree, or the slider shows one domain at one hour and
    # the other at another without saying so.
    lead_sets = {tuple(l["steps"]) for l in layers}
    if len(lead_sets) > 1:
        print("ERROR: the inputs do not share a timestep list.", file=sys.stderr)
        return 1

    # --- punch the upper domains out of the layers below them ----------------
    # The model's cutout already removed the global points under the LAM, so
    # what the global field holds there is not weather: it interpolates to
    # 0.25 degrees with nothing to interpolate from and smears values in from
    # the neighbours, which draws as wedges radiating out of Scandinavia.
    #
    # Painting the LAM on top used to hide that, but only while every pixel of
    # the LAM was opaque. The moment a field is drawn with transparency -
    # precipitation, where two thirds of the domain is dry - the artefact
    # shows straight through the gaps. Removing it here rather than relying on
    # the layer above to cover it means the seam is gone in the data, at any
    # opacity, for every variable.
    for i, lower in enumerate(layers):
        for upper in layers[i + 1:]:
            mask = covered_by(lower, upper, np)
            if not mask.any():
                continue
            for g in lower["grids"]:
                g[mask] = np.nan
            print(f"  masked {mask.mean() * 100:.1f}% of {lower['name']} "
                  f"where {upper['name']} covers it")

    # --- ONE colour scale across every layer and every step ------------------
    # Scaling each domain to its own range puts a colour jump on the seam that
    # is not in the data - the LAM would read as systematically warmer or colder
    # than the global field it sits inside.
    stack = np.concatenate([g[np.isfinite(g)].ravel()
                            for l in layers for g in l["grids"]])
    vmin, vmax = float(np.percentile(stack, 1)), float(np.percentile(stack, 99))
    # Two runs of the SAME variable - ERA5-initialised against analysis-
    # initialised - have to be read against one scale, or the eye reads the
    # difference between two colour scales as a difference between two
    # forecasts. Export one, then pin the other to the numbers it reports.
    if args.vmin is not None:
        vmin = args.vmin
    if args.vmax is not None:
        vmax = args.vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-9:
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
        pad = max(abs(hi) * 1e-3, 0.5)
        vmin, vmax = lo - pad, hi + pad
        print(f"WARNING: percentiles equal - field is near-constant. Widened to "
              f"{vmin:.3f}..{vmax:.3f}.")
    print(f"\nshared colour scale: {cmap_name}, {vmin:.2f} .. {vmax:.2f} {unit}\n")

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]
    args.out.mkdir(parents=True, exist_ok=True)

    out_layers = []
    for l in layers:
        entries = []
        for n, (k, img) in enumerate(zip(l["steps"], l["grids"])):
            rgba = cmap(norm(img))
            rgba[..., 3] = np.where(np.isfinite(img), alpha_of(img), 0.0)
            fname = f"{l['name']}_{args.var}_{n:02d}.png"
            imsave(args.out / fname, rgba)
            entries.append({"step": k, "lead_hours": k * 6,
                            "valid": l["times"][k].strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "image": fname})
        kb = sum((args.out / e["image"]).stat().st_size for e in entries) / 1024
        print(f"  {l['name']:8s} {len(entries):3d} frames, {kb:6.0f} KB")
        out_layers.append({
            "name": l["name"],
            "coordinates": [[l["lon0"], l["lat1"]], [l["lon1"], l["lat1"]],
                            [l["lon1"], l["lat0"]], [l["lon0"], l["lat0"]]],
            "width": l["W"], "height": l["H"],
            "frames": entries,
        })

    swatches = [{"value": round(float(vmin + (vmax - vmin) * f), 2),
                 "color": colors.to_hex(cmap(norm(vmin + (vmax - vmin) * f)))}
                for f in (0, .25, .5, .75, 1)]

    manifest = {
        "variable": args.var,
        "unit": unit,
        "projection": "EPSG:3857",
        "note": ("Each layer's raster is uniform in Web Mercator y, not in "
                 "latitude, and is placed by the four corners in `coordinates` "
                 "([lon, lat]: TL, TR, BR, BL). Draw them in the order given: "
                 "the LAM belongs on top, where it covers the hole the cutout "
                 "leaves in the global field."),
        "vmin": round(vmin, 3), "vmax": round(vmax, 3),
        "colormap": cmap_name,
        "legend": swatches,
        "initialised": layers[0]["times"][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "caveat": args.caveat,
        "layers": out_layers,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

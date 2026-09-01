#!/usr/bin/env python3
"""Bris forecast against actual station observations, as a time series per site.

    export FROST_CLIENT_ID=...        # or put it in ~/.frostrc
    ~/bris-env/.venv/bin/python scripts/plot_stations.py \
        --forecast ~/bris-runs/20250401T00Z/nordic_*.nc \
        -o results/verify

This is the one comparison in the repo that does NOT verify a model against a
model. ERA5 is a reanalysis - a model state fitted to observations - so scoring
against it asks whether Bris agrees with ECMWF's analysis. Frost serves what the
thermometers actually recorded.

It uses the NORDIC file, not the global one. The stations are inside the LAM
domain, where the forecast is 2.5 km and the global file has no data at all -
that region is exactly the hole the cutout leaves.

Sampling is nearest grid point by great-circle distance. No interpolation: at
2.5 km the nearest cell is within ~1.8 km of any station, and interpolating
would blur the terrain detail that is the whole point of running a LAM.

WHAT TO EXPECT. A 2 m temperature forecast is not supposed to sit on top of a
station trace. Stations sit in valleys, by walls, at altitudes the model grid
averages away, and a 2.5 km cell is a mean over 6 km^2. A constant offset at one
site is representativeness, not model error. What matters is whether the SHAPE
matches - the timing of the diurnal swing, the passage of a front, the sign of
a cold advection.

CREDENTIALS. Frost needs a client ID, free and instant from
https://frost.met.no/auth/requestCredentials.html . This script reads it from
$FROST_CLIENT_ID or ~/.frostrc and never prints it. Do not pass it on the
command line - it ends up in your shell history, and eX3 is a shared machine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

FROST = "https://frost.met.no"

# A spread down the country: coastal west, inland east, mid, arctic. Their
# coordinates are resolved from the API rather than hardcoded, so a wrong id
# fails with a clear message instead of silently sampling the wrong cell.
DEFAULT_STATIONS = ["SN18700", "SN50540", "SN68860", "SN90450"]


def client_id() -> str:
    cid = os.environ.get("FROST_CLIENT_ID", "").strip()
    if cid:
        return cid
    rc = Path.home() / ".frostrc"
    if rc.exists():
        cid = rc.read_text().strip().splitlines()[0].strip()
        if cid:
            return cid
    print("ERROR: no Frost client ID.\n", file=sys.stderr)
    print("  Register (free, instant):", file=sys.stderr)
    print("    https://frost.met.no/auth/requestCredentials.html\n", file=sys.stderr)
    print("  Then either:", file=sys.stderr)
    print("    export FROST_CLIENT_ID=<the id>", file=sys.stderr)
    print("  or put it on the first line of ~/.frostrc and chmod 600 it.", file=sys.stderr)
    raise SystemExit(2)


def frost_get(path: str, params: dict, cid: str) -> dict:
    import requests
    r = requests.get(f"{FROST}/{path}", params=params, auth=(cid, ""), timeout=90)
    if r.status_code == 401:
        raise SystemExit("Frost rejected the client ID (401). Check it is the ID, "
                         "not the secret.")
    if r.status_code == 404:
        return {"data": []}
    if not r.ok:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:300])
        except Exception:                            # noqa: BLE001
            msg = r.text[:300]
        raise SystemExit(f"Frost returned {r.status_code}: {msg}")
    return r.json()


def station_meta(ids: list[str], cid: str) -> dict:
    out = {}
    data = frost_get("sources/v0.jsonld", {"ids": ",".join(ids)}, cid).get("data", [])
    for s in data:
        geo = s.get("geometry") or {}
        coords = geo.get("coordinates")
        if not coords:
            continue
        out[s["id"]] = {"name": s.get("name", s["id"]),
                        "lon": float(coords[0]), "lat": float(coords[1]),
                        "masl": s.get("masl")}
    missing = [i for i in ids if i not in out]
    if missing:
        print(f"  warning: no coordinates for {', '.join(missing)} - skipping",
              file=sys.stderr)
    return out


def observations(sid: str, t0: dt.datetime, t1: dt.datetime, cid: str):
    ref = f"{t0:%Y-%m-%dT%H:%M:%S}Z/{t1:%Y-%m-%dT%H:%M:%S}Z"
    data = frost_get("observations/v0.jsonld",
                     {"sources": sid, "referencetime": ref,
                      "elements": "air_temperature"}, cid).get("data", [])
    times, vals = [], []
    for rec in data:
        for ob in rec.get("observations", []):
            if ob.get("elementId") != "air_temperature":
                continue
            lvl = ob.get("level") or {}
            if lvl and lvl.get("value") not in (2, None):
                continue
            times.append(dt.datetime.fromisoformat(
                rec["referenceTime"].replace("Z", "+00:00")).replace(tzinfo=None))
            vals.append(float(ob["value"]))
            break
    order = sorted(range(len(times)), key=lambda i: times[i])
    return [times[i] for i in order], [vals[i] for i in order]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, type=Path, help="nordic_*.nc")
    ap.add_argument("--stations", nargs="*", default=DEFAULT_STATIONS)
    ap.add_argument("-o", "--out", type=Path, default=Path("results/verify"))
    args = ap.parse_args()

    try:
        import numpy as np
        import xarray as xr
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc.name} missing - run inside the Bris environment.",
              file=sys.stderr)
        return 1

    cid = client_id()
    ds = xr.open_dataset(args.forecast)

    var = "air_temperature_2m"
    if var not in ds:
        print(f"ERROR: {var} not in {args.forecast.name}", file=sys.stderr)
        return 1

    lat2d = np.asarray(ds["latitude"].values, dtype="float64")
    lon2d = np.asarray(ds["longitude"].values, dtype="float64")
    times = [dt.datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(int))
             for t in ds["time"].values]
    t0, t1 = times[0], times[-1]

    print(f"forecast : {args.forecast.name}")
    print(f"           {len(times)} steps, {t0:%Y-%m-%d %HZ} .. {t1:%Y-%m-%d %HZ}")
    print(f"stations : {', '.join(args.stations)}\n")

    meta = station_meta(args.stations, cid)
    if not meta:
        print("No usable stations.", file=sys.stderr)
        return 2

    panels = []
    for sid, m in meta.items():
        # nearest grid point, great-circle
        dlat = np.deg2rad(lat2d - m["lat"])
        dlon = np.deg2rad(lon2d - m["lon"])
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.deg2rad(m["lat"])) * np.cos(np.deg2rad(lat2d)) *
             np.sin(dlon / 2) ** 2)
        dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        j, i = np.unravel_index(int(np.argmin(dist_km)), dist_km.shape)
        d = float(dist_km[j, i])

        fc = ds[var].isel({ds[var].dims[-2]: j, ds[var].dims[-1]: i}).squeeze()
        fc_c = np.asarray(fc.values, dtype="float64") - 273.15

        ot, ov = observations(sid, t0, t1, cid)
        print(f"  {m['name']:<28} {d:5.1f} km from grid point, "
              f"{len(ov):3d} observations")
        panels.append((m, fc_c, ot, ov, d))

    n = len(panels)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.6 * nrow),
                             squeeze=False, sharex=True)
    for ax, (m, fc_c, ot, ov, d) in zip(axes.ravel(), panels):
        if ov:
            ax.plot(ot, ov, "-", color="0.35", lw=1.4, label="observed (Frost)")
        ax.plot(times, fc_c, "o-", color="tab:blue", lw=2, ms=5, label="Bris")
        masl = f", {m['masl']:.0f} m" if m.get("masl") is not None else ""
        ax.set_title(f"{m['name']}{masl}   ({d:.1f} km to grid point)", fontsize=10)
        ax.set_ylabel("2 m temperature [°C]")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Bris vs observed 2 m temperature — initialised "
                 f"{t0:%Y-%m-%d %HZ}, +0h to +{(len(times) - 1) * 6}h", y=1.0)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / f"stations_{t0:%Y%m%dT%H}Z.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote {p}")
    print("\nRead the shape, not the offset. A steady gap at one site is the")
    print("station sitting somewhere a 6 km^2 cell mean cannot represent.")
    print("A diurnal swing at the wrong time, or a front arriving hours late,")
    print("is the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

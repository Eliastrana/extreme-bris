#!/usr/bin/env python3
"""Does the model shrink towards the mean? Measured over many stations.

    ~/bris-env/.venv/bin/python scripts/check_smoothing.py \
        --forecast ~/bris-runs/20250401T00Z-120h/nordic_*.nc \
        --max-stations 60 -o results/verify

Four station panels showed two things that looked like the same failure: night
minima several degrees too warm, and variability flattened at Tromsoe. Both are
shrinkage towards the mean, which is what a model trained on squared error
learns - it pays in the tails and the average barely notices.

Four sites is an anecdote. This asks the question properly.

THE METHOD. Everything is done in ANOMALIES: each station's own mean over the
period is subtracted from both series before anything is pooled. That removes
the constant offset a station has because it sits in a valley, or by a wall, or
at an altitude a 6 km^2 cell averages away - representativeness, which is not
model error and would otherwise swamp the signal. What survives is variability,
which is the thing in question.

THREE MEASURES, strongest first:

  1. BIAS AGAINST OBSERVED PERCENTILE. Bin the observed anomalies, and in each
     bin take the mean forecast error. A model that shrinks is warm where it is
     cold and cold where it is warm, so this comes out as a clear negative
     slope. A model that is merely wrong scatters without slope. This is the
     one plot that separates the two.

  2. REGRESSION SLOPE of forecast anomaly on observed anomaly. One number.
     1.0 is no shrinkage; below 1.0 is shrinkage, and the shortfall is roughly
     how much amplitude is lost.

  3. STANDARD DEVIATION RATIO per station. Cruder than the slope - it does not
     care whether the variability is in phase - but it is hard to argue with
     and it shows the spread across sites.

WHAT WOULD FALSIFY IT. A flat line in the first plot, a slope near 1.0, and a
ratio distribution centred on 1.0. That would mean the four panels were noise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bris_tls import ensure_ca_bundle, explain          # noqa: E402
from plot_stations import FROST, client_id, frost_get   # noqa: E402

BATCH = 40          # station ids per observations request


def norwegian_stations(cid: str) -> list[dict]:
    """Every Norwegian SensorSystem Frost will name, with coordinates."""
    out = []
    for params in ({"types": "SensorSystem", "country": "Norge"},
                   {"types": "SensorSystem", "countrycode": "NO"}):
        try:
            data = frost_get("sources/v0.jsonld", params, cid).get("data", [])
        except SystemExit:
            continue
        if data:
            print(f"  sources query {list(params)[1]}= returned {len(data)} stations")
            for s in data:
                geo = s.get("geometry") or {}
                c = geo.get("coordinates")
                if not c:
                    continue
                out.append({"id": s["id"], "name": s.get("name", s["id"]),
                            "lon": float(c[0]), "lat": float(c[1]),
                            "masl": s.get("masl")})
            break
    return out


def observations_batch(ids: list[str], t0: dt.datetime, t1: dt.datetime, cid: str):
    """{station_id: {datetime: value}} for a batch of stations in one request."""
    ref = f"{t0:%Y-%m-%dT%H:%M:%S}Z/{t1:%Y-%m-%dT%H:%M:%S}Z"
    data = frost_get("observations/v0.jsonld",
                     {"sources": ",".join(ids), "referencetime": ref,
                      "elements": "air_temperature"}, cid).get("data", [])
    out: dict = {}
    for rec in data:
        sid = rec.get("sourceId", "").split(":")[0]
        for ob in rec.get("observations", []):
            if ob.get("elementId") != "air_temperature":
                continue
            lvl = ob.get("level") or {}
            if lvl and lvl.get("value") not in (2, None):
                continue
            t = dt.datetime.fromisoformat(
                rec["referenceTime"].replace("Z", "+00:00")).replace(tzinfo=None)
            out.setdefault(sid, {})[t] = float(ob["value"])
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, type=Path, help="nordic_*.nc")
    ap.add_argument("--max-stations", type=int, default=60)
    ap.add_argument("--min-points", type=int, default=6,
                    help="drop stations with fewer matched times (default 6)")
    ap.add_argument("--max-dist-km", type=float, default=5.0,
                    help="drop stations further than this from a grid point")
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

    ensure_ca_bundle()
    cid = client_id()
    ds = xr.open_dataset(args.forecast)
    var = "air_temperature_2m"
    if var not in ds:
        print(f"ERROR: {var} not in {args.forecast.name}", file=sys.stderr)
        return 1

    lat2d = np.asarray(ds["latitude"].values, dtype="float64")
    lon2d = np.asarray(ds["longitude"].values, dtype="float64")
    times = [np.datetime64(t, "s").astype(object) for t in ds["time"].values]
    t0, t1 = times[0], times[-1]
    lead_h = int((t1 - t0).total_seconds() // 3600)

    print(f"forecast : {args.forecast.name}")
    print(f"           {len(times)} steps, +0h .. +{lead_h}h\n")

    print("finding stations...")
    stations = norwegian_stations(cid)
    if not stations:
        print("Frost returned no stations. Check the sources query.", file=sys.stderr)
        return 2

    # inside the forecast domain, then spread by latitude rather than taking
    # whatever the API happened to list first
    lo, hi = lat2d.min(), lat2d.max()
    wlo, whi = lon2d.min(), lon2d.max()
    inside = [s for s in stations
              if lo < s["lat"] < hi and wlo < s["lon"] < whi]
    inside.sort(key=lambda s: s["lat"])
    if len(inside) > args.max_stations:
        step = len(inside) / args.max_stations
        inside = [inside[int(i * step)] for i in range(args.max_stations)]
    print(f"  {len(inside)} inside the domain, spread by latitude\n")

    print("fetching observations...")
    obs: dict = {}
    for k in range(0, len(inside), BATCH):
        batch = [s["id"] for s in inside[k:k + BATCH]]
        obs.update(observations_batch(batch, t0, t1, cid))
        print(f"  {min(k + BATCH, len(inside)):3d}/{len(inside)} stations")

    # --- pair forecast and observation, in anomalies -------------------------
    fa_all, oa_all, ratios, names = [], [], [], []
    for s in inside:
        series = obs.get(s["id"])
        if not series:
            continue
        dlat = np.deg2rad(lat2d - s["lat"])
        dlon = np.deg2rad(lon2d - s["lon"])
        a = (np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(s["lat"])) *
             np.cos(np.deg2rad(lat2d)) * np.sin(dlon / 2) ** 2)
        dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        j, i = np.unravel_index(int(np.argmin(dist)), dist.shape)
        if float(dist[j, i]) > args.max_dist_km:
            continue

        fc = np.asarray(ds[var].isel({ds[var].dims[-2]: j,
                                      ds[var].dims[-1]: i}).squeeze().values,
                        dtype="float64") - 273.15
        pairs = [(fc[n], series[t]) for n, t in enumerate(times) if t in series]
        if len(pairs) < args.min_points:
            continue
        f = np.array([p[0] for p in pairs])
        o = np.array([p[1] for p in pairs])
        if o.std() < 0.5:                 # nothing to shrink
            continue
        fa_all.append(f - f.mean())
        oa_all.append(o - o.mean())
        ratios.append(f.std() / o.std())
        names.append(s["name"])

    if len(ratios) < 5:
        print(f"Only {len(ratios)} usable stations - too few to conclude.",
              file=sys.stderr)
        return 3

    fa = np.concatenate(fa_all)
    oa = np.concatenate(oa_all)
    ratios = np.array(ratios)
    slope = float(np.polyfit(oa, fa, 1)[0])

    print(f"\n{len(ratios)} usable stations, {len(oa)} paired values\n")
    print(f"  regression slope (forecast on observed) : {slope:.3f}")
    print(f"  std ratio, median                       : {np.median(ratios):.3f}")
    print(f"  std ratio, stations below 1.0           : "
          f"{int((ratios < 1).sum())} of {len(ratios)}")

    # --- bias against observed percentile ------------------------------------
    edges = np.percentile(oa, [0, 5, 15, 30, 50, 70, 85, 95, 100])
    centres, biases, counts = [], [], []
    for k in range(len(edges) - 1):
        m = (oa >= edges[k]) & (oa <= edges[k + 1] if k == len(edges) - 2
                                else oa < edges[k + 1])
        if m.sum() < 10:
            continue
        centres.append(float(np.median(oa[m])))
        biases.append(float((fa[m] - oa[m]).mean()))
        counts.append(int(m.sum()))

    print("\n  observed anomaly -> mean forecast error")
    for c, b, n in zip(centres, biases, counts):
        print(f"    {c:+6.2f} K   {b:+6.2f} K   (n={n})")

    cold = float((fa[oa <= edges[1]] - oa[oa <= edges[1]]).mean())
    warm = float((fa[oa >= edges[-2]] - oa[oa >= edges[-2]]).mean())
    print(f"\n  coldest 5%: {cold:+.2f} K    warmest 5%: {warm:+.2f} K")

    shrinking = slope < 0.95 and cold > 0.2 and warm < -0.2
    print("\nVERDICT: " + (
        "the model shrinks towards the mean - warm at the cold tail, cold at "
        "the warm\n         tail, and amplitude short by "
        f"{(1 - slope) * 100:.0f}%."
        if shrinking else
        "no consistent shrinkage. The four-panel reading does not\n"
        "         generalise; treat it as noise."))

    # --- plots ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    ax[0].axhline(0, color="0.6", lw=1)
    ax[0].plot(centres, biases, "o-", color="tab:red", lw=2, ms=7)
    ax[0].set_xlabel("observed anomaly [K]")
    ax[0].set_ylabel("mean forecast error [K]")
    ax[0].set_title("Error against observed percentile\n"
                    "downward slope = shrinkage", fontsize=10)
    # Floor the y-range. Left to autoscale, a null result fills the panel with
    # a dramatic zigzag spanning 0.03 K - noise drawn as if it were signal,
    # which is the opposite of what this plot is for.
    span = max(0.5, 1.25 * max(abs(b) for b in biases))
    ax[0].set_ylim(-span, span)
    ax[0].grid(alpha=.3)

    lim = float(np.percentile(np.abs(oa), 99))
    ax[1].plot(oa, fa, ".", ms=2, alpha=.25, color="tab:blue")
    ax[1].plot([-lim, lim], [-lim, lim], "-", color="0.4", lw=1.2, label="1:1")
    ax[1].plot([-lim, lim], [-lim * slope, lim * slope], "--", color="tab:red",
               lw=1.8, label=f"fit, slope {slope:.2f}")
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel("observed anomaly [K]")
    ax[1].set_ylabel("forecast anomaly [K]")
    ax[1].set_title("Amplitude", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].hist(ratios, bins=18, color="tab:blue", alpha=.75)
    ax[2].axvline(1.0, color="0.35", ls="--", lw=1.5, label="no shrinkage")
    ax[2].axvline(float(np.median(ratios)), color="tab:red", lw=1.8,
                  label=f"median {np.median(ratios):.2f}")
    ax[2].set_xlabel("std(forecast) / std(observed)")
    ax[2].set_ylabel("stations")
    ax[2].set_title(f"Variability, {len(ratios)} stations", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.suptitle(f"Shrinkage towards the mean — {len(ratios)} stations, "
                 f"+0h to +{lead_h}h, initialised {t0:%Y-%m-%d %HZ}", y=1.03)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / f"smoothing_{t0:%Y%m%dT%H}Z_{lead_h}h.png"
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

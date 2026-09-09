#!/usr/bin/env python3
"""Score the forecast at station sites, broken down by hour of day.

    ~/bris-env/.venv/bin/python scripts/check_stations.py \
        --forecast ~/bris-runs/20250401T00Z-120h/nordic_*.nc \
        -o results/verify

Two things this answers that nothing else has.

HOUR OF DAY. The shrinkage test pooled every timestep and found nothing. But
the observation that started it was about NIGHT - minima several degrees too
warm at Blindern. A warm bias confined to 00 and 06 disappears in a mean over
the whole day, especially when the model has more variance than the stations
overall. Pooling answered a different question than the one asked. The forecast
steps 6-hourly from 00Z, so the valid times fall in exactly four bins, each
sampled about five times over 120 hours.

PRECIPITATION AND WIND. Neither has ever been compared with anything.
Precipitation is the variable the thesis is about - extremes are a tail in
precipitation before they are a tail in anything else - and it has sat in the
output files unexamined.

ON SCORING PRECIPITATION. Bias and RMSE are weak for a quantity that is zero
most of the time and enormous occasionally; the mean is dominated by the many
zeros. Hit rate and false alarm rate at a threshold say more about whether the
model puts rain in the right place, so both are reported and the contingency
table is the one to read.

ACCUMULATION. The forecast's precipitation field may be accumulated per step or
since initialisation - the two look identical at a glance and give answers that
differ by a factor of the lead time. This detects which by testing whether the
series is monotonic in time, and says which it concluded rather than assuming.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bris_tls import ensure_ca_bundle                    # noqa: E402
from plot_stations import client_id, frost_get           # noqa: E402
from check_smoothing import BATCH, norwegian_stations    # noqa: E402

# forecast variable -> (frost element, unit, label, conversion)
ELEMENTS = {
    "temperature": ("air_temperature_2m", "air_temperature", "°C",
                    "2 m temperatur", lambda a: a - 273.15),
    "wind": ("wind_speed_10m", "wind_speed", "m/s",
             "10 m vindstyrke", lambda a: a),
    "precipitation": ("precipitation_amount", "sum(precipitation_amount PT1H)",
                      "mm", "Nedbør", lambda a: a),
}

# mm in 6 h. Low enough that April in Norway clears it often, high enough that
# it means something happened.
WET_THRESHOLD = 0.5


def observations_batch(ids, element, t0, t1, cid):
    """{station: {datetime: value}} for one element across a batch of stations."""
    ref = f"{t0:%Y-%m-%dT%H:%M:%S}Z/{t1:%Y-%m-%dT%H:%M:%S}Z"
    data = frost_get("observations/v0.jsonld",
                     {"sources": ",".join(ids), "referencetime": ref,
                      "elements": element}, cid).get("data", [])
    out: dict = {}
    for rec in data:
        sid = rec.get("sourceId", "").split(":")[0]
        for ob in rec.get("observations", []):
            if ob.get("elementId") != element:
                continue
            lvl = ob.get("level") or {}
            if element == "air_temperature" and lvl and lvl.get("value") not in (2, None):
                continue
            t = dt.datetime.fromisoformat(
                rec["referenceTime"].replace("Z", "+00:00")).replace(tzinfo=None)
            out.setdefault(sid, {})[t] = float(ob["value"])
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, type=Path)
    ap.add_argument("--elements", default="temperature,wind,precipitation")
    ap.add_argument("--max-stations", type=int, default=60)
    ap.add_argument("--max-dist-km", type=float, default=5.0)
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

    lat2d = np.asarray(ds["latitude"].values, dtype="float64")
    lon2d = np.asarray(ds["longitude"].values, dtype="float64")
    times = [np.datetime64(t, "s").astype(object) for t in ds["time"].values]
    step_h = int((times[1] - times[0]).total_seconds() // 3600) if len(times) > 1 else 6

    print(f"forecast : {args.forecast.name}")
    print(f"           {len(times)} steps, {step_h}h apart, "
          f"+0h .. +{(len(times)-1)*step_h}h\n")

    stations = norwegian_stations(cid)
    lo, hi = lat2d.min(), lat2d.max()
    wlo, whi = lon2d.min(), lon2d.max()
    inside = sorted([s for s in stations if lo < s["lat"] < hi and wlo < s["lon"] < whi],
                    key=lambda s: s["lat"])
    if len(inside) > args.max_stations:
        stp = len(inside) / args.max_stations
        inside = [inside[int(i * stp)] for i in range(args.max_stations)]
    print(f"  {len(inside)} stations inside the domain\n")

    # nearest grid point per station, once
    cells = {}
    for s in inside:
        dlat = np.deg2rad(lat2d - s["lat"]); dlon = np.deg2rad(lon2d - s["lon"])
        a = (np.sin(dlat/2)**2 + np.cos(np.deg2rad(s["lat"])) *
             np.cos(np.deg2rad(lat2d)) * np.sin(dlon/2)**2)
        d = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        j, i = np.unravel_index(int(np.argmin(d)), d.shape)
        if float(d[j, i]) <= args.max_dist_km:
            cells[s["id"]] = (j, i)
    print(f"  {len(cells)} within {args.max_dist_km} km of a grid point\n")

    results = {}
    for key in [e.strip() for e in args.elements.split(",")]:
        if key not in ELEMENTS:
            print(f"  unknown element {key}, skipping", file=sys.stderr)
            continue
        var, frost_el, unit, label, convert = ELEMENTS[key]
        if var not in ds:
            print(f"  {label}: {var} not in the file, skipping")
            continue

        print(f"=== {label}  [{unit}]")
        accumulated = key == "precipitation"

        # --- forecast series per station -------------------------------------
        fc = {}
        for sid, (j, i) in cells.items():
            v = convert(np.asarray(
                ds[var].isel({ds[var].dims[-2]: j, ds[var].dims[-1]: i}).squeeze().values,
                dtype="float64"))
            fc[sid] = v

        if accumulated:
            # Cumulative or per step? Monotonic in time means cumulative.
            probe = np.array([v for v in fc.values()])
            finite = np.isfinite(probe)
            rising = np.all(np.diff(np.where(finite, probe, 0), axis=1) >= -1e-6, axis=1)
            cumulative = bool(rising.mean() > 0.9)
            print(f"  accumulation: {'since initialisation' if cumulative else 'per step'}"
                  f"  ({rising.mean()*100:.0f}% of stations monotonic)")
            if cumulative:
                for sid in fc:
                    fc[sid] = np.diff(fc[sid], prepend=np.nan)

        # --- observations ----------------------------------------------------
        obs = {}
        ids = list(cells)
        for k in range(0, len(ids), BATCH):
            obs.update(observations_batch(ids[k:k+BATCH], frost_el,
                                          times[0] - dt.timedelta(hours=step_h),
                                          times[-1], cid))

        # --- pair -------------------------------------------------------------
        rows = []
        for sid, series in obs.items():
            if sid not in fc:
                continue
            for n, t in enumerate(times):
                if accumulated:
                    if n == 0:
                        continue    # undefined at lead time 0
                    hrs = [t - dt.timedelta(hours=h) for h in range(step_h)]
                    vals = [series.get(x) for x in hrs]
                    if any(v is None for v in vals):
                        continue
                    o = float(sum(vals))
                else:
                    if t not in series:
                        continue
                    o = series[t]
                f = fc[sid][n]
                if not np.isfinite(f):
                    continue
                rows.append((t.hour, n * step_h, f, o))

        if len(rows) < 50:
            print(f"  only {len(rows)} pairs - too few to report\n")
            continue

        hours = np.array([r[0] for r in rows])
        leads = np.array([r[1] for r in rows])
        f = np.array([r[2] for r in rows])
        o = np.array([r[3] for r in rows])
        print(f"  {len(rows):,} pairs from {len(set(obs) & set(fc))} stations")
        print(f"  bias {np.mean(f-o):+.2f}   MAE {np.mean(np.abs(f-o)):.2f}   "
              f"RMSE {np.sqrt(np.mean((f-o)**2)):.2f}  {unit}")

        # --- by hour of day ---------------------------------------------------
        print(f"\n  {'hour':>5} {'n':>6} {'bias':>8} {'MAE':>8}")
        by_hour = {}
        for h in sorted(set(hours)):
            m = hours == h
            by_hour[int(h)] = (float(np.mean(f[m]-o[m])), float(np.mean(np.abs(f[m]-o[m]))),
                               int(m.sum()))
            print(f"  {int(h):02d}Z {m.sum():>8,} {by_hour[int(h)][0]:>+8.2f} "
                  f"{by_hour[int(h)][1]:>8.2f}")

        if accumulated:
            # Contingency at a wet threshold: does it put rain in the right place?
            fw, ow = f >= WET_THRESHOLD, o >= WET_THRESHOLD
            hit = int((fw & ow).sum()); miss = int((~fw & ow).sum())
            fa = int((fw & ~ow).sum()); cn = int((~fw & ~ow).sum())
            pod = hit / max(hit + miss, 1)
            far = fa / max(hit + fa, 1)
            print(f"\n  wet/dry at {WET_THRESHOLD} mm per {step_h}h")
            print(f"    hit {hit}  miss {miss}  false alarm {fa}  correct dry {cn}")
            print(f"    POD {pod:.2f}   FAR {far:.2f}   "
                  f"(POD is how much observed rain it caught, FAR how much it")
            print(f"     forecast that did not happen)")

        results[key] = dict(label=label, unit=unit, by_hour=by_hour,
                            leads=leads, err=f-o)
        print()

    if not results:
        print("Nothing scored.", file=sys.stderr)
        return 2

    # --- plots ----------------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5.2*n, 7.6), squeeze=False)
    for col, (_, r) in enumerate(results.items()):
        hrs = sorted(r["by_hour"])
        ax = axes[0][col]
        ax.axhline(0, color="0.6", lw=1)
        ax.plot(hrs, [r["by_hour"][h][0] for h in hrs], "o-", color="tab:red", lw=2, ms=7)
        ax.set_xticks(hrs); ax.set_xticklabels([f"{h:02d}Z" for h in hrs])
        ax.set_ylabel(f"bias [{r['unit']}]")
        ax.set_title(f"{r['label']}\nbias by hour of day", fontsize=10)
        ax.grid(alpha=.3)
        span = max(0.3, 1.3*max(abs(v[0]) for v in r["by_hour"].values()))
        ax.set_ylim(-span, span)

        ax = axes[1][col]
        ax.axhline(0, color="0.6", lw=1)
        ls = sorted(set(r["leads"]))
        vals = [float(np.mean(r["err"][r["leads"] == l])) for l in ls]
        ax.plot(ls, vals, "o-", color="tab:blue", lw=2, ms=5)
        ax.set_xlabel("lead time [h]"); ax.set_ylabel(f"bias [{r['unit']}]")
        ax.set_title("bias by lead time", fontsize=10)
        ax.grid(alpha=.3)
        # Floor the range, or a bias that is genuinely zero autoscales to 1e-16
        # and draws floating point noise as though it were a signal.
        ax.set_ylim(*(lambda m: (-m, m))(max(0.3, 1.3 * max(abs(v) for v in vals))))

    fig.suptitle(f"Bris against station observations — {times[0]:%Y-%m-%d %HZ}, "
                 f"+0 to +{(len(times)-1)*step_h}h", y=1.01)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / f"stations_by_hour_{times[0]:%Y%m%dT%H}Z.png"
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")
    print("\nRead the top row first. A warm bias at 00Z and 06Z that is absent at")
    print("12Z is the nocturnal cooling the pooled test could not see.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

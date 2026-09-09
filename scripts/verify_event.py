#!/usr/bin/env python3
"""Score a forecast's 24 h precipitation against every gauge that reported.

    ~/bris-env/.venv/bin/python scripts/verify_event.py \
        --forecast ~/bris-runs/20251004T00Z-od-120h/nordic_*.nc \
        --date 2025-10-04

TWO THINGS THIS GETS RIGHT THAT THE OBVIOUS VERSION DOES NOT.

THE WINDOW. Frost's `sum(precipitation_amount P1D)` is a 06-06 UTC total, not
00-00 - the element's own description says so. A forecast initialised at 00Z
therefore has to be summed from +12 h to +30 h, not +6 h to +24 h. Getting
this wrong shifts the comparison by six hours and shows up as the model
missing rain it actually produced, which is indistinguishable from a real dry
bias if you never check.

SELECTION. Scoring only the stations that exceeded their own high quantile
measures the wrong thing. Those stations were CHOSEN for having observed an
unusually high value, so even a perfect forecast scores as an underestimate
there - it is regression to the mean, not model error. The honest number is
across every gauge that reported; the extreme subset is reported separately
and read as what it is, a conditioned sample.

Nearest grid point, no interpolation: a 2.5 km cell against a gauge is already
a harsh comparison and smoothing it would flatter the model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bris_tls import ensure_ca_bundle                    # noqa: E402
from plot_stations import client_id, frost_get           # noqa: E402

ELEMENT = "sum(precipitation_amount P1D)"
BATCH = 50


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, type=Path)
    ap.add_argument("--date", required=True, help="event day, YYYY-MM-DD")
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--max-dist-km", type=float, default=10.0)
    args = ap.parse_args()

    import numpy as np
    import xarray as xr

    ensure_ca_bundle()
    cid = client_id()
    day = dt.date.fromisoformat(args.date)

    ds = xr.open_dataset(args.forecast)
    lat = np.asarray(ds["latitude"].values, dtype="float64")
    lon = np.asarray(ds["longitude"].values, dtype="float64")
    times = [np.datetime64(t, "h").astype(object) for t in ds["time"].values]
    t0 = times[0]

    # The 06-06 window, located by valid time rather than by assuming an
    # index. Each step carries the accumulation over the 6 h ENDING at its
    # valid time, so the window is the four steps valid 12Z, 18Z, 00Z, 06Z.
    # First block of the 06-06 day is 06-12Z, which ENDS at 12Z - so the window
    # starts at the step valid 12Z, not the one valid 06Z. The step valid 06Z
    # carries 00-06Z, which belongs to the previous observation day.
    want_end = [dt.datetime.combine(day, dt.time(12)) + dt.timedelta(hours=6 * k)
                for k in range(4)]
    idx = []
    for w in want_end:
        m = [i for i, t in enumerate(times) if t == w]
        if not m:
            print(f"ERROR: forecast has no step valid {w:%Y-%m-%d %H}Z. "
                  f"It runs {times[0]:%Y-%m-%d %H}Z .. {times[-1]:%Y-%m-%d %H}Z.",
                  file=sys.stderr)
            return 1
        idx.append(m[0])
    print(f"t0            : {t0:%Y-%m-%d %H}Z")
    print(f"window        : {args.date} 06Z .. {day + dt.timedelta(days=1)} 06Z "
          f"(steps {idx}, valid {[f'{times[i]:%d %HZ}' for i in idx]})")

    fc = ds["precipitation_amount"].isel(time=idx).sum(dim="time").values.squeeze()

    # --- observations: every gauge in the domain that reported that day ------
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    raw = frost_get("sources/v0.jsonld",
                    {"types": "SensorSystem", "country": "Norge"}, cid).get("data", [])
    sites = {}
    for s in raw:
        c = (s.get("geometry") or {}).get("coordinates")
        if not c:
            continue
        lo, la = float(c[0]), float(c[1])
        if south < la < north and west < lo < east:
            sites[s["id"]] = (s.get("name", s["id"]), la, lo)

    ref = f"{day}/{day + dt.timedelta(days=1)}"
    obs = {}
    ids = list(sites)
    for k in range(0, len(ids), BATCH):
        chunk = ids[k:k + BATCH]
        data = frost_get("observations/v0.jsonld",
                         {"sources": ",".join(chunk), "referencetime": ref,
                          "elements": ELEMENT}, cid).get("data", [])
        for rec in data:
            sid = rec.get("sourceId", "").split(":")[0]
            if not rec["referenceTime"].startswith(str(day)):
                continue
            for o in rec.get("observations", []):
                if o.get("elementId") == ELEMENT:
                    try:
                        obs[sid] = float(o["value"])
                    except (TypeError, ValueError):
                        pass
                    break

    print(f"gauges        : {len(obs)} reported on {day}\n")

    # --- pair each gauge with its nearest grid point --------------------------
    pairs = []
    for sid, mm in obs.items():
        name, la, lo = sites[sid]
        d2 = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
        j, i = np.unravel_index(np.argmin(d2), lat.shape)
        km = np.sqrt(d2[j, i]) * 111.0
        if km > args.max_dist_km:
            continue
        f = float(fc[j, i])
        if not np.isfinite(f):
            continue
        pairs.append((name, mm, f))

    if not pairs:
        print("No gauge paired with a finite grid point.", file=sys.stderr)
        return 2

    o = np.array([p[1] for p in pairs])
    f = np.array([p[2] for p in pairs])

    def block(title, mask):
        if mask.sum() < 2:
            print(f"{title}: too few ({mask.sum()})")
            return
        oo, ff = o[mask], f[mask]
        print(f"{title}  (n={mask.sum()})")
        print(f"  observed mean {oo.mean():6.1f} mm   forecast mean {ff.mean():6.1f} mm"
              f"   bias {ff.mean() - oo.mean():+6.1f}")
        print(f"  ratio {ff.mean() / max(oo.mean(), 1e-9):.2f}"
              f"   corr {np.corrcoef(oo, ff)[0, 1]:+.3f}"
              f"   RMSE {np.sqrt(((ff - oo) ** 2).mean()):6.1f}")

    print("=" * 62)
    block("ALL GAUGES", np.ones(len(o), dtype=bool))
    print()
    block("WET ONLY (obs >= 1 mm)", o >= 1.0)
    print()
    # The conditioned sample: chosen for having observed a lot. Reported so the
    # selection effect is visible next to the unconditioned number, not instead
    # of it.
    thr = np.quantile(o, args.quantile) if len(o) > 20 else o.max()
    block(f"EXTREME TAIL (obs >= {thr:.0f} mm, selected on the observation)", o >= thr)

    print("\nworst individual misses:")
    for name, mm, ff in sorted(pairs, key=lambda p: p[2] - p[1])[:6]:
        print(f"  {name[:30]:30s} obs {mm:6.1f}  fcst {ff:6.1f}  {ff - mm:+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

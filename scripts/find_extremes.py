#!/usr/bin/env python3
"""Find candidate extreme-precipitation events from station observations alone.

    ~/bris-env/.venv/bin/python scripts/find_extremes.py --years 10 \
        -o results/extremes

Case selection is the one piece of this thesis that needs no model, no MARS and
no MEPS archive. Frost has the observations; the events either happened or they
did not. Doing it now means the date list is ready the moment the archive comes
back, instead of starting then.

WHAT COUNTS AS EXTREME. Not a fixed millimetre threshold: 40 mm in a day is an
ordinary autumn in Bergen and a once-a-decade event in Oslo, so a fixed cut
would select the west coast and call it a result. Each station is compared with
its OWN wet-day distribution, and an event day is one where many stations
exceed their own high quantile at once.

TWO KINDS OF EVENT, reported separately because they are different weather and
a model can be good at one and hopeless at the other:

  WIDESPREAD - many stations over threshold on the same day. Synoptic, usually
  an atmospheric river or a slow front. This is what a stretched-grid model
  should get right, and the natural first target.

  LOCAL - one or two stations far past their own threshold while neighbours
  stay dry. Convective, small, and the case where a 2.5 km grid either resolves
  it or does not.

The ranking is by exceedance ratio rather than raw millimetres, so a station
that beat its own 99th percentile by three times counts more than one that
scraped past it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bris_tls import ensure_ca_bundle                    # noqa: E402
from plot_stations import client_id, frost_get           # noqa: E402

ELEMENT = "sum(precipitation_amount P1D)"
BATCH = 25          # stations per request; a decade of days is a lot of rows
WET_DAY = 1.0       # mm, the conventional definition of a wet day

# The LAM domain, so events outside it are not offered as candidates.
DOMAIN = dict(south=51.0, north=74.5, west=-13.0, east=49.0)


def fetch_daily(ids, t0, t1, cid, cache: Path):
    """Daily sums per station, cached per batch and year so a rerun is free."""
    key = cache / f"{t0:%Y%m%d}_{t1:%Y%m%d}_{hash(tuple(ids)) & 0xffffff:06x}.json"
    if key.exists():
        return json.loads(key.read_text())
    ref = f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}"
    data = frost_get("observations/v0.jsonld",
                     {"sources": ",".join(ids), "referencetime": ref,
                      "elements": ELEMENT}, cid).get("data", [])
    out: dict = {}
    for rec in data:
        sid = rec.get("sourceId", "").split(":")[0]
        for ob in rec.get("observations", []):
            if ob.get("elementId") != ELEMENT:
                continue
            day = rec["referenceTime"][:10]
            try:
                out.setdefault(sid, {})[day] = float(ob["value"])
            except (TypeError, ValueError):
                pass
            break
    cache.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--max-stations", type=int, default=60)
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="wet-day quantile that defines a station's threshold")
    ap.add_argument("--min-stations", type=int, default=3,
                    help="stations over threshold for a day to count as widespread")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("-o", "--out", type=Path, default=Path("results/extremes"))
    args = ap.parse_args()

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc.name} missing - run inside the Bris environment.",
              file=sys.stderr)
        return 1

    ensure_ca_bundle()
    cid = client_id()
    today = dt.date.today()
    start = dt.date(today.year - args.years, today.month, 1)

    print(f"window   : {start} .. {today}  ({args.years} years)")
    print(f"threshold: each station's own {args.quantile:.0%} wet-day quantile\n")

    # Record length first, geography second. Spreading by latitude and then
    # discarding short records threw away more than half the sample: 25 of 60
    # survived, which made "4 stations at once" a weak claim about a widespread
    # event. Frost reports validFrom, so stations that cannot cover the window
    # are dropped before they cost an API call.
    raw = frost_get("sources/v0.jsonld",
                    {"types": "SensorSystem", "country": "Norge"}, cid).get("data", [])
    stations = []
    for src in raw:
        geo = src.get("geometry") or {}
        c = geo.get("coordinates")
        if not c:
            continue
        lat_, lon_ = float(c[1]), float(c[0])
        if not (DOMAIN["south"] < lat_ < DOMAIN["north"]
                and DOMAIN["west"] < lon_ < DOMAIN["east"]):
            continue
        vf = (src.get("validFrom") or "")[:10]
        if not vf or vf > str(start):
            continue                      # opened after the window began
        vt = (src.get("validTo") or "")[:10]
        if vt and vt < str(today - dt.timedelta(days=90)):
            continue                      # closed before the window ended
        stations.append({"id": src["id"], "name": src.get("name", src["id"]),
                         "lat": lat_, "lon": lon_, "masl": src.get("masl")})

    print(f"  {len(stations)} stations cover the whole window")
    stations.sort(key=lambda s: s["lat"])
    if len(stations) > args.max_stations:
        step = len(stations) / args.max_stations
        stations = [stations[int(i * step)] for i in range(args.max_stations)]
    print(f"  {len(stations)} kept, spread by latitude\n")

    cache = args.out / "cache"
    series: dict = {}
    print("fetching daily sums (cached; a rerun costs nothing)")
    for k in range(0, len(stations), BATCH):
        ids = [s["id"] for s in stations[k:k + BATCH]]
        for yr in range(start.year, today.year + 1):
            a = max(start, dt.date(yr, 1, 1))
            b = min(today, dt.date(yr, 12, 31))
            if a > b:
                continue
            got = fetch_daily(ids, a, b, cid, cache)
            for sid, days in got.items():
                series.setdefault(sid, {}).update(days)
        print(f"  {min(k + BATCH, len(stations)):3d}/{len(stations)} stations")

    # --- per-station threshold from its own wet days -------------------------
    thresh, nwet = {}, {}
    for sid, days in series.items():
        wet = np.array([v for v in days.values() if v >= WET_DAY])
        if wet.size < 100:            # too short a record to define a tail
            continue
        thresh[sid] = float(np.quantile(wet, args.quantile))
        nwet[sid] = int(wet.size)
    print(f"\n  {len(thresh)} of {len(stations)} stations have enough wet days "
          f"to define a threshold")
    if not thresh:
        print("Nothing to rank.", file=sys.stderr)
        return 2
    tv = np.array(list(thresh.values()))
    print(f"  thresholds span {tv.min():.0f} to {tv.max():.0f} mm/day "
          f"(median {np.median(tv):.0f}) - which is why a fixed cut would not do\n")

    # --- exceedances by day ---------------------------------------------------
    byday: dict = {}
    for sid, days in series.items():
        if sid not in thresh:
            continue
        for day, v in days.items():
            if v >= thresh[sid]:
                byday.setdefault(day, []).append((sid, v, v / thresh[sid]))

    widespread = sorted(
        ((d, hits) for d, hits in byday.items() if len(hits) >= args.min_stations),
        key=lambda kv: (len(kv[1]), max(h[2] for h in kv[1])), reverse=True)
    local = sorted(
        ((d, hits) for d, hits in byday.items() if len(hits) < args.min_stations),
        key=lambda kv: max(h[2] for h in kv[1]), reverse=True)

    name = {s["id"]: s["name"] for s in stations}
    # MEPS keeps the 00/06/12/18 cycles past 16 months; the combined
    # meps_det_2_5km files are the ones dropped after 30 days. So this marks
    # what the docs SAY is recoverable - to be confirmed when the archive is up.
    meps_from = dt.date(2023, 10, 1)   # the 2023 archive regime this recipe targets

    print(f"=== WIDESPREAD: {args.min_stations}+ stations over their own threshold")
    print(f"  {'date':<12} {'n':>3} {'max ratio':>10}  worst station")
    for d, hits in widespread[:args.top]:
        w = max(hits, key=lambda h: h[2])
        flag = "" if dt.date.fromisoformat(d) >= meps_from else "  (pre-2023 regime)"
        print(f"  {d:<12} {len(hits):>3} {w[2]:>9.1f}x  "
              f"{name.get(w[0], w[0])[:26]:<26} {w[1]:.0f} mm{flag}")

    print(f"\n=== LOCAL: a single station far past its own threshold")
    print(f"  {'date':<12} {'ratio':>7}  station")
    for d, hits in local[:12]:
        w = max(hits, key=lambda h: h[2])
        print(f"  {d:<12} {w[2]:>6.1f}x  {name.get(w[0], w[0])[:30]:<30} {w[1]:.0f} mm")

    # --- output ---------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": [str(start), str(today)],
        "quantile": args.quantile,
        "stations": len(thresh),
        "widespread": [
            {"date": d, "stations": len(h),
             "max_ratio": round(max(x[2] for x in h), 2),
             "sites": [{"id": s, "name": name.get(s, s), "mm": round(v, 1),
                        "ratio": round(r, 2)}
                       for s, v, r in sorted(h, key=lambda x: -x[2])[:8]]}
            for d, h in widespread[:args.top]],
        "local": [
            {"date": d, "max_ratio": round(max(x[2] for x in h), 2),
             "sites": [{"id": s, "name": name.get(s, s), "mm": round(v, 1),
                        "ratio": round(r, 2)} for s, v, r in h]}
            for d, h in local[:args.top]],
    }
    (args.out / "candidates.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(12, 4))
    days = [dt.date.fromisoformat(d) for d, _ in byday.items()]
    counts = [len(h) for _, h in byday.items()]
    ax.scatter(days, counts, s=14, alpha=.55, color="tab:blue")
    ax.axhline(args.min_stations - 0.5, color="tab:red", ls="--", lw=1.2,
               label=f"{args.min_stations}+ stations")
    ax.set_ylabel("stations over own threshold")
    ax.set_title(f"Days exceeding the station's own {args.quantile:.0%} wet-day "
                 f"quantile — {len(thresh)} stations")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    p = args.out / "exceedance_timeline.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote {args.out}/candidates.json")
    print(f"wrote {p}")
    print("\nThe widespread list is the place to start: synoptic events are what a")
    print("stretched-grid model is built for, and they give many stations to")
    print("verify against on the same day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

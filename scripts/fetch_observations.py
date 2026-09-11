#!/usr/bin/env python3
"""Cache the gauge record for the whole test window, once, before it is needed.

    scripts/fetch_observations.py                       # the test window
    scripts/fetch_observations.py --elements precipitation,temperature,wind

WHY A SEPARATE FETCH. Every verification script here takes a forecast file and
queries Frost for the hours that forecast covers. That was right when the
question was one event on one day. It is wrong for what comes next: three arms,
scored over a year of daily forecasts, against a control. The same station
hours would be requested hundreds of times, the scoring could not start until
forecasts existed, and a network hiccup partway through an evaluation would
look like a change in skill.

So the observations are fetched once, into a file, and everything downstream
reads that. It also means this can be done now, while the GPU queue is a week
deep, rather than on the critical path afterwards.

WHAT IT STORES. Hourly values, not six-hourly sums. Frost serves precipitation
as hourly totals and the model accumulates over six hours, so a sum has to
happen somewhere; doing it here would bake in one choice of window and one
choice of which hour a day starts at. The station work uses 06 to 06 and the
extreme ranking uses the same, but a scorer should be able to ask a different
question without re-downloading a year.

Missing is stored as NaN and means exactly that: the station reported nothing
for that hour. A gauge that reports nothing all year is dropped at the end
rather than kept as a column of NaN.

RESUMING IS FREE. Each request's answer is cached under the output directory
before the next one is made, so an interrupted run re-requests nothing. Frost
is somebody else's service and this asks it for a year of data across several
hundred stations; doing that twice because a laptop slept is rude.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("numpy", "requests")

import numpy as np  # noqa: E402

# This is long enough to be worth running detached, and detached means stdout is
# a file, which Python block-buffers. Progress that appears an hour late is
# progress nobody can act on, and the obvious workaround is an environment
# variable the person running it has to remember.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

from bris_tls import ensure_ca_bundle                    # noqa: E402
from plot_stations import client_id, frost_get           # noqa: E402
from check_smoothing import BATCH, norwegian_stations    # noqa: E402

# The forecast-side name, the Frost element id, and the unit it arrives in.
ELEMENTS = {
    "precipitation": ("sum(precipitation_amount PT1H)", "mm"),
    "temperature": ("air_temperature", "degC"),
    "wind": ("wind_speed", "m/s"),
}

# The window nothing has seen: after the checkpoint's validation period and
# after the datasets the arms train on. See bris/train/finetune.yaml.
DEFAULT_START = "2025-08-01"
DEFAULT_END = "2026-09-07"


def chunks(start: dt.datetime, end: dt.datetime, days: int):
    t = start
    while t < end:
        nxt = min(t + dt.timedelta(days=days), end)
        yield t, nxt
        t = nxt


def fetch_part(ids, element, t0, t1, cid) -> list[tuple[str, str, float]]:
    """One request: (station, iso hour, value) for whatever came back."""
    ref = f"{t0:%Y-%m-%dT%H:%M:%S}Z/{t1:%Y-%m-%dT%H:%M:%S}Z"
    data = frost_get("observations/v0.jsonld",
                     {"sources": ",".join(ids), "referencetime": ref,
                      "elements": element,
                      # Ask for hourly and nothing finer. Temperature and wind
                      # are instantaneous and many stations report them every
                      # ten minutes, which is six times the data for no gain:
                      # the model steps six-hourly. Precipitation is already
                      # hourly by virtue of the element name.
                      "timeresolutions": "PT1H"}, cid).get("data", [])
    out = []
    for rec in data:
        sid = rec.get("sourceId", "").split(":")[0]
        when = rec.get("referenceTime", "")
        for ob in rec.get("observations", []):
            if ob.get("elementId") != element:
                continue
            level = ob.get("level") or {}
            # Frost returns the same element at several heights for some
            # stations; take the standard one and skip the rest.
            if element == "air_temperature" and level and level.get("value") not in (2, None):
                continue
            if element == "wind_speed" and level and level.get("value") not in (10, None):
                continue
            try:
                out.append((sid, when, float(ob["value"])))
            except (TypeError, ValueError):
                pass
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--elements", default="precipitation",
                    help="comma separated: " + ", ".join(ELEMENTS))
    ap.add_argument("--days", type=int, default=10,
                    help="days per request (default 10)")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path.home() / "bris-runs" / "observations")
    args = ap.parse_args()

    wanted = [e.strip() for e in args.elements.split(",") if e.strip()]
    unknown = [e for e in wanted if e not in ELEMENTS]
    if unknown:
        raise SystemExit(f"unknown element(s): {', '.join(unknown)}")

    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    ensure_ca_bundle()
    cid = client_id()

    cache = args.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"=== stations")
    stations = norwegian_stations(cid)
    if not stations:
        raise SystemExit("Frost named no Norwegian stations; nothing to fetch.")
    ids = [s["id"] for s in stations]
    print(f"  {len(ids)} Norwegian stations\n")

    batches = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    windows = list(chunks(start, end, args.days))
    total = len(batches) * len(windows)
    print(f"=== {args.start} .. {args.end}")
    print(f"  {len(batches)} station batches x {len(windows)} windows "
          f"= {total} requests per element\n")

    for name in wanted:
        element, unit = ELEMENTS[name]
        rows: list[tuple[str, str, float]] = []
        done = 0
        started = dt.datetime.now()
        for bi, batch in enumerate(batches):
            for t0, t1 in windows:
                part = cache / f"{name}-{bi:03d}-{t0:%Y%m%d}.json"
                if part.exists():
                    rows += [tuple(r) for r in json.loads(part.read_text())]
                else:
                    got = fetch_part(batch, element, t0, t1, cid)
                    part.write_text(json.dumps(got))
                    rows += got
                done += 1
                if done % 25 == 0 or done == total:
                    per = (dt.datetime.now() - started).total_seconds() / done
                    left = dt.timedelta(seconds=int(per * (total - done)))
                    print(f"  {name}: {done}/{total}  {len(rows):,} values  "
                          f"about {left} left")

        if not rows:
            print(f"  {name}: nothing returned; skipping\n")
            continue

        # ---- keep whole hours only ------------------------------------------
        # The time axis is the union of every timestamp seen. A single station
        # reporting every ten minutes therefore adds five empty columns for
        # every station that reports hourly, and the hourly ones then look
        # five sixths absent. The first run of this reported three usable
        # temperature stations out of 1120 for exactly that reason.
        #
        # Applied here rather than only in the request, so answers already
        # cached are fixed without asking Frost for them again.
        whole = [r for r in rows if r[1][14:16] == "00" and r[1][17:19] == "00"]
        dropped = len(rows) - len(whole)
        if dropped:
            print(f"  {name}: dropped {dropped:,} sub-hourly values of "
                  f"{len(rows):,}")
        rows = whole
        if not rows:
            print(f"  {name}: nothing left on the hour; skipping\n")
            continue

        # ---- dense grid, stations that said nothing dropped ------------------
        seen_ids = sorted({r[0] for r in rows})
        times = sorted({r[1] for r in rows})
        si = {s: i for i, s in enumerate(seen_ids)}
        ti = {t: i for i, t in enumerate(times)}
        grid = np.full((len(seen_ids), len(times)), np.nan, dtype="float32")
        for sid, when, value in rows:
            grid[si[sid], ti[when]] = value

        meta = {s["id"]: s for s in stations}
        out = args.out / f"{name}.npz"
        np.savez_compressed(
            out,
            values=grid,
            stations=np.array(seen_ids),
            times=np.array(times),
            lat=np.array([meta[s]["lat"] for s in seen_ids], dtype="float64"),
            lon=np.array([meta[s]["lon"] for s in seen_ids], dtype="float64"),
            element=element, unit=unit, start=args.start, end=args.end,
        )
        filled = float(np.isfinite(grid).mean())
        print(f"\n  {name}: {len(seen_ids)} stations x {len(times)} hours, "
              f"{filled:.1%} present")
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)\n")

    print("Done. The cache under", cache, "can be deleted once these look right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

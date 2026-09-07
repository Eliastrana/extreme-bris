#!/usr/bin/env python3
"""Find the holes in the MEPS archive before a build walks into them.

    scripts/scan_meps_archive.py --start 2025-09-08 --end 2026-09-06 \
        -o ~/bris-runs/recipes/missing-year1.txt

WHY. A year-long build spends five hours retrieving and then dies on a single
absent cycle: 2026-09-02T00Z is simply not in the archive, and the 404 took
down a job that was 99% finished. Worse are the days that answer but answer
short. 2026-01-31 serves pressure-level files with six levels where every
other day has thirteen, so the four states of that day loaded, wrote NaN, and
passed every structural check. Only the finite-fraction test in the sbatch
caught them.

Both failures are knowable in advance for the price of some small HTTP
requests, and both have the same remedy: list the dates in the recipe's
`dates.missing`, so anemoi records them as missing instead of dying on one and
silently zeroing the other.

WHAT IT COSTS THE SERVER. One catalogue page per day, then one .dds per
pressure-level cycle. The .dds is the header only, not the data. Keep the
worker count low: thredds.met.no is a shared public service.

TWO KINDS OF HOLE, ONE OUTPUT. Absent files and short files are indistinguish-
able once the build is running, and the recipe treats them the same way, so
they are reported together and separately explained in the summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://thredds.met.no/thredds"
CATALOG = BASE + "/catalog/meps25epsarchive/{y:04d}/{m:02d}/{d:02d}/catalog.html"
DODS = BASE + "/dodsC/meps25epsarchive/{y:04d}/{m:02d}/{d:02d}/{name}.ncml.dds"

# The recipe asks for these twelve. The archive normally serves thirteen; a day
# that serves six cannot satisfy the request and yields NaN without complaining.
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000]

NCML = re.compile(r"meps_det_(sfc|pl)_(\d{8}T\d{2}Z)\.ncml")
PRESSURE = re.compile(r"Float32 pressure\[pressure = (\d+)\]")


def cycles(start: dt.datetime, end: dt.datetime, hours: int):
    t = start
    while t <= end:
        yield t
        t += dt.timedelta(hours=hours)


def day_catalog(session: requests.Session, day: dt.date) -> set[str]:
    """Names present in one day's catalogue, or an empty set if the day is gone."""
    url = CATALOG.format(y=day.year, m=day.month, d=day.day)
    try:
        r = session.get(url, timeout=30)
    except requests.RequestException:
        return set()
    if r.status_code != 200:
        return set()
    return {f"meps_det_{k}_{s}" for k, s in NCML.findall(r.text)}


def level_count(session: requests.Session, t: dt.datetime) -> int:
    """Pressure levels the pl file for this cycle actually offers. -1 on error."""
    url = DODS.format(y=t.year, m=t.month, d=t.day,
                      name=f"meps_det_pl_{t:%Y%m%dT%H}Z")
    try:
        r = session.get(url, timeout=45)
    except requests.RequestException:
        return -1
    if r.status_code != 200:
        return -1
    m = PRESSURE.search(r.text)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--frequency", type=int, default=6, help="hours between states")
    ap.add_argument("--workers", type=int, default=6,
                    help="keep this modest; thredds is shared")
    ap.add_argument("--skip-levels", action="store_true",
                    help="only check presence, not level count")
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    wanted = list(cycles(start, end, args.frequency))
    days = sorted({t.date() for t in wanted})
    print(f"{len(wanted)} states over {len(days)} days", file=sys.stderr)

    session = requests.Session()

    # Stage 1: presence. One request per day covers every cycle in it.
    print("catalogues ...", file=sys.stderr)
    with ThreadPoolExecutor(args.workers) as pool:
        present = dict(zip(days, pool.map(lambda d: day_catalog(session, d), days)))

    absent = [t for t in wanted
              if f"meps_det_sfc_{t:%Y%m%dT%H}Z" not in present[t.date()]
              or f"meps_det_pl_{t:%Y%m%dT%H}Z" not in present[t.date()]]
    print(f"  absent: {len(absent)}", file=sys.stderr)

    # Stage 2: shape. A file that exists can still be too thin to use.
    short: list[tuple[dt.datetime, int]] = []
    if not args.skip_levels:
        check = [t for t in wanted if t not in set(absent)]
        print(f"headers for {len(check)} pl files ...", file=sys.stderr)
        with ThreadPoolExecutor(args.workers) as pool:
            for t, n in zip(check, pool.map(lambda t: level_count(session, t), check)):
                if n < len(LEVELS):
                    short.append((t, n))
        print(f"  short: {len(short)}", file=sys.stderr)

    bad = sorted(set(absent) | {t for t, _ in short})

    print(f"\n=== {len(bad)} unusable of {len(wanted)} states "
          f"({100 * len(bad) / len(wanted):.2f}%)")
    if absent:
        print(f"--- absent from the archive ({len(absent)}) ---")
        for t in absent:
            print(f"  {t:%Y-%m-%dT%H:%M:%S}")
    if short:
        print(f"--- present but too few pressure levels ({len(short)}, "
              f"need {len(LEVELS)}) ---")
        for t, n in short:
            print(f"  {t:%Y-%m-%dT%H:%M:%S}  {n} levels")

    if args.out:
        args.out.write_text("".join(f"{t:%Y-%m-%dT%H:%M:%S}\n" for t in bad))
        print(f"\nwrote {len(bad)} dates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

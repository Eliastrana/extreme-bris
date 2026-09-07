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

A THIRD KIND IS NOT A HOLE. The first version of this returned "absent" for
any request that did not succeed, so when the login node turned out to sit
behind a TLS-inspecting proxy that Python does not trust, it declared all 1456
states missing and wrote a file that would have emptied the whole recipe. An
unreachable server is not an empty archive. Anything that is not a clean 200
or a clean 404 is now counted separately and makes the run exit non-zero, so
a broken scan cannot be mistaken for a scanned year.

Run it where TLS works. curl on the login node trusts the proxy root and
Python does not; the compute nodes reach thredds directly:

    srun -p defq -n1 -t 30 scripts/scan_meps_archive.py --start ... --end ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import re
import sys
import threading
import time
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

# One Session per thread. requests.Session is documented as not thread-safe,
# and sharing one across the pool wedged a scan of year two: six workers sat
# in poll() for ten minutes with no CPU while single requests to the same
# server were answering in a tenth of a second. Thread-local sessions keep the
# connection reuse without the shared mutable state.
_local = threading.local()

# Progress counter, so a hang is visible while it is happening rather than
# inferred afterwards from thread stacks.
_done = itertools.count(1)
_progress_lock = threading.Lock()
_last_report = [0.0]


def session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = _local.session = requests.Session()
    return s


def reset_progress() -> None:
    """The counter is module state, so the second stage must start from one."""
    global _done
    _done = itertools.count(1)
    _last_report[0] = 0.0


def tick(total: int) -> None:
    n = next(_done)
    now = time.time()
    with _progress_lock:
        if now - _last_report[0] > 5 or n == total:
            _last_report[0] = now
            print(f"    {n}/{total}", end="\r", file=sys.stderr, flush=True)


def get(url: str, timeout: int, tries: int = 3):
    """(response, error). Retries transient failures; a 404 is an answer.

    A read that wedges is far more likely than a server that is genuinely
    slow, so the timeout is short and a retry cheap.
    """
    last = ""
    for attempt in range(tries):
        try:
            return session().get(url, timeout=timeout), None
        except requests.RequestException as exc:
            last = type(exc).__name__
            # A wedged connection stays wedged; drop the pool before retrying.
            _local.session = None
            time.sleep(2 ** attempt)
    return None, last


def cycles(start: dt.datetime, end: dt.datetime, hours: int):
    t = start
    while t <= end:
        yield t
        t += dt.timedelta(hours=hours)


def day_catalog(day: dt.date, timeout: int, total: int):
    """(names, error) for one day. A 404 is an empty day; anything else is an error.

    Never collapse the two. A day that is genuinely gone and a day the scanner
    could not ask about look identical in the return value if errors are
    swallowed, and the second one silently deletes a real day from the recipe.
    """
    url = CATALOG.format(y=day.year, m=day.month, d=day.day)
    r, err = get(url, timeout)
    tick(total)
    if err:
        return set(), err
    if r.status_code == 404:
        return set(), None
    if r.status_code != 200:
        return set(), f"HTTP {r.status_code}"
    return {f"meps_det_{k}_{s}" for k, s in NCML.findall(r.text)}, None


def level_count(t: dt.datetime, timeout: int, total: int):
    """(levels, error) for one cycle's pl file. levels is None when unknown."""
    url = DODS.format(y=t.year, m=t.month, d=t.day,
                      name=f"meps_det_pl_{t:%Y%m%dT%H}Z")
    r, err = get(url, timeout)
    tick(total)
    if err:
        return None, err
    if r.status_code == 404:
        return 0, None
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    m = PRESSURE.search(r.text)
    if not m:
        return None, "no pressure dimension in .dds"
    return int(m.group(1)), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--frequency", type=int, default=6, help="hours between states")
    ap.add_argument("--workers", type=int, default=4,
                    help="keep this modest; thredds is shared")
    ap.add_argument("--timeout", type=int, default=20,
                    help="seconds per request before retrying")
    ap.add_argument("--skip-levels", action="store_true",
                    help="only check presence, not level count")
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    wanted = list(cycles(start, end, args.frequency))
    days = sorted({t.date() for t in wanted})
    print(f"{len(wanted)} states over {len(days)} days", file=sys.stderr)

    errors: dict[str, str] = {}

    # Stage 1: presence. One request per day covers every cycle in it.
    print(f"catalogues ({len(days)}) ...", file=sys.stderr)
    reset_progress()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(
            lambda d: day_catalog(d, args.timeout, len(days)), days))
    print(file=sys.stderr)
    present = {}
    for day, (names, err) in zip(days, results):
        present[day] = names
        if err:
            errors[f"{day} catalogue"] = err

    bad_days = {day for day, (_n, err) in zip(days, results) if err}
    absent = [t for t in wanted
              if t.date() not in bad_days
              and (f"meps_det_sfc_{t:%Y%m%dT%H}Z" not in present[t.date()]
                   or f"meps_det_pl_{t:%Y%m%dT%H}Z" not in present[t.date()])]
    print(f"  absent: {len(absent)}", file=sys.stderr)

    # Stage 2: shape. A file that exists can still be too thin to use.
    short: list[tuple[dt.datetime, int]] = []
    if not args.skip_levels:
        skip = set(absent)
        check = [t for t in wanted if t not in skip and t.date() not in bad_days]
        print(f"headers for {len(check)} pl files ...", file=sys.stderr)
        reset_progress()
        with ThreadPoolExecutor(args.workers) as pool:
            for t, (n, err) in zip(check, pool.map(
                    lambda t: level_count(t, args.timeout, len(check)), check)):
                if err:
                    errors[f"{t:%Y-%m-%dT%H} levels"] = err
                elif n < len(LEVELS):
                    short.append((t, n))
        print(f"\n  short: {len(short)}", file=sys.stderr)

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

    # A scan that could not ask is not a scan that found nothing. Say so, and
    # do not write a list that would delete days the archive may well hold.
    if errors:
        kinds: dict[str, int] = {}
        for v in errors.values():
            kinds[v] = kinds.get(v, 0) + 1
        print(f"\n=== {len(errors)} REQUESTS FAILED - result is incomplete")
        for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {k}")
        print("  no list written; rerun where thredds is reachable")
        return 2

    if args.out:
        args.out.write_text("".join(f"{t:%Y-%m-%dT%H:%M:%S}\n" for t in bad))
        print(f"\nwrote {len(bad)} dates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

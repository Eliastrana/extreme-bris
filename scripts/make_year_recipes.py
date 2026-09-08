#!/usr/bin/env python3
"""Emit one recipe per year-chunk, for both halves of the cutout.

    ~/bris-data-env/bin/python scripts/make_year_recipes.py \
        --meta ~/bris-runs/ckpt-metadata.json -o ~/bris-runs/recipes

The MEPS archive runs from 2023-10-01 to now, so it splits into three chunks.
They are built separately rather than as one dataset for three reasons: a
failure costs one chunk instead of everything, progress is visible, and the
newest year - the one worth having first - lands first.

anemoi opens several zarrs as one dataset with `concat`, so splitting here
costs nothing at training time.

WHY THE NEWEST YEAR IS CHUNK 1. Everything from roughly May 2025 has all eight
daily cycles; before that the archive is thinned to 00/06/12/18. Only in the
un-thinned window can training pairs be built at a 3 h offset (03->09, 09->15),
which doubles the samples available exactly where oversampling needs to find
extreme cases. Build that year first, in case the others never get built.

BOTH HALVES. Fine-tuning needs the global side too, so each chunk gets a MARS
recipe alongside its MEPS one. The MEPS half is the slow one - it is read over
OPeNDAP, one aggregation per state - while MARS is a tape archive with its own
queue. They are independent and can run at the same time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

# End of the archive is "yesterday" in practice; the last cycles of today may
# not be written yet. Chunk 1 is the newest full year.
CHUNKS = [
    ("year1", dt.datetime(2025, 9, 8, 0), dt.datetime(2026, 9, 6, 18)),
    ("year2", dt.datetime(2024, 9, 8, 0), dt.datetime(2025, 9, 7, 18)),
    # The archive begins 2023-10-01, so the oldest chunk is short by two months.
    # 06, not 00. Precipitation for a state comes from the cycle six hours
    # earlier, and 2023-09-30 is not in the archive: the first state of the
    # archive cannot be the first state of a dataset.
    ("year3", dt.datetime(2023, 10, 1, 6), dt.datetime(2024, 9, 7, 18)),
]

ARCHIVE_START = dt.datetime(2023, 10, 1, 0)


def dates_block(start: dt.datetime, end: dt.datetime,
                missing: list[dt.datetime] | None = None) -> str:
    """The recipe's dates block, with known-bad states declared missing.

    Six states of year one cannot be built: 2026-09-02 has no 00Z or 06Z cycle
    at all, and every cycle of 2026-01-31 serves six pressure levels where the
    recipe asks for twelve. Left undeclared, the first kills the build with a
    404 after five hours of retrieval and the second writes NaN quietly.
    Declared here, anemoi records them as missing and carries on.

    The same list goes into both halves even though only MEPS has the holes.
    The two are joined by cutout, so a date axis that differs between them is
    a problem waiting for a later run to find; losing six states of global
    analysis costs 0.4% and keeps the halves identical.
    """
    out = (f"dates:\n"
           f"  start: {start:%Y-%m-%dT%H:%M:%S}\n"
           f"  end: {end:%Y-%m-%dT%H:%M:%S}\n"
           f"  frequency: 6h\n")
    if missing:
        out += "  missing:\n"
        out += "".join(f"  - {d:%Y-%m-%dT%H:%M:%S}\n" for d in sorted(missing))
    return out


def replace_dates(text: str, start: dt.datetime, end: dt.datetime,
                  missing: list[dt.datetime] | None = None) -> str:
    """Swap the recipe's dates block, leaving every other line untouched.

    The URL templates must survive: each state is read from its own cycle as
    its own analysis, which only works because scripts/anemoi_create.py repairs
    iterate_patterns. Pinning a cycle here would silently make every t0 a
    six-hour forecast.
    """
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("dates:"):
            out.append(dates_block(start, end, missing))
            skipping = True
            continue
        if skipping:
            # inside the block while lines are indented or blank
            if line.strip() == "" or line.startswith(" "):
                continue
            skipping = False
        out.append(line)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path,
                    default=Path.home() / "bris-runs" / "ckpt-metadata.json")
    ap.add_argument("--meps-base", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "bris" / "configs" / "meps_2p5km.yaml")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path.home() / "bris-runs" / "recipes")
    ap.add_argument("--data-dir", default="$BRIS_DATA_DIR")
    ap.add_argument("--missing", type=Path, action="append", default=[],
                    help="file of ISO datetimes to declare missing; repeatable. "
                         "Produce one with scripts/scan_meps_archive.py.")
    args = ap.parse_args()

    # Only the dates inside a chunk belong in that chunk's recipe, so read the
    # whole set once and filter per chunk rather than trusting file names.
    missing_all: list[dt.datetime] = []
    for f in args.missing:
        if not f.exists():
            print(f"ERROR: no missing-dates file at {f}", file=sys.stderr)
            return 1
        for line in f.read_text().split():
            missing_all.append(dt.datetime.fromisoformat(line))

    if not args.meps_base.exists():
        print(f"ERROR: no MEPS recipe at {args.meps_base}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.meps_base.read_text()

    gen = Path(__file__).resolve().parent / "make_era5_recipe.py"
    made = []

    for name, start, end in CHUNKS:
        if start < ARCHIVE_START:
            print(f"WARNING: {name} starts before the archive ({ARCHIVE_START:%Y-%m-%d})",
                  file=sys.stderr)
        states = int((end - start).total_seconds() // (6 * 3600)) + 1

        gaps = [d for d in missing_all if start <= d <= end]

        meps = args.out / f"meps-{name}.yaml"
        meps.write_text(replace_dates(base, start, end, gaps))

        # The MARS generator is date-driven; emit it for the chunk's end and
        # then widen the dates block to the whole chunk.
        od = args.out / f"od-{name}.yaml"
        r = subprocess.run(
            [sys.executable, str(gen), str(args.meta), "--source", "od",
             "--date", f"{end:%Y-%m-%dT%H:%M:%S}", "-o", str(od)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"ERROR generating {od.name}:\n{r.stderr}", file=sys.stderr)
            return 1
        od.write_text(replace_dates(od.read_text(), start, end, gaps))

        made.append((name, start, end, states, meps, od))
        print(f"{name}: {start:%Y-%m-%d} .. {end:%Y-%m-%d}  {states:5d} states"
              + (f"  ({len(gaps)} declared missing)" if gaps else ""))
        print(f"   {meps}")
        print(f"   {od}")

    print("\n--- submit, newest year first ---")
    for name, _s, _e, _n, meps, od in made:
        print(f"# {name}")
        print(f"sbatch --export=ALL,RECIPE={meps},"
              f"OUT={args.data_dir}/meps-2p5km-{name}-6h-v1.zarr,THREADS=4 \\\n"
              f"    bris/slurm/build_dataset.sbatch")
        print(f"sbatch --export=ALL,RECIPE={od},"
              f"OUT={args.data_dir}/od-an-n320-{name}-6h-v1.zarr,THREADS=4 \\\n"
              f"    bris/slurm/build_dataset.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

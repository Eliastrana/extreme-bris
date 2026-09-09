#!/usr/bin/env python3
"""Turn find_extreme_states.py output into a series the blog can draw.

    scripts/extremes_to_web.py ~/bris-runs/extremes/meps-year*.json \
        --field max_mm -o ~/alertall/public/data/meps-daily-max.json

WHY A SEPARATE STEP. The ranking output is a record per state: twelve fields
each, 538 KB for a year and 1.5 MB for three. A browser would load that
happily, and it would still be the wrong thing to send. A year is 1456 states
against roughly 950 pixels of chart, so there are more points than pixels to
draw them on; the line becomes noise that hides exactly the peaks it exists to
show. And the chart uses one field of the twelve.

So this reduces to what a chart can actually render: one value per day, one
field, rounded. Around 365 points a year, which is about two and a half pixels
per point at the article's width.

DAYS RUN 06 TO 06 UTC, matching the gauge sums the station work uses. That
window is not cosmetic: comparing against a 00-00 day flattered a score by
0.08 once already.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELDS = {
    "max_mm": ("max_mm", "Kraftigste punkt"),
    "sum_mean_mm": ("sum_mean_mm", "Områdemiddel over døgnet"),
    "max_frac_exceed": ("max_frac_exceed", "Andel punkter over egen persentil"),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--field", choices=sorted(FIELDS), default="max_mm")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--compare", type=Path,
                    help="a second extremes file, drawn as the dashed series")
    ap.add_argument("--series", action="store_true",
                    help="keep the inputs apart as named series the chart can "
                         "switch on and off, rather than merging them")
    ap.add_argument("--names", help="comma-separated names for --series; "
                                    "defaults to each file's stem")
    args = ap.parse_args()

    def series(paths: list[Path]) -> list[dict]:
        key = FIELDS[args.field][0]
        rows: dict[str, float] = {}
        for p in paths:
            if not p.exists():
                # An unmatched shell glob arrives here as a literal path with
                # a star still in it. Saying "no file at .../meps-year*.json"
                # sends the reader looking for a typo; the real answer is that
                # the ranking has not been run yet.
                if any(ch in str(p) for ch in "*?["):
                    print(f"ERROR: nothing matched {p}\n"
                          "The ranking has not produced these files yet. Run "
                          "bris/slurm/find_extremes.sbatch\non each dataset "
                          "first; it writes one JSON per dataset.",
                          file=sys.stderr)
                else:
                    print(f"ERROR: no file at {p}", file=sys.stderr)
                raise SystemExit(2)
            doc = json.loads(p.read_text())
            for d in doc.get("days", []):
                if key in d:
                    rows[d["day"]] = round(float(d[key]), args.round)
        # Several years arrive as several files; sort so the axis is monotone
        # rather than trusting the order the shell expanded the glob in.
        return [{"x": day, "y": rows[day]} for day in sorted(rows)]

    # One file per year, kept apart. The years do not overlap in time, so a
    # merged line would run continuously across all three and there would be
    # nothing to switch off; as named series the reader can isolate a year.
    if args.series:
        names = (args.names.split(",") if args.names
                 else [p.stem for p in args.inputs])
        if len(names) != len(args.inputs):
            print(f"ERROR: {len(names)} names for {len(args.inputs)} files",
                  file=sys.stderr)
            return 2
        out = {"series": []}
        for name, path in zip(names, args.inputs):
            rows = series([path])
            if not rows:
                print(f"ERROR: no daily records in {path}", file=sys.stderr)
                return 2
            out["series"].append({"name": name.strip(), "data": rows})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out))
        total = sum(len(s["data"]) for s in out["series"])
        print(f"{len(out['series'])} series, {total} days, {args.field}, "
              f"{args.out.stat().st_size / 1024:.0f} KB -> {args.out}")
        for s in out["series"]:
            print(f"  {s['name']:10s} {len(s['data']):4d} days  "
                  f"{s['data'][0]['x']} .. {s['data'][-1]['x']}  "
                  f"max {max(p['y'] for p in s['data'])}")
        return 0

    data = series(args.inputs)
    if not data:
        print("ERROR: no daily records found; was the ranking run?",
              file=sys.stderr)
        return 2

    out: dict = {"data": data}
    if args.compare:
        out["compare"] = series([args.compare])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out if args.compare else data))
    size = args.out.stat().st_size
    print(f"{len(data)} days, {args.field}, {size/1024:.0f} KB -> {args.out}")
    print(f"  {data[0]['x']} .. {data[-1]['x']}   "
          f"max {max(p['y'] for p in data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

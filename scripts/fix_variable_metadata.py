#!/usr/bin/env python3
"""Make the Nordic half's variable metadata say what its arrays are called.

    scripts/fix_variable_metadata.py --dry-run
    scripts/fix_variable_metadata.py

WHAT IS WRONG. anemoi-datasets 0.5.24 cannot run its conversion filters on
OPeNDAP input, so the MEPS recipe renames each field to its ECMWF name while it
still holds MET's quantity, and postprocess_meps.py converts the values. The
rename changes the array's name. It does not change the metadata anemoi stores
alongside it, which keeps saying x_wind_10m for the array now called 10u.

Nothing noticed until training started. The data path goes by array name, so
every check, every statistic and every plot was reading the right column. The
loss scalers go by metadata, and they stop with

    Variable 10u is not allowed to have a separate scaling besides x_wind_10m

because the packaged config assigns a weight to a name the dataset does not
admit to having.

WHY COPY RATHER THAN PATCH. The global half came through MARS and carries
proper ECMWF metadata for the same 98 variables, and anemoi is content with it.
Rewriting the Nordic half's entries by hand would mean guessing at conventions
this project has already guessed wrong about twice; copying the half that works
guesses at nothing. It also carries across things the Nordic build never
recorded, such as precipitation being a six-hour accumulation, which is true of
it since the recipe was fixed.

NOTHING IS LOST AND NOTHING IS RECOMPUTED. Only the attributes change; not one
byte of data is read or written. The originals are kept under
variables_metadata_source, so this is reversible and so the provenance of a
column is still recoverable afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("zarr")

BACKUP = "variables_metadata_source"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", type=Path,
                    help="zarrs to fix (default: the three MEPS years)")
    ap.add_argument("--reference", type=Path,
                    default=Path.home() / "bris-data" / "od-an-n320-year3-6h-v1.zarr",
                    help="the half whose metadata anemoi already accepts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import zarr

    targets = args.datasets or [
        Path.home() / "bris-data" / f"meps-2p5km-year{y}-6h-v1.zarr" for y in (1, 2, 3)
    ]

    ref = zarr.open(str(args.reference), mode="r")
    ref_meta = ref.attrs.get("variables_metadata") or {}
    if not ref_meta:
        raise SystemExit(f"{args.reference.name} carries no variables_metadata")
    print(f"reference: {args.reference.name}, {len(ref_meta)} variables\n")

    for path in targets:
        z = zarr.open(str(path), mode="r" if args.dry_run else "r+")
        names = list(z.attrs["variables"])
        meta = z.attrs.get("variables_metadata") or {}
        print(f"=== {path.name}")

        if z.attrs.get(BACKUP):
            print("  already fixed; the original is under "
                  f"{BACKUP}. Skipping.\n")
            continue

        missing = [n for n in names if n not in ref_meta]
        if missing:
            print(f"  ERROR: the reference has no metadata for "
                  f"{len(missing)} variable(s): {', '.join(missing[:6])}",
                  file=sys.stderr)
            print("  Copying would leave those entries stale. Nothing written.",
                  file=sys.stderr)
            return 1

        changed = []
        for n in names:
            was = (meta.get(n) or {}).get("mars", {}).get("param")
            now = (ref_meta[n] or {}).get("mars", {}).get("param")
            if was != now:
                changed.append((n, was, now))

        print(f"  {len(changed)} of {len(names)} variables name something else")
        for n, was, now in changed[:8]:
            print(f"    {n:8s} {was}  ->  {now}")
        if len(changed) > 8:
            print(f"    ... and {len(changed) - 8} more")

        if args.dry_run:
            print("  Dry run; nothing written.\n")
            continue

        z.attrs[BACKUP] = json.loads(json.dumps(meta))
        z.attrs["variables_metadata"] = {n: ref_meta[n] for n in names}
        print(f"  rewritten; original kept under {BACKUP}\n")

    print("Attributes only. No data was read or written, so this takes seconds\n"
          "and can be undone by copying the backup key back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

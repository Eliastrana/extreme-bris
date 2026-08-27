#!/usr/bin/env python3
"""Generate an anemoi-datasets recipe for the global (N320) side of the cutout.

The variable list, pressure levels and date range are derived from the
checkpoint metadata rather than typed by hand, because getting them subtly wrong
is the most likely way to build a dataset the model silently rejects.

    python scripts/make_era5_recipe.py ~/bris-runs/ckpt-metadata.json \
        --date 2025-04-01T00:00:00 -o bris/configs/era5_n320.yaml

What it emits is a starting point, not a finished recipe: the `input:` stanza
needs checking against the anemoi-datasets version actually installed, and ERA5
retrieval needs CDS credentials in ~/.cdsapirc. Everything below `input:` —
the variables, levels, grid and dates — is checkpoint-derived and should be
taken literally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# Accumulated fields. ERA5's analysis stream does not carry these — they exist
# only in the forecast stream, as accumulations over a period. Requesting them
# with type: an silently drops them, which shows up as a dataset with three
# fewer variables than expected and no error anywhere.
ACCUMULATED = {"tp", "ssrd", "strd", "cp", "sf", "e", "ro", "sshf", "slhf", "ssr", "str", "tsr", "ttr"}

# Computed by anemoi-datasets at load time; never stored in the dataset.
COMPUTED = {
    "cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time",
    "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
    "insolation",
}


def all_variables(meta: dict) -> list[str]:
    """The full ordered variable list, however this checkpoint spells it."""
    v = meta.get("dataset", {}).get("variables")
    if isinstance(v, list) and len(v) > 20:
        return [str(x) for x in v]
    n2i = meta.get("dataset", {}).get("name_to_index")
    if isinstance(n2i, dict) and len(n2i) > 20:
        return [n for n, _ in sorted(n2i.items(), key=lambda kv: kv[1])]
    return []


def stored_fields(meta: dict) -> list[str]:
    """Every field the dataset must physically contain.

    data_indices holds integer positions, not names, so they are resolved
    through the variable list. Falling back to the whole list is safe: the only
    entries dropped are the computed forcings.
    """
    names = all_variables(meta)
    di = meta.get("data_indices", {}).get("data", {})
    found: set[str] = set()
    for section in ("input", "output"):
        for group, vals in di.get(section, {}).items():
            if group in ("full", "target") or not isinstance(vals, list):
                continue
            for v in vals:
                if isinstance(v, str):
                    found.add(v)
                elif isinstance(v, int) and names and 0 <= v < len(names):
                    found.add(names[v])
    if not found:
        found = set(names)
    return sorted(found - COMPUTED)


def split(fields: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    surface, levels = [], {}
    for f in fields:
        if "_" in f and f.rsplit("_", 1)[1].isdigit():
            base, lev = f.rsplit("_", 1)
            levels.setdefault(base, []).append(int(lev))
        else:
            surface.append(f)
    return sorted(surface), {k: sorted(v) for k, v in sorted(levels.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata", type=Path, help="ckpt-metadata.json from inspect_checkpoint.py")
    ap.add_argument("--date", default="2025-04-01T00:00:00", help="initialisation time (t0)")
    ap.add_argument("--multistep", type=int, default=2, help="input states required")
    ap.add_argument("--frequency", default="6h")
    ap.add_argument("-o", "--out", type=Path, default=Path("era5_n320.yaml"))
    args = ap.parse_args()

    meta = json.loads(args.metadata.read_text())
    fields = stored_fields(meta)
    if not fields:
        print("ERROR: no variables found in metadata", file=sys.stderr)
        return 1
    surface_all, levels = split(fields)
    surface = [f for f in surface_all if f not in ACCUMULATED]
    accum = [f for f in surface_all if f in ACCUMULATED]

    step_h = int(args.frequency.rstrip("h"))
    t0 = dt.datetime.fromisoformat(args.date)
    start = t0 - dt.timedelta(hours=step_h * (args.multistep - 1))

    lev_set = sorted({l for v in levels.values() for l in v})
    if len({tuple(v) for v in levels.values()}) > 1:
        print("WARNING: pressure levels differ between variables; using the union",
              file=sys.stderr)

    forcings_block = """    - forcings:
        # The nine computed forcings. The model needs them as inputs, and the
        # inference config selects them by name, so they must be stored
        # variables in the dataset — hence KeyError: 'cos_julian_day' when they
        # were absent. `template` supplies the grid only; the values depend on
        # position and time alone.
        template: ${input.join.0.mars}
        param:
          - cos_julian_day
          - sin_julian_day
          - cos_local_time
          - sin_local_time
          - cos_latitude
          - sin_latitude
          - cos_longitude
          - sin_longitude
          - insolation
"""

    accum_block = ""
    if accum:
        accum_block = f"""    - accumulations:
        # tp/ssrd/strd are accumulations and do not exist in the analysis
        # stream. This source pulls them from forecasts and accumulates over
        # the period, matching the dataset frequency.
        #
        # The key is `accumulation_period`, not `user_accumulation_period` —
        # the YAML entrypoint is accumulations(context, dates,
        # use_cdsapi_dataset=None, **request), so anything else falls through
        # into the MARS request and is rejected there. No `type:` here either:
        # the source selects the forecast stream itself. For class ea it
        # applies data_accumulation_period=1 with base_times (6, 18).
        use_cdsapi_dataset: reanalysis-era5-complete
        class: ea
        expver: "0001"
        stream: oper
        grid: N320
        levtype: sfc
        param: {accum}
        accumulation_period: {step_h}
"""

    yaml = f"""# anemoi-datasets recipe — global N320 side of the Bris cutout.
#
# GENERATED by scripts/make_era5_recipe.py from {args.metadata.name}.
# Variables, levels and dates are checkpoint-derived — do not edit them by hand.
#
# NOT generated, and needing your attention:
#   * the `input:` stanza follows the recipe format in MET's own regional
#     tutorial (metno/anemoi-regional-tutorial), which uses `level:` rather than
#     MARS' `levelist:`. Still worth checking against the installed
#     anemoi-datasets version
#   * ERA5 retrieval needs credentials in ~/.cdsapirc, and the licence must be
#     accepted once through the CDS web form before the API will serve anything
#   * cdsapi is NOT in the inference lockfile — build datasets in a separate
#     environment, see docs/INPUTS.md
#   * `output.order_by` is deliberately omitted. Newer anemoi-datasets rejects
#     it as deprecated; 0.5.24 defaults it to
#     ['valid_datetime', 'param_level', 'number'], which is what newer versions
#     hard-code. Omitting it behaves identically on both.
#   * this is a MARS-style request (class: ea, grid: N320), which means
#     `reanalysis-era5-complete` rather than the standard CDS ERA5 datasets.
#     The standard ones return a regular 0.25 deg lat/lon grid, which does NOT
#     match the checkpoint's graph — the global side must be native N320.
#     era5-complete is API-only and served from tape, so expect hours to days
#     even for two states. Submit early.
#   * this substitutes ERA5 (class ea) for the operational analysis (class od)
#     that Bris was trained on — see docs/INPUTS.md
#
# Build with:
#   cd $BRIS_ENV_DIR && uv run anemoi-datasets create <this file> \\
#       $BRIS_DATA_DIR/era5-n320-{t0:%Y%m%d}-{args.frequency}-v1.zarr

dates:
  start: {start:%Y-%m-%dT%H:%M:%S}
  end: {t0:%Y-%m-%dT%H:%M:%S}
  frequency: {args.frequency}

# multistep_input = {args.multistep}, so the range above spans {args.multistep} states
# ending at the initialisation time. A single state is not enough.

input:
  join:
    - mars:
        # Routes this MARS-style request through the CDS API instead of
        # requiring direct MARS access. era5-complete is the only ERA5 product
        # that serves the native N320 grid; the standard CDS datasets return a
        # regular 0.25 deg grid, which does not match the checkpoint's graph.
        use_cdsapi_dataset: reanalysis-era5-complete
        class: ea            # ERA5. Bris was trained on 'od'.
        expver: "0001"
        stream: oper
        type: an
        grid: N320
        levtype: sfc
        param: {surface}
    - mars:
        use_cdsapi_dataset: reanalysis-era5-complete
        class: ea
        expver: "0001"
        stream: oper
        type: an
        grid: N320
        levtype: pl
        level: {lev_set}
        param: {sorted(levels)}
{accum_block}{forcings_block}
output:
  statistics: valid_datetime
"""
    args.out.write_text(yaml)

    print(f"wrote {args.out}")
    print(f"  stored fields : {len(fields)}")
    print(f"  surface       : {len(surface)}  {', '.join(surface)}")
    print(f"  accumulated   : {len(accum)}  {', '.join(accum)}  (forecast stream)")
    print(f"  upper-air     : {len(levels)} x {len(lev_set)} levels = "
          f"{len(levels) * len(lev_set)}")
    print(f"  levels        : {lev_set}")
    print(f"  dates         : {start:%Y-%m-%dT%H} .. {t0:%Y-%m-%dT%H} "
          f"({args.multistep} states, {args.frequency})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

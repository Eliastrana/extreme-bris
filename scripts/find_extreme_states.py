#!/usr/bin/env python3
"""Rank every state in an anemoi dataset by how extreme its precipitation is.

    scripts/find_extreme_states.py $BRIS_DATA_DIR/meps-2p5km-year1-6h-v1.zarr \
        -o ~/bris-runs/extremes/year1.json

WHAT THIS IS FOR. Fine-tuning on the tail needs to know which states the tail
is in: oversampling has to draw them more often, and a threshold-weighted
score has to be told where the thresholds are. Both need a defensible,
reproducible answer to "which states are extreme", not a hand-picked list.

WHY NOT ONE NUMBER. No single measure is right, so this reports three and
lets the caller choose:

  * frac_exceed  share of sampled points above their OWN climatological
                 percentile. Finds widespread, regionally unusual rain.
  * max          the wettest point in the domain. Finds intense but local
                 events that a domain average hides completely.
  * area_ge_X    share of the domain above an absolute threshold. This is the
                 one that maps directly onto a twCRPS threshold.

PER-POINT PERCENTILES, NOT ONE ABSOLUTE THRESHOLD. The west coast gets several
times the rain of the interior. A single absolute threshold would select the
west coast in autumn and call it a finding. Percentiles computed per grid
point ask instead whether this point is having an unusual day, which is the
same logic the station work uses on gauges.

WET STATES ONLY, BY DEFAULT. Most six-hour states at most points are dry, so a
percentile over all states is dominated by zeros and says nothing. The default
computes the percentile over states where the point actually got measurable
precipitation, and drops points with too few wet samples to support one.

THE ZERO GATE. This refuses to report anything if the precipitation field is
identically zero, and that is not a hypothetical. All three MEPS year datasets
were built with `precipitation_amount_acc` read at step 0, where accumulated
precipitation is zero by definition, so 756 GB of data carried no
precipitation at all. Every existing check passed it, because the checks ask
whether values are finite and zero is perfectly finite. A field of zeros has a
mean, a maximum and a percentile, and all of them are zero, and a ranking
built on them would look like a list of ordinary days rather than an error.
Measure the thing before ranking by it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

# Absolute thresholds in mm per state. The upper ones are chosen to bracket the
# event that motivated this work: on 2025-10-04, 24 gauges recorded more than
# 75 mm in a day and the model produced none.
THRESHOLDS_MM = [0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0]

# Histogram edges in the file's own units, covering both conventions: values
# below 1 are metres (so 0.0001 is 0.1 mm) and values above are millimetres.
# Counting into these once per state means the area above any threshold is a
# tail sum afterwards, without keeping the field or reading it twice.
RAW_BINS = [0.0] + [x * 10.0 ** e for e in range(-5, 3)
                    for x in (1.0, 2.0, 3.0, 5.0)] + [np.inf]


def open_dataset(path: Path):
    z = zarr.open(str(path), mode="r")
    names = list(z.attrs["variables"])
    return z, names


def unit_scale(sample_max: float, declared: str) -> tuple[float, str]:
    """Factor that turns the stored values into millimetres.

    The two halves of the cutout do not agree: MEPS carries
    precipitation_amount_acc in kg/m2, which is millimetres, while the MARS
    side carries tp in metres like the rest of IFS. Guessing from magnitude is
    a heuristic and it is stated out loud rather than applied silently, since
    getting it wrong moves every number by a factor of a thousand.
    """
    if declared == "mm":
        return 1.0, "declared mm"
    if declared == "m":
        return 1000.0, "declared m, scaled by 1000"
    # A six-hour accumulation above 1 metre is not weather; above 1 mm in the
    # stored unit it is almost certainly already millimetres.
    if sample_max > 1.0:
        return 1.0, f"guessed mm (sample max {sample_max:.3f})"
    return 1000.0, f"guessed m (sample max {sample_max:.5f}), scaled by 1000"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--var", default="tp")
    ap.add_argument("--units", choices=["auto", "mm", "m"], default="auto")
    ap.add_argument("--stride", type=int, default=20,
                    help="spatial subsample for the percentile field")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--wet-mm", type=float, default=0.1,
                    help="a point counts as wet above this, in mm")
    ap.add_argument("--min-wet", type=int, default=30,
                    help="points with fewer wet samples get no percentile")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N states only")
    args = ap.parse_args()

    z, names = open_dataset(args.dataset)
    if args.var not in names:
        print(f"ERROR: no variable {args.var!r}; have {names[:12]}...",
              file=sys.stderr)
        return 2
    j = names.index(args.var)
    data, dates = z["data"], z["dates"][:]
    n = data.shape[0] if not args.limit else min(args.limit, data.shape[0])
    missing = {str(x)[:19] for x in z.attrs.get("missing_dates", [])}

    print(f"{args.dataset.name}: {n} states, variable {args.var!r}, "
          f"stride {args.stride}", file=sys.stderr)

    # ---- pass 1: subsample plus every whole-field summary ------------------
    # Everything that needs the full field is computed here and only scalars
    # are kept. An earlier version stashed the field itself on each record,
    # which is 1,014,481 floats times 1456 states: six gigabytes to compute
    # numbers that are a few bytes each.
    sub_n = len(range(0, data.shape[3], args.stride))
    sub = np.full((n, sub_n), np.nan, dtype="float32")
    summaries: list[dict] = [None] * n

    def one(i: int):
        if str(dates[i])[:19] in missing:
            return i, None, {"date": str(dates[i])[:19], "missing": True}
        full = np.asarray(data[i, j, 0, :], dtype="float32")
        finite = np.isfinite(full)
        nfin = int(finite.sum())
        s = {
            "date": str(dates[i])[:19],
            "missing": False,
            "raw_max": float(np.nanmax(full)) if nfin else 0.0,
            "raw_mean": float(np.nanmean(full)) if nfin else 0.0,
            "finite": nfin,
            # Counts, not fractions: the millimetre scale is not known yet, so
            # the raw thresholds are applied after the unit is settled. Store
            # the sorted field's tail instead, which is enough for any of them.
            "raw_hist": np.histogram(full[finite],
                                     bins=RAW_BINS)[0].tolist() if nfin else [],
        }
        return i, full[:: args.stride].copy(), s

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, s, summary in pool.map(one, range(n)):
            done += 1
            if done % 100 == 0 or done == n:
                print(f"  {done}/{n}", end="\r", file=sys.stderr, flush=True)
            summaries[i] = summary
            if s is not None:
                sub[i] = s
    print(file=sys.stderr)

    built = [s for s in summaries if not s["missing"]]
    if not built:
        print("ERROR: every state is declared missing", file=sys.stderr)
        return 2

    raw_max = max(s["raw_max"] for s in built)

    # ---- the gate ---------------------------------------------------------
    if raw_max <= 0.0:
        print(f"\nERROR: {args.var!r} is identically zero across all "
              f"{len(built)} states.\n"
              "There is nothing to rank. This is what a MEPS dataset built "
              "from\nprecipitation_amount_acc at step 0 looks like: the field "
              "exists, is\nfinite everywhere, and is zero everywhere, because "
              "nothing has\naccumulated at the start of a run. Fix the "
              "dataset before ranking it.",
              file=sys.stderr)
        return 3

    scale, how = unit_scale(raw_max, args.units)
    print(f"units: {how}", file=sys.stderr)

    # ---- percentile field, per point --------------------------------------
    wet = sub * scale >= args.wet_mm
    n_wet = wet.sum(axis=0)
    thresh = np.full(sub_n, np.nan, dtype="float32")
    enough = n_wet >= args.min_wet
    for k in np.flatnonzero(enough):
        thresh[k] = np.percentile(sub[wet[:, k], k], args.percentile)
    print(f"percentile field: {int(enough.sum())} of {sub_n} sampled points "
          f"have >= {args.min_wet} wet states", file=sys.stderr)

    # ---- per-state metrics -------------------------------------------------
    edges_mm = np.array(RAW_BINS[1:]) * scale
    records = []
    for i, s in enumerate(summaries):
        if s["missing"]:
            records.append({"date": s["date"], "missing": True})
            continue
        row = sub[i] * scale
        exceed = np.zeros(sub_n, dtype=bool)
        np.greater(row, thresh, out=exceed, where=enough & np.isfinite(row))
        counts = np.array(s["raw_hist"], dtype="float64")
        rec = {
            "date": s["date"],
            "missing": False,
            "max_mm": round(s["raw_max"] * scale, 3),
            "mean_mm": round(s["raw_mean"] * scale, 5),
            "frac_exceed": round(float(exceed.sum() / max(1, enough.sum())), 6),
        }
        # Area above a threshold is the tail of the histogram from that bin on.
        for t in THRESHOLDS_MM:
            k = int(np.searchsorted(edges_mm, t, side="left"))
            above = counts[k:].sum() if k < len(counts) else 0.0
            rec[f"area_ge_{t:g}"] = round(float(above / max(1, s["finite"])), 6)
        records.append(rec)

    # ---- flag the extremes -------------------------------------------------
    # A state is extreme when its exceedance share is itself in the top
    # percentile across states: unusual rain, over an unusually large area.
    vals = np.array([r["frac_exceed"] for r in records if not r["missing"]])
    cut = float(np.percentile(vals, args.percentile))
    n_flag = 0
    for r in records:
        if r["missing"]:
            continue
        r["extreme"] = bool(r["frac_exceed"] >= cut and r["frac_exceed"] > 0)
        n_flag += r["extreme"]

    # ---- daily roll-up, 06-06 UTC ------------------------------------------
    # The station work sums gauges from 06 to 06 UTC, so a day here means the
    # same window. Comparing against a 00-00 day inflated a score by 0.08 once
    # already.
    days: dict[str, dict] = {}
    for r in records:
        if r["missing"]:
            continue
        t = dt.datetime.fromisoformat(r["date"])
        day = (t - dt.timedelta(hours=6)).date().isoformat()
        d = days.setdefault(day, {"day": day, "sum_mean_mm": 0.0,
                                  "max_mm": 0.0, "states": 0,
                                  "max_frac_exceed": 0.0})
        d["sum_mean_mm"] += r["mean_mm"]
        d["max_mm"] = max(d["max_mm"], r["max_mm"])
        d["max_frac_exceed"] = max(d["max_frac_exceed"], r["frac_exceed"])
        d["states"] += 1
    daily = sorted(days.values(), key=lambda d: d["day"])
    for d in daily:
        d["sum_mean_mm"] = round(d["sum_mean_mm"], 4)

    out = {
        "dataset": str(args.dataset),
        "variable": args.var,
        "units": how,
        "percentile": args.percentile,
        "wet_mm": args.wet_mm,
        "stride": args.stride,
        "sampled_points": sub_n,
        "points_with_percentile": int(enough.sum()),
        "extreme_cut_frac_exceed": round(cut, 6),
        "n_states": len(records),
        "n_extreme": n_flag,
        "states": records,
        "days": daily,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    top = sorted((r for r in records if not r["missing"]),
                 key=lambda r: -r["frac_exceed"])[:10]
    print(f"\n{n_flag} of {len(vals)} states flagged extreme "
          f"(frac_exceed >= {cut:.4f})")
    print(f"{'date':20s} {'max mm':>8s} {'mean mm':>9s} {'exceed':>8s}")
    for r in top:
        print(f"{r['date']:20s} {r['max_mm']:8.2f} {r['mean_mm']:9.4f} "
              f"{r['frac_exceed']:8.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

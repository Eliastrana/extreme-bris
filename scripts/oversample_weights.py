#!/usr/bin/env python3
"""Weights for drawing heavy-rain training states more often (XBRIS_OVERSAMPLE).

    scripts/oversample_weights.py ~/bris-runs/extremes/meps-year3.json \\
        ~/bris-runs/extremes/meps-year2.json \\
        --start 2023-10-01 --end 2025-03-31 --measure area_ge_20 \\
        --top 0.10 --factor 4 -o ~/bris-runs/oversample/weights.json

The rankings come from scripts/find_extreme_states.py. The states whose
measure is in the top fraction of the training period get the factor, every
other state weighs 1. Only training dates are written, so nothing in the
validation or test periods can be weighted even by mistake.

With --top 0.10 and --factor 4, heavy states make up 0.4 / (0.4 + 0.9), about
31 per cent of draws instead of 10. The file records the inputs, the cut
value and the expected share, so the run that used it can be reproduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rankings", nargs="+")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--measure", default="area_ge_20")
    ap.add_argument("--top", type=float, default=0.10)
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    states = {}
    for path in a.rankings:
        for s in json.load(open(path))["states"]:
            day = s["date"][:10]
            if a.start <= day <= a.end and not s.get("missing") and s.get(a.measure) is not None:
                states[s["date"]] = float(s[a.measure])
    if not states:
        raise SystemExit("no training states in the rankings")
    values = sorted(states.values(), reverse=True)
    n_top = max(1, round(a.top * len(values)))
    cut = values[n_top - 1]
    if cut <= 0:
        raise SystemExit(f"the cut is {cut}; the measure is zero too often for --top {a.top}")
    weights = {d: a.factor for d, v in sorted(states.items()) if v >= cut}
    n_w = len(weights)
    share = n_w * a.factor / (n_w * a.factor + len(states) - n_w)
    out = {
        "rankings": [str(Path(p).resolve()) for p in a.rankings],
        "start": a.start, "end": a.end, "measure": a.measure,
        "top": a.top, "factor": a.factor, "cut": cut,
        "n_states": len(states), "n_weighted": n_w,
        "expected_share_of_draws": round(share, 4),
        "weights": weights,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{n_w} of {len(states)} states weighted x{a.factor} ({a.measure} >= {cut:.6f}); "
          f"expected share of draws {share:.3f} -> {a.out}")


if __name__ == "__main__":
    main()

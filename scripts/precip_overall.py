#!/usr/bin/env python3
"""Daily precipitation over all station-days, not just heavy ones.

EXPLORATORY. The heavy-rain scores say nothing about whether a model has
become too wet in general. From a hits_and_calibration.py case cache, this
reports for each model: mean precipitation relative to observed, the share of
wet days (ensemble mean over 1 mm), and fair CRPS over all station-days and
split into dry, 1-20 mm and over 20 mm days. Each model minus the comparator
gets a paired seven-day block bootstrap on the all-days CRPS.

  python scripts/precip_overall.py --cache ~/bris-runs/c2-replication/testyear-cases.npz \\
      --comparator control2000s2 --out ~/bris-runs/c2-replication/precip-overall.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LEAD = 30


def fair_crps(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    m = ens.shape[1]
    t1 = np.abs(ens - obs[:, None]).mean(axis=1)
    t2 = np.abs(ens[:, :, None] - ens[:, None, :]).sum(axis=(1, 2)) / (2 * m * (m - 1))
    return t1 - t2


def block_ci(diff: np.ndarray, days: np.ndarray, reps: int, seed: int = 0) -> tuple[float, float]:
    d = days.astype("datetime64[D]").astype(int)
    blk = (d - d.min()) // 7
    ub, inv = np.unique(blk, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=diff), np.bincount(inv)
    rng = np.random.default_rng(seed)
    est = [sums[i].sum() / cnts[i].sum() for i in (rng.integers(0, ub.size, ub.size) for _ in range(reps))]
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--comparator", required=True)
    ap.add_argument("--replicates", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.cache.expanduser(), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    o, days = d[f"obs_{LEAD}"], d[f"dates_{LEAD}"]
    models = {k[len("model_"):-len(f"_{LEAD}")]: v for k, v in d.items() if k.startswith("model_")}
    if args.comparator not in models:
        raise SystemExit(f"comparator {args.comparator} not in {list(models)}")
    ref = fair_crps(models[args.comparator], o)
    dry, mid, hi = o <= 0.1, (o > 1) & (o <= 20), o > 20
    report = {"exploratory": True, "station_days": int(o.size), "observed_mean_mm": float(o.mean()),
              "observed_wet_share": float((o > 1).mean()), "comparator": args.comparator, "models": {}}
    print(f"{o.size:,} station-days, observed mean {o.mean():.2f} mm, wet (>1 mm) {100 * (o > 1).mean():.1f} %")
    for lab, e in models.items():
        c, em = fair_crps(e, o), e.mean(axis=1)
        r = {"mean_ratio": float(em.mean() / o.mean()), "wet_share": float((em > 1).mean()),
             "crps": float(c.mean()), "crps_dry": float(c[dry].mean()), "crps_1_20": float(c[mid].mean()),
             "crps_over_20": float(c[hi].mean())}
        line = (f"  {lab:16s} mean x{r['mean_ratio']:.2f}  wet {100 * r['wet_share']:.1f}%  CRPS {r['crps']:.3f} "
                f"(dry {r['crps_dry']:.3f}, 1-20 {r['crps_1_20']:.3f}, >20 {r['crps_over_20']:.2f})")
        if lab != args.comparator:
            diff = c - ref
            lo, hi_ = block_ci(diff, days, args.replicates)
            r["crps_minus_comparator"] = {"mean": float(diff.mean()), "ci_lower": lo, "ci_upper": hi_,
                                          "relative": float(diff.mean() / ref.mean())}
            line += f"  vs {args.comparator} {100 * diff.mean() / ref.mean():+.1f}% [{100 * lo / ref.mean():+.1f}, {100 * hi_ / ref.mean():+.1f}]"
        report["models"][lab] = r
        print(line)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

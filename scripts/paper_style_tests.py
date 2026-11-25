#!/usr/bin/env python3
"""Why the paper-style Brier skill score and the calibrated daily one disagree. EXPLORATORY.

Measured the way Nordhagen et al. (2025, arXiv:2511.23043) measure it (raw
six-hour precipitation, bilinear, leads 6 to 36 h pooled; scripts/paper_figures.py),
Pavo + spectral is no better than the control arm + spectral over 15 and 20 mm
per 6 h, while the pre-registered calibrated daily Brier (evaluation/
bris_m16_plan.json) has it clearly better. Two things changed at once, the
accumulation window and the calibration, and the paper-style numbers carry no
uncertainty. Three tests separate them, on cases that already exist:

  A  raw 6 h Brier, with paired seven-day block bootstrap intervals
  B  the same after month-out isotonic calibration (hits_and_calibration.py)
  C  raw daily Brier at +30 h on the 98,117 station-days of the plan

If B favours Pavo where A does not, Pavo carries the information and is only
too wet; if C agrees with A, the window does not matter and calibration does.
Every number here is exploratory: none of these tests was registered.

    python scripts/paper_style_tests.py \\
        --cases ~/bris-runs/paper-figures-m16/{bris-m16,controlfft2000-m16,tailCfft2000-m16}.npz \\
        --meps-members "~/bris-runs/meps-ens/mbr0*.npz" --obs-dir ~/bris-runs/observations \\
        --daily ~/bris-runs/members/bris-cases.npz --out ~/bris-runs/paper-figures-m16/tests.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402

import paper_figures as pf  # noqa: E402
from hits_and_calibration import calibrated_cv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import paired_block_bootstrap  # noqa: E402

SIX_HOUR = [0.5, 1, 2, 5, 10, 15, 20]
DAILY = [10, 20, 50]
NAMES = {"bris-m16": "Bris", "controlfft2000-m16": "Control", "tailCfft2000-m16": "Pavo"}
PAIRS = [("Pavo", "Control"), ("Pavo", "Bris"), ("Control", "Bris"),
         ("Pavo", "MEPS15"), ("Control", "MEPS15"), ("Bris", "MEPS15")]


def six_hour_cases(paths: list[Path], meps_glob: str, obs_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Aligned precipitation ensembles, observations and init dates, as paper_figures.combine aligns them."""
    obs_all = pf.load_obs(obs_dir, "precipitation")
    sets = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            sets[NAMES[str(z["label"])]] = {k: z[f"precipitation_{k}"] for k in ("init", "lead", "station", "ens")}
    sets["MEPS15"] = pf.meps_member_cases(meps_glob, obs_all)
    keys = {lab: pf.key(s["init"], s["lead"], s["station"]) for lab, s in sets.items()}
    common = None
    for k in keys.values():
        common = k if common is None else np.intersect1d(common, k)
    aligned = {}
    for lab, s in sets.items():
        order = np.argsort(keys[lab])
        p = order[np.searchsorted(keys[lab], common, sorter=order)]
        aligned[lab] = {k: v[p] for k, v in s.items()}
    ref = aligned["Bris"]
    valid = ref["init"].astype("datetime64[h]") + ref["lead"].astype("timedelta64[h]")
    tix = np.array([obs_all["index"].get(v, -1) for v in valid])
    obs = np.full(common.size, np.nan)
    okt = tix >= 5
    st = ref["station"][okt]
    blocks = np.stack([obs_all["values"][st, tix[okt] - k] for k in range(6)], axis=1)
    obs[okt] = np.where(np.isfinite(blocks).all(axis=1), blocks.sum(axis=1), np.nan)
    keep = np.isfinite(obs)
    ens = {lab: a["ens"][keep] for lab, a in aligned.items()}
    print(f"six-hour: {int(keep.sum()):,} cases, {np.unique(ref['station'][keep]).size} stations")
    return ens, obs[keep], ref["init"][keep].astype("datetime64[D]")


def score(ens: dict, obs: np.ndarray, dates: np.ndarray, thresholds: list, replicates: int) -> dict:
    month = dates.astype("datetime64[M]").astype(int)
    out = {}
    for t in thresholds:
        event = (obs > t).astype("float64")
        clim = event.mean()
        ref = ((clim - event) ** 2).mean()
        row = {"events": int(event.sum()), "observed_freq": float(clim), "models": {}, "pairs": {}}
        bs = {}
        for lab, e in ens.items():
            p = (e > t).mean(axis=1)
            raw = (p - event) ** 2
            cal = (calibrated_cv(p, event, month) - event) ** 2
            bs[lab] = {"raw": raw, "calibrated": cal}
            row["models"][lab] = {"forecast_freq": float(p.mean()),
                                  "bss_raw": float(1 - raw.mean() / ref), "bss_calibrated": float(1 - cal.mean() / ref)}
        for a, b in PAIRS:
            if a not in bs or b not in bs:
                continue
            row["pairs"][f"{a} - {b}"] = {}
            for kind in ("raw", "calibrated"):
                ci = paired_block_bootstrap(bs[a][kind] - bs[b][kind], dates, replicates=replicates)
                base = bs[b][kind].mean()
                row["pairs"][f"{a} - {b}"][kind] = {
                    "brier_difference": ci["mean_difference"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
                    "percent": 100 * ci["mean_difference"] / base,
                    "percent_ci": [100 * ci["ci_lower"] / base, 100 * ci["ci_upper"] / base]}
        out[str(t)] = row
        print(f"\n  > {t:g} mm: {row['events']:,} events ({100 * clim:.2f} %)")
        for lab, m in row["models"].items():
            print(f"    {lab:8s} freq {100 * m['forecast_freq']:.2f} %  BSS raw {m['bss_raw']:.3f}  calibrated {m['bss_calibrated']:.3f}")
        for name, d in row["pairs"].items():
            r, c = d["raw"], d["calibrated"]
            print(f"    {name:18s} raw {r['percent']:+6.1f} % [{r['percent_ci'][0]:+.1f}, {r['percent_ci'][1]:+.1f}]"
                  f"   calibrated {c['percent']:+6.1f} % [{c['percent_ci'][0]:+.1f}, {c['percent_ci'][1]:+.1f}]")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", type=Path, required=True)
    ap.add_argument("--meps-members", required=True)
    ap.add_argument("--obs-dir", type=Path, required=True)
    ap.add_argument("--daily", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    res = {"exploratory": True, "replicates": args.replicates,
           "note": "Brier differences are a minus b; negative favours a. Percent is of b's mean Brier score."}
    print("===== A and B: six-hour precipitation, leads 6 to 36 h, raw and calibrated")
    ens, obs, dates = six_hour_cases([p.expanduser() for p in args.cases], args.meps_members, args.obs_dir.expanduser())
    res["six_hour"] = {"cases": int(obs.size), "thresholds": score(ens, obs, dates, SIX_HOUR, args.replicates)}

    print("\n===== C: daily precipitation at +30 h, the station-days of evaluation/bris_m16_plan.json")
    with np.load(args.daily.expanduser(), allow_pickle=True) as z:
        daily = {lab: z[f"model_{key}_30"] for key, lab in NAMES.items()}
        daily["MEPS15"] = z["model_meps15_30"]
        dobs, ddates = z["obs_30"].astype("float64"), z["dates_30"].astype("datetime64[D]")
    res["daily"] = {"cases": int(dobs.size), "thresholds": score(daily, dobs, ddates, DAILY, args.replicates)}

    args.out.expanduser().write_text(json.dumps(res, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

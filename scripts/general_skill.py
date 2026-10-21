#!/usr/bin/env python3
"""Temperature and wind at the gauges: did weighting precipitation cost anything else?

EXPLORATORY, AND SAID SO. Every result so far is precipitation. The extra loss
term only sees tp, but it changes the whole network, so 2 m temperature and
10 m wind could have got worse without any precipitation score showing it.
This scores both against Frost's hourly observations on the test year.

Two steps, so the slow part runs in parallel, one job per model:

  collect   read one model's forecast files and keep, for every gauge within
            max_distance_km of a cell and every lead 6..36 h, the members and
            the observation at the valid hour. Kelvin to Celsius.
  combine   align the models on (init day, lead, gauge), and report fair CRPS,
            bias and RMSE of the ensemble mean per model, and each model minus
            the comparator with a paired seven-day block bootstrap over init days.

No elevation correction: the observation files carry no station heights. The
same cell is read for every model, so differences between models are not
affected; the absolute temperature bias is, and is reported as such.
Mean sea-level pressure is not scored: there are no pressure observations.

  python scripts/general_skill.py collect --model "tailA2000=~/bris-runs/evaluation/tailA2000/nordic_*.nc" \\
      --obs-dir ~/bris-runs/observations --out ~/bris-runs/general-skill/tailA2000.npz
  python scripts/general_skill.py combine --cases ~/bris-runs/general-skill/*.npz \\
      --comparator control2000 --out ~/bris-runs/general-skill/report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.stations import nearest  # noqa: E402

VARIABLES = {
    # name in the obs file -> (variable in the forecast file, offset to the obs unit)
    "temperature": ("air_temperature_2m", -273.15),
    "wind": ("wind_speed_10m", 0.0),
}
LEADS = [6, 12, 18, 24, 30, 36]


def load_obs(obs_dir: Path, name: str) -> dict:
    with np.load(obs_dir / f"{name}.npz", allow_pickle=False) as f:
        times = np.array([t[:19] for t in f["times"]], dtype="datetime64[h]")
        return {"values": f["values"].astype("float64"), "lat": f["lat"], "lon": f["lon"],
                "times": times, "index": {t: i for i, t in enumerate(times)}}


def collect(args: argparse.Namespace) -> int:
    label, pattern = args.model.split("=", 1)
    files = sorted(glob.glob(str(Path(pattern).expanduser())))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    obs = {name: load_obs(args.obs_dir.expanduser(), name) for name in VARIABLES}
    match: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    out: dict[str, list] = {f"{n}_{k}": [] for n in VARIABLES for k in ("init", "lead", "station", "obs", "ens")}

    for n_file, path in enumerate(files, 1):
        with xr.open_dataset(path) as ds:
            if not match:
                lat = np.asarray(ds["latitude"].values, dtype="float64")
                lon = np.asarray(ds["longitude"].values, dtype="float64")
                for name, o in obs.items():
                    idx, dist = nearest(o["lat"], o["lon"], lat.ravel(), lon.ravel())
                    keep = np.flatnonzero(dist <= args.max_distance_km)
                    rows, cols = np.divmod(idx[keep], lat.shape[1])
                    match[name] = (keep, rows, cols)
            times = np.asarray(ds["time"].values).astype("datetime64[h]")
            init = times[0]
            leads = ((times - init) / np.timedelta64(1, "h")).astype(int)
            for name, (var, offset) in VARIABLES.items():
                keep, rows, cols = match[name]
                da = ds[var]
                for dim in list(da.dims):
                    if dim not in ("time", "ensemble_member", "y", "x"):
                        da = da.isel({dim: 0}, drop=True)
                field = np.asarray(da.transpose("time", "ensemble_member", "y", "x").values)
                pts = field[:, :, rows, cols].astype("float64") + offset  # (time, member, gauge)
                o = obs[name]
                for li, lead in enumerate(leads):
                    if lead not in LEADS:
                        continue
                    t = o["index"].get(init + np.timedelta64(int(lead), "h"))
                    if t is None:
                        continue
                    ob = o["values"][keep, t]
                    ens = pts[li].T  # (gauge, member)
                    ok = np.isfinite(ob) & np.isfinite(ens).all(axis=1)
                    if not ok.any():
                        continue
                    k = int(ok.sum())
                    out[f"{name}_init"].append(np.full(k, init.astype("datetime64[D]")))
                    out[f"{name}_lead"].append(np.full(k, lead, dtype="int16"))
                    out[f"{name}_station"].append(keep[ok].astype("int32"))
                    out[f"{name}_obs"].append(ob[ok])
                    out[f"{name}_ens"].append(ens[ok])
        if n_file % 50 == 0:
            print(f"  {n_file}/{len(files)} files", flush=True)

    flat = {k: np.concatenate(v) for k, v in out.items()}
    flat["label"] = np.array(label)
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.expanduser(), **flat)
    for name in VARIABLES:
        print(f"{label} {name}: {flat[name + '_obs'].size:,} cases, {np.unique(flat[name + '_station']).size} gauges")
    return 0


def fair_crps(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    m = ens.shape[1]
    t1 = np.abs(ens - obs[:, None]).mean(axis=1)
    t2 = np.abs(ens[:, :, None] - ens[:, None, :]).sum(axis=(1, 2)) / (2 * m * (m - 1))
    return t1 - t2


def block_ci(diff: np.ndarray, days: np.ndarray, reps: int, seed: int = 0) -> tuple[float, float]:
    d = days.astype("datetime64[D]").astype(int)
    blk = (d - d.min()) // 7
    ub, inv = np.unique(blk, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    est = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, ub.size, ub.size)
        est[r] = sums[i].sum() / cnts[i].sum()
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def combine(args: argparse.Namespace) -> int:
    files = sorted({p for pattern in args.cases for p in glob.glob(str(Path(pattern).expanduser()))})
    data = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            data[str(z["label"])] = {k: z[k] for k in z.files if k != "label"}
    labels = list(data)
    if args.comparator not in labels:
        raise SystemExit(f"comparator {args.comparator} not among {labels}")
    report = {"exploratory": True, "leads_hours": LEADS, "comparator": args.comparator,
              "elevation_correction": False, "variables": {}}

    for name in VARIABLES:
        keys = {}
        for lab, d in data.items():
            init = d[f"{name}_init"].astype("datetime64[D]").astype(np.int64)
            keys[lab] = (init * 100 + d[f"{name}_lead"]) * 100000 + d[f"{name}_station"]
        common = keys[labels[0]]
        for lab in labels[1:]:
            common = np.intersect1d(common, keys[lab])
        rows = {}
        for lab, d in data.items():
            order = np.argsort(keys[lab])
            pos = order[np.searchsorted(keys[lab], common, sorter=order)]
            rows[lab] = {k: d[f"{name}_{k}"][pos] for k in ("init", "lead", "obs", "ens")}
        obs = rows[labels[0]]["obs"]
        days = rows[labels[0]]["init"]
        lead = rows[labels[0]]["lead"]
        crps = {lab: fair_crps(r["ens"], obs) for lab, r in rows.items()}
        res = {"cases": int(common.size), "models": {}}
        ref = crps[args.comparator]
        print(f"\n===== {name}: {common.size:,} common cases")
        print(f"  {'model':16s} {'CRPS':>7s} {'bias':>7s} {'RMSE':>7s} {'spread':>7s}   CRPS minus {args.comparator}")
        for lab, r in rows.items():
            mean = r["ens"].mean(axis=1)
            err = mean - obs
            entry = {"crps": float(crps[lab].mean()), "bias": float(err.mean()),
                     "rmse": float(np.sqrt((err ** 2).mean())), "spread": float(r["ens"].std(axis=1, ddof=1).mean()),
                     "crps_by_lead": {int(L): float(crps[lab][lead == L].mean()) for L in LEADS}}
            line = f"  {lab:16s} {entry['crps']:7.4f} {entry['bias']:+7.3f} {entry['rmse']:7.4f} {entry['spread']:7.4f}"
            if lab != args.comparator:
                diff = crps[lab] - ref
                lo, hi = block_ci(diff, days, args.replicates)
                entry["crps_minus_comparator"] = {"mean": float(diff.mean()), "ci_lower": lo, "ci_upper": hi,
                                                  "relative": float(diff.mean() / ref.mean())}
                line += f"   {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  ({100 * diff.mean() / ref.mean():+.1f}%)"
            res["models"][lab] = entry
            print(line)
        report["variables"][name] = res

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True, metavar="LABEL=GLOB")
    c.add_argument("--obs-dir", type=Path, required=True)
    c.add_argument("--max-distance-km", type=float, default=5.0)
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("combine")
    m.add_argument("--cases", nargs="+", required=True)
    m.add_argument("--comparator", required=True)
    m.add_argument("--replicates", type=int, default=2000)
    m.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return collect(args) if args.cmd == "collect" else combine(args)


if __name__ == "__main__":
    raise SystemExit(main())

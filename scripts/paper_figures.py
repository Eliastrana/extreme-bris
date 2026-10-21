#!/usr/bin/env python3
"""The verification of Nordhagen et al. (2025), arXiv:2511.23043, on the test year.

That paper introduced Bris CRPS-FFT, the checkpoint every model here is
fine-tuned from, and verified it with four kinds of figure. This reproduces
three of them for our models, so A and Pavo can be read against the paper:

  Fig. 6   fair CRPS against lead time, per variable
  Fig. 7   ensemble spread and ensemble-mean RMSE against lead time
  Fig. 8   Brier skill score against threshold, 6 h precipitation and wind
  Fig. 4   quantile-quantile of forecast against observed, leads 24-36 h

Following the paper: point values by bilinear interpolation from the four
cells around each station, 6 h accumulated precipitation, fair CRPS (epsilon
0), MEPS as the operational reference. Different from the paper, and said so
wherever the numbers are shown:

  - the test year (2025-08-01 .. 2026-09-06), not 2022-06 .. 2023-05
  - 4 members, not 10; fair CRPS corrects for that, the spread uses the
    unbiased (m+1)/m form so it compares with the RMSE of an m-member mean
  - leads 6..36 h only, as the forecasts were made to 36 h
  - every Frost station within 5 km of a cell, not the 254 synoptic ones
  - no lapse-rate correction for temperature: the station files carry no
    heights. Model differences are unaffected, the level is not
  - no MSLP: there are no pressure observations
  - MEPS is the single member we have (its control run), scored as a
    one-member ensemble (CRPS = absolute error); the paper used all 30

Two steps. collect reads one model's files (one job per model, in parallel);
combine aligns everything on (init, lead, station) and writes the figures'
data as JSON.

  python scripts/paper_figures.py collect --model "tailA2000=~/bris-runs/evaluation/tailA2000/nordic_*.nc" \\
      --obs-dir ~/bris-runs/observations --out ~/bris-runs/paper-figures/tailA2000.npz
  python scripts/paper_figures.py combine --cases "~/bris-runs/paper-figures/*.npz" \\
      --meps-dir ~/bris-runs/meps --obs-dir ~/bris-runs/observations --out ~/bris-runs/paper-figures/figures.json
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

import evaluate_experiment as ev  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.stations import nearest  # noqa: E402

LEADS = [6, 12, 18, 24, 30, 36]
# obs file -> (forecast variable, offset to obs units, kind)
VARS = {
    "temperature": ("air_temperature_2m", -273.15, "instant"),
    "wind": ("wind_speed_10m", 0.0, "instant"),
    "precipitation": ("precipitation_amount", 0.0, "step"),
}
BSS_THRESHOLDS = {"precipitation": [0.5, 1, 2, 5, 10, 15, 20], "wind": [5, 8, 10, 12, 15, 18]}
QQ_LEADS = (24, 36)
QQ_PROBS = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
# THE HOURLY PRECIPITATION HAS NO QUALITY CODES, AND IT NEEDS SOME. The first
# run found hourly values up to 177 mm, 640 hours over 50 mm and 854 negative
# ones, which made 6 h sums of up to 945 mm against forecasts under 1 mm and
# put every model's RMSE at 5.6 mm (1.9 without them). An hour outside
# [0, HOURLY_MAX_MM] is treated as missing, so its 6 h sum is dropped. 50 mm in
# one hour is near the Norwegian record; the daily scores elsewhere use Frost's
# own quality codes and are unaffected.
HOURLY_MAX_MM = 50.0


def load_obs(obs_dir: Path, name: str) -> dict:
    with np.load(obs_dir / f"{name}.npz", allow_pickle=False) as f:
        times = np.array([t[:19] for t in f["times"]], dtype="datetime64[h]")
        values = f["values"].astype("float64")
        if name == "precipitation":
            bad = (values < 0) | (values > HOURLY_MAX_MM)
            print(f"  precipitation QC: {int(bad.sum()):,} hourly values outside [0, {HOURLY_MAX_MM:g}] mm set missing")
            values[bad] = np.nan
        return {"values": values, "lat": f["lat"], "lon": f["lon"],
                "stations": f["stations"], "times": times, "index": {t: i for i, t in enumerate(times)}}


def obs_at(o: dict, name: str, valid: np.datetime64, keep: np.ndarray) -> np.ndarray | None:
    """Instantaneous value at the valid hour; for precipitation the six hourly sums ending at it."""
    t = o["index"].get(valid)
    if t is None:
        return None
    if name != "precipitation":
        return o["values"][keep, t]
    if t < 5:
        return None
    block = o["values"][keep, t - 5:t + 1]
    out = block.sum(axis=1)
    out[~np.isfinite(block).all(axis=1)] = np.nan
    return out


def collect(args: argparse.Namespace) -> int:
    label, pattern = args.model.split("=", 1)
    files = sorted(glob.glob(str(Path(pattern).expanduser())))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    obs = {n: load_obs(args.obs_dir.expanduser(), n) for n in VARS}
    corners: dict[str, tuple[np.ndarray, list]] = {}
    out: dict[str, list] = {f"{n}_{k}": [] for n in VARS for k in ("init", "lead", "station", "obs", "ens")}
    for n_file, path in enumerate(files, 1):
        with xr.open_dataset(path) as ds:
            if not corners:
                lat = np.asarray(ds["latitude"].values, dtype="float64")
                lon = np.asarray(ds["longitude"].values, dtype="float64")
                for n, o in obs.items():
                    _, dist = nearest(o["lat"], o["lon"], lat.ravel(), lon.ravel())
                    keep = np.flatnonzero(dist <= args.max_distance_km)
                    corners[n] = (keep, ev.bilinear_corners(ds, o["lat"][keep], o["lon"][keep], Path(path)))
            times = np.asarray(ds["time"].values).astype("datetime64[h]")
            init = times[0]
            leads = ((times - init) / np.timedelta64(1, "h")).astype(int)
            for n, (var, offset, kind) in VARS.items():
                keep, cs = corners[n]
                da = ds[var]
                for dim in list(da.dims):
                    if dim not in ("time", "ensemble_member", "y", "x"):
                        da = da.isel({dim: 0}, drop=True)
                field = np.asarray(da.transpose("time", "ensemble_member", "y", "x").values)
                pts = sum(w * field[:, :, r, c].astype("float64") for r, c, w in cs) + offset
                for li, lead in enumerate(leads):
                    if lead not in LEADS:
                        continue
                    ob = obs_at(obs[n], n, init + np.timedelta64(int(lead), "h"), keep)
                    if ob is None:
                        continue
                    ens = pts[li].T
                    if kind == "step":
                        ens = np.clip(ens, 0.0, None)
                    ok = np.isfinite(ob) & np.isfinite(ens).all(axis=1)
                    k = int(ok.sum())
                    if not k:
                        continue
                    out[f"{n}_init"].append(np.full(k, init.astype("datetime64[D]")))
                    out[f"{n}_lead"].append(np.full(k, lead, dtype="int16"))
                    out[f"{n}_station"].append(keep[ok].astype("int32"))
                    out[f"{n}_obs"].append(ob[ok])
                    out[f"{n}_ens"].append(ens[ok])
        if n_file % 50 == 0:
            print(f"  {n_file}/{len(files)} files", flush=True)
    flat = {k: np.concatenate(v) for k, v in out.items()}
    flat["label"] = np.array(label)
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.expanduser(), **flat)
    for n in VARS:
        print(f"{label} {n}: {flat[n + '_obs'].size:,} cases")
    return 0


def meps_cases(meps_dir: Path, obs: dict, name: str) -> dict:
    """MEPS station values as a one-member ensemble, keyed like the model cases."""
    with np.load(meps_dir / f"{name}.npz", allow_pickle=False) as f:
        # Precipitation here is already per 6 h step ("differenced from run
        # start" by fetch_meps_forecasts.py), as MepsReader reads it.
        vals, stations, cycles, leads = f["values"].astype("float64"), f["stations"], f["cycles"], f["leads"]
    pos = {s: i for i, s in enumerate(obs["stations"])}
    idx = np.array([pos.get(s, -1) for s in stations])
    rows = {"init": [], "lead": [], "station": [], "ens": []}
    lead_pos = {int(L): i for i, L in enumerate(leads)}
    for ci, cyc in enumerate(cycles):
        init = np.datetime64(cyc[:13], "h")
        if init.astype("datetime64[D]") != init or int(str(init)[11:13]) != 0:
            continue
        for L in LEADS:
            v = vals[:, ci, lead_pos[L]]
            if name == "temperature" and np.nanmedian(v) > 100:
                v = v - 273.15
            ok = np.isfinite(v) & (idx >= 0)
            rows["init"].append(np.full(int(ok.sum()), init.astype("datetime64[D]")))
            rows["lead"].append(np.full(int(ok.sum()), L, dtype="int16"))
            rows["station"].append(idx[ok].astype("int32"))
            rows["ens"].append(np.clip(v[ok], 0.0, None)[:, None] if name == "precipitation" else v[ok][:, None])
    return {k: np.concatenate(v) for k, v in rows.items()}


def fair_crps(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    m = ens.shape[1]
    t1 = np.abs(ens - obs[:, None]).mean(axis=1)
    if m == 1:
        return t1
    t2 = np.abs(ens[:, :, None] - ens[:, None, :]).sum(axis=(1, 2)) / (2 * m * (m - 1))
    return t1 - t2


def key(init: np.ndarray, lead: np.ndarray, station: np.ndarray) -> np.ndarray:
    return (init.astype("datetime64[D]").astype(np.int64) * 100 + lead) * 100000 + station


def combine(args: argparse.Namespace) -> int:
    files = sorted({p for pat in args.cases for p in glob.glob(str(Path(pat).expanduser()))})
    data = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            data[str(z["label"])] = {k: z[k] for k in z.files if k != "label"}
    obs_dirs = {n: load_obs(args.obs_dir.expanduser(), n) for n in VARS}
    out = {"exploratory": True, "reference": "Nordhagen et al. (2025), arXiv:2511.23043",
           "leads_hours": LEADS, "variables": {}}
    for n in VARS:
        sets = {lab: {k: d[f"{n}_{k}"] for k in ("init", "lead", "station", "obs", "ens")} for lab, d in data.items()}
        if args.meps_dir:
            sets["MEPS"] = meps_cases(args.meps_dir.expanduser(), obs_dirs[n], n)
        keys = {lab: key(s["init"], s["lead"], s["station"]) for lab, s in sets.items()}
        common = None
        for k in keys.values():
            common = k if common is None else np.intersect1d(common, k)
        ref = next(lab for lab in sets if lab != "MEPS")
        aligned = {}
        for lab, s in sets.items():
            order = np.argsort(keys[lab])
            p = order[np.searchsorted(keys[lab], common, sorter=order)]
            aligned[lab] = {k: v[p] for k, v in s.items() if k != "obs"}
            if "obs" in s:
                aligned[lab]["obs"] = s["obs"][p]
        obs = aligned[ref]["obs"]
        lead = aligned[ref]["lead"]
        if n == "precipitation":
            # Recomputed from the QC'd hourly values, so caches collected
            # before the QC need not be read again.
            o = obs_dirs[n]
            valid = (aligned[ref]["init"].astype("datetime64[h]") + lead.astype("timedelta64[h]"))
            tix = np.array([o["index"].get(v, -1) for v in valid])
            fresh = np.full(obs.shape, np.nan)
            okt = tix >= 5
            st = aligned[ref]["station"][okt]
            blocks = np.stack([o["values"][st, tix[okt] - k] for k in range(6)], axis=1)
            fresh[okt] = np.where(np.isfinite(blocks).all(axis=1), blocks.sum(axis=1), np.nan)
            keep = np.isfinite(fresh)
            print(f"  precipitation: {int((~keep).sum()):,} of {keep.size:,} cases dropped by QC")
            aligned = {lab: {k: v[keep] for k, v in a.items()} for lab, a in aligned.items()}
            obs, lead = fresh[keep], lead[keep]
            common = common[keep]
        res = {"cases": int(common.size), "stations": int(np.unique(aligned[ref]["station"]).size), "models": {}}
        print(f"\n===== {n}: {common.size:,} common cases, {res['stations']} stations")
        for lab, a in aligned.items():
            e = a["ens"]
            m = e.shape[1]
            c = fair_crps(e, obs)
            mean = e.mean(axis=1)
            var = e.var(axis=1, ddof=1) if m > 1 else np.zeros_like(mean)
            entry = {"members": int(m), "crps": {}, "rmse": {}, "spread": {}}
            for L in LEADS:
                s = lead == L
                entry["crps"][L] = float(c[s].mean())
                entry["rmse"][L] = float(np.sqrt(((mean[s] - obs[s]) ** 2).mean()))
                entry["spread"][L] = float(np.sqrt(var[s].mean() * (m + 1) / m)) if m > 1 else None
            if n in BSS_THRESHOLDS:
                entry["bss"] = {}
                for t in BSS_THRESHOLDS[n]:
                    ev_ = (obs > t).astype(float)
                    clim = ev_.mean()
                    bs_ref = ((clim - ev_) ** 2).mean()
                    p = (e > t).mean(axis=1)
                    entry["bss"][t] = float(1 - ((p - ev_) ** 2).mean() / bs_ref) if bs_ref > 0 else None
            if n in ("precipitation", "wind"):
                s = (lead >= QQ_LEADS[0]) & (lead <= QQ_LEADS[1])
                entry["qq"] = {"probs": QQ_PROBS, "forecast": [float(np.quantile(e[s].ravel(), q)) for q in QQ_PROBS]}
            res["models"][lab] = entry
            print(f"  {lab:14s} m={m:2d}  CRPS " + " ".join(f"{entry['crps'][L]:.3f}" for L in LEADS))
        if n in ("precipitation", "wind"):
            s = (lead >= QQ_LEADS[0]) & (lead <= QQ_LEADS[1])
            res["qq_observed"] = [float(np.quantile(obs[s], q)) for q in QQ_PROBS]
        out["variables"][n] = res
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(out, indent=2) + "\n")
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
    m.add_argument("--obs-dir", type=Path, required=True)
    m.add_argument("--meps-dir", type=Path, default=None)
    m.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return collect(args) if args.cmd == "collect" else combine(args)


if __name__ == "__main__":
    raise SystemExit(main())

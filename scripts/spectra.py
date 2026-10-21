#!/usr/bin/env python3
"""Have the fine-tuned models lost small-scale detail? DCT power spectra against the MEPS analysis.

EXPLORATORY. Nordhagen et al. (2025, arXiv:2511.23043) found the spectral
loss term necessary for spatially coherent fields: without it, a CRPS-trained
model drifts towards smooth, blurred fields. That term (CRPSFFT) is missing
from every fine-tuning here, because anemoi-training 0.5.1 has no such class.
Station scores cannot see blurring. This measures it, as the paper's Fig. 5
does: the DCT power spectrum of a field over the 2.5 km domain, per member,
against the MEPS analysis at the same valid time.

Spectrum: 2-D DCT-II of the anomaly (field minus its mean, orthonormal), the
variance of each coefficient assigned to the normalised wavenumber
alpha = sqrt((m/Ny)^2 + (n/Nx)^2), wavelength 2*dx/alpha (Denis et al. 2002).
Variance is summed in wavelength bands and averaged over members and days.
A ratio under 1 in the small-scale bands is a model with too little detail.

The forecast files are the MEPS grid with 50 cells trimmed from each edge;
the analysis is read from the year-1 dataset, reshaped to its field shape and
trimmed the same way (checked: latitudes and longitudes match exactly).

  python scripts/spectra.py --model "control2000=~/bris-runs/evaluation/control2000" \\
      --model "tailA2000=~/bris-runs/evaluation/tailA2000" \\
      --zarr ~/bris-data/meps-2p5km-year1-6h-v1.zarr --every 12 --out ~/bris-runs/spectra/spectra.json
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
import xarray as xr  # noqa: E402
import zarr  # noqa: E402
from scipy.fft import dctn  # noqa: E402

DX_KM = 2.5
TRIM = 50
BANDS_KM = [5, 10, 20, 50, 100, 200, 500, 5000]
FIELDS = {
    # name: (forecast variable, analysis recipe)
    "wind": ("wind_speed_10m", ("10u", "10v")),
    "temperature": ("air_temperature_2m", ("2t",)),
    "precipitation": ("precipitation_amount", ("tp",)),
}


class Spectrum:
    """Band index of every DCT coefficient for one grid shape, built once."""

    def __init__(self, ny: int, nx: int) -> None:
        m = np.arange(ny)[:, None] / ny
        n = np.arange(nx)[None, :] / nx
        alpha = np.sqrt(m ** 2 + n ** 2)
        alpha[0, 0] = np.nan  # the mean, removed anyway
        wavelength = 2 * DX_KM / alpha
        self.band = np.digitize(wavelength, BANDS_KM) - 1  # 0 .. len-2 inside the bands
        self.band[~np.isfinite(wavelength)] = -1
        self.nb = len(BANDS_KM) - 1

    def __call__(self, field: np.ndarray) -> np.ndarray:
        f = np.nan_to_num(field - np.nanmean(field))
        var = dctn(f, norm="ortho") ** 2 / f.size
        out = np.zeros(self.nb)
        ok = (self.band >= 0) & (self.band < self.nb)
        np.add.at(out, self.band[ok], var[ok])
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--zarr", type=Path, required=True)
    ap.add_argument("--start", default="2025-09-10")
    ap.add_argument("--end", default="2026-09-04")
    ap.add_argument("--every", type=int, default=12, help="every Nth day")
    ap.add_argument("--leads", type=int, nargs="+", default=[6, 36])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    g = zarr.open(str(args.zarr.expanduser()), "r")
    names = list(g.attrs["variables"])
    fy, fx = g.attrs["field_shape"]
    zdates = np.array(g["dates"][:]).astype("datetime64[h]")
    zindex = {d: i for i, d in enumerate(zdates)}
    data = g["data"]

    def analysis(name: str, valid: np.datetime64) -> np.ndarray | None:
        i = zindex.get(valid)
        if i is None:
            return None
        parts = [np.asarray(data[i, names.index(v), 0, :], dtype="float64").reshape(fy, fx)[TRIM:-TRIM, TRIM:-TRIM]
                 for v in FIELDS[name][1]]
        return np.sqrt(parts[0] ** 2 + parts[1] ** 2) if name == "wind" else parts[0]

    specs = dict(item.split("=", 1) for item in args.model)
    days = np.arange(np.datetime64(args.start), np.datetime64(args.end) + 1, args.every)
    spec = None
    acc = {lab: {f: {L: [] for L in args.leads} for f in FIELDS} for lab in [*specs, "MEPS-analyse"]}
    used = 0
    for day in days:
        files = {lab: Path(d).expanduser() / f"nordic_bris_{lab}_{day}T00Z.nc" for lab, d in specs.items()}
        if not all(p.exists() for p in files.values()):
            continue
        opened = {lab: xr.open_dataset(p) for lab, p in files.items()}
        try:
            for L in args.leads:
                valid = day.astype("datetime64[h]") + np.timedelta64(L, "h")
                for f, (var, _) in FIELDS.items():
                    truth = analysis(f, valid)
                    if truth is None:
                        continue
                    if spec is None:
                        spec = Spectrum(*truth.shape)
                    rows = {}
                    for lab, ds in opened.items():
                        da = ds[var]
                        for dim in list(da.dims):
                            if dim not in ("time", "ensemble_member", "y", "x"):
                                da = da.isel({dim: 0}, drop=True)
                        times = np.asarray(ds["time"].values).astype("datetime64[h]")
                        ti = int(np.flatnonzero(times == valid)[0])
                        members = np.asarray(da.isel(time=ti).transpose("ensemble_member", "y", "x").values, dtype="float64")
                        rows[lab] = np.mean([spec(mm) for mm in members], axis=0)
                    for lab, r in rows.items():
                        acc[lab][f][L].append(r)
                    acc["MEPS-analyse"][f][L].append(spec(truth))
            used += 1
        finally:
            for ds in opened.values():
                ds.close()
        print(f"  {day}: done ({used} days)", flush=True)

    out = {"exploratory": True, "reference": "Nordhagen et al. (2025), Fig. 5", "days": used,
           "every_nth_day": args.every, "bands_km": BANDS_KM, "fields": {}}
    for f in FIELDS:
        out["fields"][f] = {}
        for L in args.leads:
            ref = np.mean(acc["MEPS-analyse"][f][L], axis=0)
            entry = {"MEPS-analyse": ref.tolist()}
            print(f"\n== {f}, +{L} h: variance relative to the MEPS analysis, by band (km)")
            print("   " + " ".join(f"{BANDS_KM[i]}-{BANDS_KM[i + 1]:>4}" for i in range(len(BANDS_KM) - 1)))
            for lab in specs:
                s = np.mean(acc[lab][f][L], axis=0)
                entry[lab] = s.tolist()
                print(f"   {lab:12s} " + " ".join(f"{r:8.2f}" for r in s / ref))
            out["fields"][f][L] = entry
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

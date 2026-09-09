#!/usr/bin/env python3
"""Apply the physical conversions anemoi's filters cannot, to a built dataset.

Every conversion filter in anemoi-datasets 0.5.24 reads
`metadata(namespace="mars")["param"]`, and xarray-sourced fields return an empty
mars namespace, so all of them raise KeyError on NetCDF or OPeNDAP input. Only
`rename` works. The recipe therefore renames columns to their target names while
they still hold the source quantity, and this converts the values in place.

Three conversions, each of which silently produces plausible-looking output if
skipped:

  winds     x_wind/y_wind are aligned with the Lambert grid; the model expects
            earth-relative u/v
  2d        holds relative humidity; must become dewpoint in kelvin. The
            OPeNDAP source serves it as a FRACTION, not percent, so the scale
            is detected from the data rather than assumed. Feeding 0.82 to a
            formula expecting percent reads it as 0.82% humidity and returns a
            dewpoint tens of degrees too low, which is plausible-looking and
            wrong.
  w         holds m/s; ECMWF w is Pa/s, which differs by -rho*g, an order of
            magnitude and a sign

    scripts/postprocess_meps.py <dataset> --dry-run
    sbatch bris/slurm/postprocess.sbatch <dataset>

ONE PASS PER STATE, NOT ONE PER VARIABLE. The data is chunked as one state by
all 98 variables, roughly 400 MB compressed as a single blob. Zarr cannot write
part of a chunk, so assigning one variable of one state decompresses the whole
chunk, patches four megabytes of it, and recompresses the whole thing. The
first version of this script looped over variables and then over states, which
did that forty times per state and ran at a few megabytes a second: two
variables in one hour and forty-five minutes, or well over a day per file.

So the loops are the other way round. Each state is read once, every conversion
is applied to it in memory, and it is written once. Same arithmetic, one
fortieth of the compression work.

IT RECORDS PROGRESS AND CAN RESUME. Converting a variable twice rotates twice
or subtracts a constant twice, and on a quarter of a terabyte with no backup
there is no way back. Progress is written to the dataset attributes after every
state, so a job that hits its walltime can simply be resubmitted. Where the
humidity column is among the pending conversions it also serves as a per-state
witness: relative humidity and a dewpoint in kelvin are not confusable, so the
narrow window between writing a state and recording it cannot cause a state to
be converted twice.

IT REWRITES THE STATISTICS. An earlier version changed the data and left the
mean, stdev, minimum and maximum arrays describing the values that used to be
there. Normalisation reads those arrays, not the data, so a dataset could be
converted correctly and still be normalised as though it had not been, with
nothing on screen to say so.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv

_venv.ensure("zarr", "numpy")

import numpy as np  # noqa: E402

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000)
WIND_PAIRS = [("10u", "10v")] + [(f"u_{l}", f"v_{l}") for l in LEVELS]
MARKER = "bris_postprocessed"
PROGRESS = "bris_postprocess_done"
PARTIAL = "bris_postprocess_partial"

R_D = 287.05        # J/(kg K)
G = 9.80665         # m/s2


class Stats:
    """Running mean, spread and extremes for the variables this script edits.

    Accumulated while the conversions stream through, so no extra pass over a
    quarter of a terabyte is needed. Only touched variables are rewritten; the
    rest of the statistics arrays are left exactly as built.

    It serialises, because a resumed run must not start its averages over.
    """

    def __init__(self) -> None:
        self.acc: dict[int, list] = {}

    def note(self, index: int, values) -> None:
        v = np.asarray(values, dtype="float64")
        good = np.isfinite(v)
        if not good.any():
            return
        v = v[good]
        slot = self.acc.get(index)
        if slot is None:
            self.acc[index] = [int(v.size), float(v.sum()), float((v ** 2).sum()),
                               float(v.min()), float(v.max())]
            return
        slot[0] += int(v.size)
        slot[1] += float(v.sum())
        slot[2] += float((v ** 2).sum())
        slot[3] = min(slot[3], float(v.min()))
        slot[4] = max(slot[4], float(v.max()))

    def dump(self) -> str:
        return json.dumps({str(k): v for k, v in self.acc.items()})

    def load(self, blob: str) -> None:
        self.acc = {int(k): list(v) for k, v in json.loads(blob).items()}

    def write(self, z, names: list[str]) -> None:
        for index in sorted(self.acc):
            n, total, squares, lo, hi = self.acc[index]
            mean = total / n
            var = max(squares / n - mean * mean, 0.0)
            new = {"mean": mean, "stdev": var ** 0.5, "minimum": lo, "maximum": hi}
            for key, value in new.items():
                if key in z:
                    z[key][index] = value
            # anemoi keeps the raw sums alongside the derived values in some
            # builds; leaving them stale would make any recomputation disagree.
            for key, value in (("sums", total), ("squares", squares)):
                if key in z:
                    z[key][index] = value
            print(f"  {names[index]:8s} mean {mean:13.6g}  stdev {new['stdev']:12.6g}"
                  f"  range {lo:.6g} .. {hi:.6g}")


def rotation_angle(lat, lon, nx, ny):
    """Angle from grid north to true north, per point, in radians."""
    lat2 = np.deg2rad(np.asarray(lat).reshape(ny, nx))
    lon2 = np.deg2rad(np.asarray(lon).reshape(ny, nx))
    dlat = np.gradient(lat2, axis=0)
    dlon = np.gradient(lon2, axis=0)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    return np.arctan2(dlon * np.cos(lat2), dlat).reshape(-1)


def rh_scale(sample_max: float) -> float:
    """Factor turning the stored relative humidity into percent.

    MET's OPeNDAP relative_humidity_2m comes as a fraction in [0, 1]; other
    sources use percent. The two are a factor of a hundred apart and there is
    no overlap between a plausible fraction and a plausible percentage, so the
    maximum decides. Anything outside both ranges stops the run rather than
    picking the nearer one.
    """
    if sample_max <= 1.5:
        return 100.0
    if sample_max <= 120.0:
        return 1.0
    raise SystemExit(
        f"relative humidity tops out at {sample_max:g}, which is neither a "
        "fraction nor a percentage. Look at the field before converting it."
    )


def rh_to_dewpoint(rh_pct, t_kelvin):
    """Magnus formula. rh in percent, t in kelvin, result in kelvin."""
    a, b = 17.625, 243.04
    t_c = np.asarray(t_kelvin, dtype="float64") - 273.15
    rh = np.clip(np.asarray(rh_pct, dtype="float64"), 1e-3, 100.0)
    alpha = np.log(rh / 100.0) + (a * t_c) / (b + t_c)
    return (b * alpha) / (a - alpha) + 273.15


def wz_to_omega(wz, t_kelvin, pressure_pa):
    """Geometric vertical velocity (m/s) to omega (Pa/s): w = -rho g wz."""
    rho = pressure_pa / (R_D * np.asarray(t_kelvin, dtype="float64"))
    return -rho * G * np.asarray(wz, dtype="float64")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--nx", type=int, default=949)
    ap.add_argument("--ny", type=int, default=1069)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if already marked")
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args()

    import zarr

    z = zarr.open(str(args.dataset), mode="r" if args.dry_run else "r+")
    if z.attrs.get(MARKER) and not args.force:
        print(f"ERROR: already post-processed ({z.attrs[MARKER]}).", file=sys.stderr)
        print("Converting twice is worse than not at all. Use --force only if "
              "you know the previous run did not complete.", file=sys.stderr)
        return 1

    names = list(z.attrs["variables"])
    idx = {n: i for i, n in enumerate(names)}
    data = z["data"]
    nt = data.shape[0]

    # ---- what is left to do -------------------------------------------------
    done = set(z.attrs.get(PROGRESS, []))
    pairs = [(u, v) for u, v in WIND_PAIRS
             if u in idx and v in idx and not (u in done and v in done)]
    do_2d = "2d" in idx and "2t" in idx and "2d" not in done
    levels = [l for l in LEVELS
              if f"w_{l}" in idx and f"t_{l}" in idx and f"w_{l}" not in done]

    pending = [n for uv in pairs for n in uv]
    pending += ["2d"] if do_2d else []
    pending += [f"w_{l}" for l in levels]
    if not pending:
        print("Nothing left to convert.")
        return 0
    if done:
        print(f"{len(done)} variable(s) already converted: {', '.join(sorted(done))}")
    print(f"{len(pending)} to go: {', '.join(pending)}\n")

    # ---- resume position ----------------------------------------------------
    stats = Stats()
    first_state = 0
    partial = z.attrs.get(PARTIAL)
    if partial:
        if sorted(partial["variables"]) != sorted(pending):
            print("ERROR: the interrupted run was converting a different set of "
                  "variables than this one would.\n"
                  f"  interrupted: {', '.join(sorted(partial['variables']))}\n"
                  f"  this run   : {', '.join(sorted(pending))}\n"
                  "Resuming would leave the two sets converted over different "
                  "ranges of states, which nothing downstream could detect.",
                  file=sys.stderr)
            return 1
        first_state = int(partial["states_done"])
        stats.load(partial["stats"])
        print(f"resuming at state {first_state} of {nt}\n")

    # ---- the rotation, computed once ----------------------------------------
    lat, lon = np.asarray(z["latitudes"]), np.asarray(z["longitudes"])
    if lat.size != args.nx * args.ny:
        print(f"ERROR: {lat.size:,} points is not {args.nx}x{args.ny}", file=sys.stderr)
        return 1
    ang = rotation_angle(lat, lon, args.nx, args.ny)
    print(f"rotation angle: {np.degrees(ang.min()):.2f} .. {np.degrees(ang.max()):.2f} deg")
    print("  a Lambert grid over the Nordics should span roughly -30 to +30;")
    print("  a near-zero range means the angle was not recovered.")
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    # ---- the humidity scale, read from the data, not the statistics ---------
    scale = 1.0
    if do_2d:
        sample = np.asarray(data[first_state, idx["2d"], 0, :], dtype="float64")
        scale = rh_scale(float(np.nanmax(sample)))
        print(f"\nhumidity: maximum {np.nanmax(sample):g}, so it holds "
              f"{'a fraction' if scale == 100.0 else 'percent'}; "
              f"multiplying by {scale:g}")

    states = range(first_state, first_state + 1) if args.dry_run \
        else range(first_state, nt)
    above = 0
    drift = 0.0

    print(f"\n=== converting {len(states)} state(s)")
    started = datetime.datetime.now()
    for si, t in enumerate(states):
        block = data[t]                       # (nvar, 1, npoints), one chunk
        get = lambda n: np.asarray(block[idx[n], 0, :], dtype="float64")  # noqa: E731

        # A dewpoint in kelvin and a relative humidity cannot be mistaken for
        # each other, so where 2d is pending it says whether this state has
        # already been through. That closes the gap between writing a state
        # and recording it.
        if do_2d and float(np.nanmax(get("2d"))) > 1.5:
            print(f"  state {t} already converted; skipping")
            continue

        for un, vn in pairs:
            u, v = get(un), get(vn)
            ue, ve = u * cos_a - v * sin_a, u * sin_a + v * cos_a
            drift = max(drift, float(np.abs(np.hypot(u, v) - np.hypot(ue, ve)).max()))
            block[idx[un], 0, :] = ue.astype(block.dtype)
            block[idx[vn], 0, :] = ve.astype(block.dtype)
            stats.note(idx[un], ue)
            stats.note(idx[vn], ve)

        if do_2d:
            t2 = get("2t")
            td = rh_to_dewpoint(get("2d") * scale, t2)
            above += int(np.nansum(td > t2 + 0.5))
            block[idx["2d"], 0, :] = td.astype(block.dtype)
            stats.note(idx["2d"], td)

        for lev in levels:
            om = wz_to_omega(get(f"w_{lev}"), get(f"t_{lev}"), lev * 100.0)
            block[idx[f"w_{lev}"], 0, :] = om.astype(block.dtype)
            stats.note(idx[f"w_{lev}"], om)

        if not args.dry_run:
            data[t] = block
            z.attrs[PARTIAL] = {"variables": pending, "states_done": t + 1,
                                "stats": stats.dump()}

        if si % args.report_every == 0 or t == nt - 1:
            per = (datetime.datetime.now() - started).total_seconds() / (si + 1)
            left = datetime.timedelta(seconds=int(per * (len(states) - si - 1)))
            print(f"  state {t + 1}/{nt}  {per:.2f} s each  about {left} left")

    print(f"\nwind speed drift {drift:.2e}, must be ~0")
    if do_2d:
        print(f"{above} points ended with a dewpoint above their temperature")
        if above:
            print("  that is unphysical; a handful from rounding is expected, a "
                  "large share means the humidity scale was read wrong")

    if args.dry_run:
        print("\nDry run on one state. Nothing written.")
        return 0

    print("\n=== rewriting statistics for the variables that changed")
    stats.write(z, names)
    z.attrs[PROGRESS] = sorted(done | set(pending))
    del z.attrs[PARTIAL]
    z.attrs[MARKER] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("\nmarked as post-processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

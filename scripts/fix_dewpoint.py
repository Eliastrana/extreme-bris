#!/usr/bin/env python3
"""Repair 2d in a dataset post-processed before the humidity scale was fixed.

    scripts/fix_dewpoint.py ~/bris-data/meps-2p5km-2025-6h-v1.zarr --dry-run
    scripts/fix_dewpoint.py ~/bris-data/meps-2p5km-2025-6h-v1.zarr

WHAT WENT WRONG. postprocess_meps.py converted relative humidity to dewpoint
with a formula that expects percent. MET's OPeNDAP source serves it as a
fraction, so a humidity of 0.79 was read as 0.79 percent. The result is a
dewpoint tens of degrees too cold, written over the original values, and
marked as post-processed.

Nothing about the output looks wrong from a distance. It is in kelvin, it is
finite everywhere, it is smooth, and it is below the temperature as a dewpoint
must be. It is only wrong by about fifty degrees.

WHY THE ORIGINAL DATA IS NOT NEEDED. The Magnus formula inverts, and the bug is
a constant in the one place it matters. Writing the intermediate as

    alpha = ln(rh/100) + a*T / (b + T)          T in celsius

the conversion is td = b*alpha / (a - alpha). Reading a fraction f as a
percentage computes ln(f/100) where it should compute ln(f), and those differ
by exactly ln(100) whatever f is. So recovering alpha from the stored dewpoint,
adding ln(100), and converting again gives the dewpoint that should have been
written. No re-download, no rebuild, and no dependence on the temperature field
having stayed put.

The one place this is not exact is where the original humidity was clipped at
the formula's lower bound. As a fraction that bound is 0.001 percent, which is
drier than the Nordic surface ever gets, so in practice it does not bite.

THE STATISTICS ARE REWRITTEN TOO. The version of postprocess_meps.py that made
this mess also left the statistics describing the humidity that used to be in
the column, which is why the stored maximum still reads 1 on these datasets and
why the damage was invisible to every check that reads statistics.
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv

_venv.ensure("zarr", "numpy")

import numpy as np  # noqa: E402

from postprocess_meps import Stats  # noqa: E402

A, B = 17.625, 243.04
LN100 = math.log(100.0)
MARKER = "bris_dewpoint_repaired"

# Implied humidity below this, in percent, means the stored dewpoint was
# produced by the bug. The Nordic surface does not sit at a few percent
# humidity across a whole field; the bug puts it there by construction.
BUG_RH_MAX = 5.0


def alpha_from_dewpoint(td_kelvin: np.ndarray) -> np.ndarray:
    td_c = np.asarray(td_kelvin, dtype="float64") - 273.15
    return A * td_c / (B + td_c)


def dewpoint_from_alpha(alpha: np.ndarray) -> np.ndarray:
    return B * alpha / (A - alpha) + 273.15


def implied_rh_percent(td_kelvin, t_kelvin) -> np.ndarray:
    """What humidity the stored dewpoint corresponds to, read correctly."""
    t_c = np.asarray(t_kelvin, dtype="float64") - 273.15
    alpha = alpha_from_dewpoint(td_kelvin)
    return 100.0 * np.exp(alpha - A * t_c / (B + t_c))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="repair even if the humidity does not look impossible")
    args = ap.parse_args()

    import zarr

    z = zarr.open(str(args.dataset), mode="r" if args.dry_run else "r+")
    if z.attrs.get(MARKER) and not args.force:
        print(f"ERROR: already repaired ({z.attrs[MARKER]}). Adding ln(100) "
              "twice is a hundredfold error in the other direction.",
              file=sys.stderr)
        return 1

    names = list(z.attrs["variables"])
    if "2d" not in names or "2t" not in names:
        print("ERROR: 2d or 2t missing from this dataset", file=sys.stderr)
        return 1
    data = z["data"]
    i2d, i2t = names.index("2d"), names.index("2t")
    nt = data.shape[0]

    # ---- diagnose before touching anything ---------------------------------
    td0 = np.asarray(data[0, i2d, 0, ::997], dtype="float64")
    t0 = np.asarray(data[0, i2t, 0, ::997], dtype="float64")
    if np.nanmax(td0) <= 1.5:
        print("This dataset still holds RELATIVE HUMIDITY in 2d, not a dewpoint.\n"
              "It has not been post-processed at all, so there is nothing here to\n"
              "repair. Run postprocess_meps.py instead, which now reads the scale\n"
              "off the data.", file=sys.stderr)
        return 1

    rh = implied_rh_percent(td0, t0)
    median_rh = float(np.nanmedian(rh))
    depression = float(np.nanmedian(t0 - td0))
    print(f"=== {args.dataset.name}")
    print(f"  stored dewpoint      {np.nanmin(td0):.2f} .. {np.nanmax(td0):.2f} K")
    print(f"  temperature          {np.nanmin(t0):.2f} .. {np.nanmax(t0):.2f} K")
    print(f"  median depression    {depression:.2f} K")
    print(f"  implied humidity     {median_rh:.3f} %")

    if median_rh > BUG_RH_MAX and not args.force:
        print(f"\n  That humidity is not impossible, so this dataset does not carry\n"
              "  the signature of the bug. Refusing. Use --force only if you have\n"
              "  another reason to believe it does.", file=sys.stderr)
        return 1

    corrected = dewpoint_from_alpha(alpha_from_dewpoint(td0) + LN100)
    print(f"\n  after repair         {np.nanmin(corrected):.2f} .. "
          f"{np.nanmax(corrected):.2f} K")
    print(f"  median depression    {float(np.nanmedian(t0 - corrected)):.2f} K")
    print(f"  implied humidity     "
          f"{float(np.nanmedian(implied_rh_percent(corrected, t0))):.1f} %")

    if args.dry_run:
        print("\nDry run on the first state only. Nothing written.")
        return 0

    # ---- repair -------------------------------------------------------------
    stats = Stats()
    above = 0
    for t in range(nt):
        td = np.asarray(data[t, i2d, 0, :], dtype="float64")
        fixed = dewpoint_from_alpha(alpha_from_dewpoint(td) + LN100)
        t2 = np.asarray(data[t, i2t, 0, :], dtype="float64")
        above += int(np.nansum(fixed > t2 + 0.5))
        stats.note(i2d, fixed)
        data[t, i2d, 0, :] = fixed.astype(data.dtype)
        if t % 100 == 0:
            print(f"  {t}/{nt}")

    stats.write(z, names)
    z.attrs[MARKER] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\nrepaired {nt} states; {above} points end above their temperature")
    if above:
        print("  a dewpoint above the temperature is unphysical. A handful from\n"
              "  rounding is expected; a large share is not, and means the\n"
              "  diagnosis was wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

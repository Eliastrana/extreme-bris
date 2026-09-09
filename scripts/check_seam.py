#!/usr/bin/env python3
"""Compare the two halves of the cutout variable by variable, and find the rescale.

    scripts/check_seam.py

TWO QUESTIONS, ONE READ.

FIRST: is precipitation the only variable that disagrees? The MEPS recipe maps
MET's names onto ECMWF's and renames without converting. tp was caught because
kg/m2 against metres is a factor of a thousand and impossible to miss. A
variable that differs by a factor of three, or by an additive offset, would not
have announced itself, and every one of them is a seam running through the
middle of the training target. So this prints every variable the two halves
share, side by side, and flags the ones whose scales do not match.

The comparison is between DIFFERENT DOMAINS, so a difference is not proof of a
bug: the tropics really are wetter than Norway, and Norway really is colder
than the global mean. What the flag means is "look at this one", not "this is
wrong". Read it with the physics in mind.

SECOND: which rescale syntax does the installed anemoi-datasets take? The fix
for tp is to scale the LAM half in the dataloader rather than rebuild 750 GB.
That is only worth writing into a config once it is known to work, and the
form has moved between versions, so this tries the candidates and reports
which one composes and whether the statistics follow the data. A rescale that
changes the values and leaves the statistics behind would be worse than none
at all, because normalisation reads the statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv
import _compose

_venv.ensure("anemoi")

# Ratio beyond which two halves get flagged for a look. Wide on purpose: real
# climatological differences between the Nordics and the globe are large, and
# a threshold tight enough to catch every unit error would flag everything.
FLAG_RATIO = 20.0


def stats_of(paths: list[Path]) -> tuple[list[str], dict]:
    """Per-variable statistics, pooled across a half's year files by extremes."""
    import numpy as np
    import zarr

    names: list[str] | None = None
    acc: dict[str, dict] = {}
    for p in paths:
        z = zarr.open(str(p), mode="r")
        v = list(z.attrs["variables"])
        if names is None:
            names = v
        elif v != names:
            raise SystemExit(
                f"{p.name} lists variables in a different order than {paths[0].name}. "
                "Concatenating those would silently mix up channels."
            )
        for key in ("mean", "stdev", "maximum", "minimum"):
            arr = np.asarray(z[key][:], dtype="float64")
            slot = acc.setdefault(key, {"v": arr, "n": 1})
            if key == "maximum":
                slot["v"] = np.maximum(slot["v"], arr)
            elif key == "minimum":
                slot["v"] = np.minimum(slot["v"], arr)
            else:
                slot["v"] = slot["v"] + arr
                slot["n"] += 1
    out = {}
    for key, slot in acc.items():
        out[key] = slot["v"] / slot["n"] if key in ("mean", "stdev") else slot["v"]
    return names or [], out


def compare(cfg) -> int:
    halves = cfg.dataloader.dataset.cutout
    labels = ("LAM", "global")
    collected = []
    for half in halves:
        spec = half["dataset"]
        paths = [Path(p) for p in (spec["concat"] if "concat" in spec else [spec])]
        names, stats = stats_of(paths)
        # A mismatch fixed in the dataloader is fixed. Comparing the files
        # alone would keep flagging it, and a flag that is always on is a flag
        # nobody reads.
        for var, spec_r in (half.get("rescale") or {}).items():
            if var not in names:
                continue
            i = names.index(var)
            scale = float(spec_r.get("scale", 1.0))
            offset = float(spec_r.get("offset", 0.0))
            print(f"  applying the config's rescale to {var}: "
                  f"x{scale:g} {offset:+g}")
            for key in ("mean", "minimum", "maximum"):
                stats[key][i] = stats[key][i] * scale + offset
            stats["stdev"][i] = stats["stdev"][i] * abs(scale)
        collected.append((names, stats))

    (na, sa), (nb, sb) = collected
    shared = [v for v in na if v in nb]
    print(f"=== {len(shared)} variables in both halves\n")
    print(f"{'variable':16s} {'LAM mean':>13s} {'global mean':>13s} "
          f"{'LAM max':>13s} {'global max':>13s} {'ratio':>9s}")

    flagged = []
    for v in shared:
        ia, ib = na.index(v), nb.index(v)
        ma, mb = sa["mean"][ia], sb["mean"][ib]
        xa, xb = sa["maximum"][ia], sb["maximum"][ib]
        # Compare on spread rather than level: an additive offset between two
        # domains is ordinary, a multiplicative one usually is not.
        da, db = sa["stdev"][ia], sb["stdev"][ib]
        ratio = (da / db) if db else float("inf")
        mark = ""
        if ratio and (ratio > FLAG_RATIO or ratio < 1.0 / FLAG_RATIO):
            mark = "  <-- scales differ"
            flagged.append((v, ratio))
        print(f"{v:16s} {ma:13.5g} {mb:13.5g} {xa:13.5g} {xb:13.5g} "
              f"{ratio:9.3g}{mark}")

    print()
    if flagged:
        print(f"{len(flagged)} variable(s) flagged: "
              + ", ".join(f"{v} ({r:.4g}x)" for v, r in flagged))
        print("Check each against the physics before assuming a unit error, and\n"
              "against bris/configs/meps_2p5km.yaml, where MET's names are mapped\n"
              "onto ECMWF's without any conversion.")
    else:
        print("No variable's spread differs by more than "
              f"{FLAG_RATIO:g}x between the halves.")
    return len(flagged)


def probe_rescale(cfg) -> None:
    """Find a rescale spelling this anemoi-datasets accepts, and verify it."""
    from anemoi.datasets import open_dataset

    lam = cfg.dataloader.dataset.cutout[0]["dataset"]
    one = (lam["concat"] if "concat" in lam else [lam])[0]

    base = open_dataset(one)
    i = list(base.variables).index("tp")
    before = float(base.statistics["maximum"][i])
    print(f"\n=== rescale probe, on {Path(str(one)).name}")
    print(f"  tp maximum as built: {before:.6g}")

    candidates = [
        ("scale/offset dict",
         {"dataset": one, "rescale": {"tp": {"scale": 0.001, "offset": 0.0}}}),
        ("scale only",
         {"dataset": one, "rescale": {"tp": {"scale": 0.001}}}),
        ("unit pair",
         {"dataset": one, "rescale": {"tp": ("kg m-2", "m")}}),
        ("unit pair, from/to",
         {"dataset": one, "rescale": {"tp": {"from": "kg m-2", "to": "m"}}}),
    ]

    working = []
    for name, spec in candidates:
        try:
            ds = open_dataset(spec)
            after = float(ds.statistics["maximum"][list(ds.variables).index("tp")])
        except Exception as exc:  # noqa: BLE001 - reporting every failure is the point
            print(f"  {name:22s} REJECTED  {type(exc).__name__}: {exc}"[:200])
            continue
        # The statistics have to move with the data or normalisation reads the
        # old scale and the rescale makes things worse rather than better.
        ok = abs(after - before / 1000.0) < abs(before) * 1e-6
        verdict = "statistics follow" if ok else f"STATISTICS DID NOT FOLLOW ({after:.6g})"
        print(f"  {name:22s} accepted, tp maximum now {after:.6g}  -> {verdict}")
        if ok:
            working.append(name)

    print()
    if working:
        print(f"Use: {working[0]}. Add it beside reorder and trim_edge on the LAM\n"
              "entry of the cutout, in both arm configs, identically.")
    else:
        print("No candidate both composed and carried its statistics. The fix\n"
              "then has to happen at build time: add the conversion to\n"
              "bris/configs/meps_2p5km.yaml and rebuild the three MEPS years.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", type=Path, default=REPO / "bris" / "train")
    ap.add_argument("--config-name", default="finetune")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    cfg = _compose.compose(args.config_dir, args.config_name)
    flagged = compare(cfg)
    if not args.no_probe:
        probe_rescale(cfg)
    return 0 if flagged == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

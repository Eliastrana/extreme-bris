#!/usr/bin/env python3
"""Convert a precipitation threshold in millimetres into the number the loss needs.

    scripts/tail_threshold.py --mm 20
    scripts/tail_threshold.py --mm 20 --config-name finetune

WHY THIS EXISTS. The threshold-weighted term in finetune_tail.yaml clips
precipitation at a threshold, and it sees normalised values, not millimetres.
The conversion depends on the normaliser stored in the checkpoint the run
warm-starts from, not on the dataset's statistics. Both are knowable and neither is
guessable, so this reads them and prints the number to paste in.

Getting it wrong is silent. A threshold set a standard deviation too high
makes the term score nothing at all; too low and it scores everything and
stops being a tail term. Either way the run completes and the numbers look
plausible, which is the worst kind of mistake this project keeps producing.

IT ALSO CHECKS THE UNITS OF THE TWO HALVES, and that check is not incidental.
MEPS carries precipitation as precipitation_amount_acc in kg/m2, which is
millimetres. The IFS side carries tp in metres, as IFS always has. The MEPS
recipe renames the field and does not rescale it. If that is still true when
this runs, the cutout stitches a field whose two halves differ by a factor of
a thousand at the domain boundary, one set of normalisation statistics is
computed across both, and every number downstream is meaningless. Refusing
here costs a second. Finding out afterwards costs the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv
import _compose

_venv.ensure('anemoi')

# Below this, a precipitation field is being stored in metres; above it, in
# millimetres. Six-hour totals of half a metre do not occur and six-hour
# totals under 1 mm at every point over a year do not either, so the gap
# between the two conventions is wide enough to decide from the maximum.
UNIT_CUT = 1.0


def classify_units(sample_max: float) -> str:
    if sample_max <= 0:
        return "empty"
    return "m" if sample_max < UNIT_CUT else "mm"


def declared_rescale(half, variable: str) -> tuple[float, float]:
    """The scale and offset the config applies to this half of the cutout.

    A unit mismatch can be fixed in the built data or declared in the
    dataloader, and this has to judge what the run will actually see. Reading
    the files alone would keep reporting a mismatch that the config already
    corrects, and a warning that is always on is a warning nobody reads.
    """
    spec = (half.get("rescale") or {}).get(variable) or {}
    return float(spec.get("scale", 1.0)), float(spec.get("offset", 0.0))


def half_units(paths: list[Path], scale: float = 1.0,
               offset: float = 0.0) -> list[tuple[str, str, float]]:
    """Read the stored maximum of tp from each zarr and name its units."""
    import zarr

    out = []
    for p in paths:
        z = zarr.open(str(p), mode="r")
        names = list(z.attrs["variables"])
        if "tp" not in names:
            out.append((p.name, "no tp", 0.0))
            continue
        i = names.index("tp")
        mx = float(z["maximum"][i]) * scale + offset
        out.append((p.name, classify_units(mx), mx))
    return out


def model_normaliser(cfg, variable: str) -> tuple[float, float, str]:
    """Scale and offset the warm-started model applies to one variable.

    Read from the checkpoint the run loads, because that is what the loss sees:
    transfer learning copies the normaliser's buffers along with the weights.
    """
    import torch

    path = str(cfg.hardware.files.warm_start)
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(ckpt, dict):
        state, indices = ckpt["state_dict"], ckpt["hyper_parameters"]["data_indices"]
    else:
        state, indices = ckpt.state_dict(), ckpt.data_indices
    idx = indices.name_to_index[variable]
    key_mul = next(k for k in state if k.endswith("pre_processors.processors.normalizer._norm_mul"))
    key_add = next(k for k in state if k.endswith("pre_processors.processors.normalizer._norm_add"))
    return (float(state[key_mul].reshape(-1)[idx]),
            float(state[key_add].reshape(-1)[idx]), path)


def configured_tail_threshold(cfg) -> float | None:
    """The threshold the tail term is configured with, if this config has one."""
    for loss in cfg.training.training_loss.get("losses", []):
        if str(loss.get("_target_", "")).endswith("TailWeightedKernelCRPS"):
            value = loss.get("tail_threshold")
            return None if value is None else float(value)
    return None


def normaliser_for(cfg, variable: str) -> str:
    """Which normalisation method the config assigns to one variable."""
    spec = cfg.data.normalizer
    for method, variables in spec.items():
        if method == "default":
            continue
        if variables and variable in list(variables):
            return str(method)
    return str(spec.get("default", "mean-std"))


def to_normalised(value: float, method: str, stats: dict) -> float:
    """Apply the same affine map the normaliser will apply to the data."""
    if method in ("none", "None"):
        return value
    if method in ("mean-std", "mean_std", "std"):
        return (value - stats["mean"]) / stats["stdev"]
    if method == "max":
        return value / stats["maximum"]
    if method in ("min-max", "min_max"):
        return (value - stats["minimum"]) / (stats["maximum"] - stats["minimum"])
    raise SystemExit(
        f"normaliser {method!r} is not one this script knows how to invert. "
        "Add it here rather than converting by hand."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm", type=float, default=20.0,
                    help="threshold in millimetres per 6 h state (default 20)")
    ap.add_argument("--config-dir", type=Path, default=REPO / "bris" / "train")
    ap.add_argument("--config-name", default="finetune")
    ap.add_argument("--variable", default="tp")
    ap.add_argument("--skip-units", action="store_true",
                    help="report the threshold even if the halves disagree")
    args = ap.parse_args()

    from anemoi.datasets import open_dataset

    cfg = _compose.compose(args.config_dir, args.config_name)

    # ---- the units check, on each half separately --------------------------
    halves = cfg.dataloader.dataset.cutout
    labels = ("LAM, MEPS", "global, IFS")
    verdicts = []
    print("=== units of tp, per source file")
    for label, half in zip(labels, halves):
        spec = half["dataset"]
        paths = [Path(p) for p in (spec["concat"] if "concat" in spec else [spec])]
        scale, offset = declared_rescale(half, "tp")
        rows = half_units(paths, scale, offset)
        note = "" if scale == 1.0 and offset == 0.0 else \
            f"  (config rescales by {scale:g})"
        for name, units, mx in rows:
            print(f"  {label:12s} {name:38s} max {mx:12.6g}  -> {units}{note}")
        found = {u for _, u, _ in rows if u not in ("empty", "no tp")}
        verdicts.append((label, found))

    unit_sets = [u for _, u in verdicts]
    if len(set(frozenset(u) for u in unit_sets)) != 1 or any(len(u) != 1 for u in unit_sets):
        print("\nTHE TWO HALVES DO NOT AGREE ON UNITS.", file=sys.stderr)
        print(
            "The cutout would join a field that jumps by a factor of about a\n"
            "thousand at the domain boundary, and one set of normalisation\n"
            "statistics would be fitted across both. Fine-tuning on that\n"
            "measures the seam, not the weather.\n\n"
            "The fix is on the MEPS side, in bris/configs/meps_2p5km.yaml:\n"
            "precipitation_amount_acc is renamed to tp and never rescaled from\n"
            "kg/m2 to metres. Either add the rescale there and rebuild, or\n"
            "apply it in the dataloader so the built data is left alone.\n\n"
            "Re-run with --skip-units only to look at the numbers. Do not\n"
            "start a run.",
            file=sys.stderr,
        )
        if not args.skip_units:
            return 2
        print("\n--skip-units given; continuing with numbers that are not comparable\n",
              file=sys.stderr)
    else:
        print(f"  both halves store tp in {unit_sets[0].pop()}\n")

    # ---- the threshold, in the units the loss actually sees ------------------
    # NOT FROM THE DATASET'S STATISTICS. The normaliser's scale is stored in the
    # checkpoint with the weights, and the warm start loads it over whatever the
    # dataset would have given. For precipitation MET's normaliser divides by a
    # spread and subtracts no mean. The first version of this converted with the
    # dataset's mean and spread instead, and the 12.66 it printed meant 30 mm to
    # the model rather than 20.
    ds = open_dataset(cfg.dataloader.dataset)
    names = list(ds.variables)
    if args.variable not in names:
        raise SystemExit(f"{args.variable!r} is not in the dataset: {names[:8]} ...")
    i = names.index(args.variable)
    stats = {k: float(v[i]) for k, v in ds.statistics.items()}
    stored_units = classify_units(stats["maximum"])
    native = args.mm / 1000.0 if stored_units == "m" else args.mm

    mul, add, source = model_normaliser(cfg, args.variable)
    normalised = native * mul + add

    print("=== dataset the run will open")
    print(f"  states      {ds.shape[0]}")
    print(f"  dates       {ds.dates[0]} .. {ds.dates[-1]}")
    print(f"\n=== {args.variable}")
    print(f"  dataset     mean {stats['mean']:.6g}, stdev {stats['stdev']:.6g}  (not used by the model)")
    print(f"  model       normalised = x * {mul:.6g} + {add:.6g}  (from {Path(source).name})")
    print(f"\n=== threshold")
    print(f"  {args.mm:g} mm  =  {native:.8g} in stored units")
    print(f"\n  tail_threshold: {normalised:.8g}")

    configured = configured_tail_threshold(cfg)
    if configured is not None:
        implied = (configured - add) / mul
        implied_mm = implied * 1000.0 if stored_units == "m" else implied
        print(f"\n  configured tail_threshold {configured:.8g} means {implied_mm:.2f} mm per 6 h")
        if abs(configured - normalised) > 1e-3 * max(1.0, abs(normalised)):
            print(f"\nTHE CONFIGURED THRESHOLD DOES NOT MEAN {args.mm:g} mm. Set it to "
                  f"{normalised:.8g} or pass the --mm it was meant for.", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

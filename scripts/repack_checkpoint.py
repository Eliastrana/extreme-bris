#!/usr/bin/env python3
"""Turn the published inference checkpoint into one anemoi can warm start from.

    scripts/repack_checkpoint.py
    scripts/repack_checkpoint.py --out ~/bris-models/bris-forecaster/warm-start.ckpt

THE TWO CHECKPOINTS AND WHY NEITHER WORKS AS PUBLISHED. MET publish a training
checkpoint and an inference checkpoint. The training one is what transfer
learning wants and it cannot be opened here: it unpickles into
anemoi.models.migrations, a module absent from the anemoi-models their own
pyproject pins. The inference one opens cleanly but is a pickled model object,
while transfer_learning_loading expects a dict and reaches straight for
checkpoint["state_dict"].

So this takes the weights out of the object and writes the dict.

THE PART THAT HAS TO BE CHECKED RATHER THAN ASSUMED. anemoi loads with

    model.load_state_dict(state_dict, strict=False)

and discards what it returns. Keys that do not match are not an error: they are
skipped in silence. A repackaged file with the wrong key prefix therefore loads
nothing at all, reports nothing at all, and trains a randomly initialised model
while every log line says transfer learning succeeded. That failure would be
invisible until the results made no sense, which is weeks later.

The Lightning module holds the model as an attribute called `model`, so its
keys should be the inference object's keys with `model.` in front. Should is
not good enough. This reads the key names out of the training checkpoint too,
by stubbing the missing module just far enough to unpickle, and compares. If
that succeeds the naming is confirmed against MET's own file rather than
against my reading of the class hierarchy.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("torch", "anemoi")

import torch  # noqa: E402

PREFIX = "model."


def stub_missing(name: str) -> None:
    """Make an absent module importable enough for the unpickler to walk past it.

    Only used to read key names, never to run anything. Any attribute becomes a
    throwaway class, which is enough for pickle to construct placeholders for
    objects whose contents are not wanted.
    """
    if name in sys.modules:
        return
    module = types.ModuleType(name)

    class _Any:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    module.__getattr__ = lambda attr: type(attr, (_Any,), {})  # noqa: ARG005
    sys.modules[name] = module


def training_checkpoint_keys(path: Path) -> set[str] | None:
    """Key names from MET's training checkpoint, if it can be coaxed open."""
    if not path.exists():
        return None
    for missing in ("anemoi.models.migrations",
                    "anemoi.models.migrations.migrator"):
        stub_missing(missing)
    try:
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not open the training checkpoint: "
              f"{type(exc).__name__}: {str(exc)[:160]}")
        return None
    sd = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if not sd:
        print("  the training checkpoint has no state_dict to compare against")
        return None
    return set(sd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    models = Path.home() / "bris-models" / "bris-forecaster"
    ap.add_argument("--inference", type=Path,
                    default=models / "bris-crpsfft_inference.ckpt")
    ap.add_argument("--training", type=Path,
                    default=models / "bris-crpsfft.ckpt",
                    help="only read, to confirm the key names")
    ap.add_argument("-o", "--out", type=Path,
                    default=models / "bris-crpsfft_warmstart.ckpt")
    args = ap.parse_args()

    print(f"=== reading {args.inference.name}")
    obj = torch.load(args.inference, weights_only=False, map_location="cpu")
    if isinstance(obj, dict):
        raise SystemExit(
            "this file is already a dict, not a model object. If it has a "
            "state_dict key it needs no repacking."
        )
    inner = obj.state_dict()
    print(f"  {len(inner)} tensors, "
          f"{sum(t.numel() for t in inner.values()) / 1e6:.1f} M parameters")
    print(f"  first keys: {', '.join(list(inner)[:3])}")

    state_dict = {PREFIX + k: v for k, v in inner.items()}

    print(f"\n=== confirming key names against {args.training.name}")
    target = training_checkpoint_keys(args.training)
    if target is None:
        print("  UNCONFIRMED. Proceeding on the assumption that the Lightning\n"
              "  module holds the model under the attribute 'model'. If the\n"
              "  loss at step 0 looks like an untrained model, this is why.")
    else:
        ours, overlap = set(state_dict), set(state_dict) & target
        print(f"  training checkpoint has {len(target)} keys")
        print(f"  {len(overlap)} of our {len(ours)} match")
        if len(overlap) < 0.9 * len(ours):
            missing = sorted(ours - target)[:5]
            extra = sorted(target - ours)[:5]
            print("\n  THE KEYS DO NOT LINE UP. Loading this would be silent and "
                  "empty.", file=sys.stderr)
            print(f"  ours not in theirs:   {', '.join(missing)}", file=sys.stderr)
            print(f"  theirs not in ours:   {', '.join(extra)}", file=sys.stderr)
            return 1
        print("  confirmed against MET's own file.")

    data_indices = getattr(obj, "data_indices", None)
    if data_indices is None or not hasattr(data_indices, "name_to_index"):
        raise SystemExit(
            "the inference model carries no data_indices with a name_to_index. "
            "anemoi reads that straight after the weights, to compare the "
            "checkpoint's variable ordering against the dataset's, and there is "
            "nothing sensible to put there instead."
        )
    print(f"\n=== data_indices: {len(data_indices.name_to_index)} variables")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict,
                "hyper_parameters": {"data_indices": data_indices}}, args.out)
    size = args.out.stat().st_size / 1e9
    print(f"\nwrote {args.out} ({size:.2f} GB)")
    print("\nPoint hardware/files/ex3.yaml at this, not at the inference file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

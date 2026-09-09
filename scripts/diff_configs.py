#!/usr/bin/env python3
"""Compose two training configs and print every key on which they differ.

    scripts/diff_configs.py finetune finetune_tail

WHY. The two arms of the experiment are supposed to differ in the loss and in
nothing else. That claim is what lets a difference in the results be
attributed to the weighting rather than to the data, the schedule, the
ensemble size or a stray edit. It is also a claim about two YAML files that
inherit from each other and get composed by Hydra, which is not the kind of
thing to take on trust.

Run it on the login node before submitting. It takes a second and it composes
the configs exactly as anemoi will, so it doubles as a syntax check: a
defaults list that does not resolve fails here rather than three hours into a
queue.

Expected output for the two arms as written:

    training.training_loss.losses[2]...     <missing>  ->  ...
    training.training_loss.loss_weights     [1.0, 0.1] ->  [1.0, 0.1, 100.0]

Anything else is a bug in one of the two files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv

_venv.ensure('hydra')

MISSING = object()


def flatten(node, prefix: str = "") -> dict:
    """Every leaf as a dotted path, with list elements indexed.

    Lists are walked rather than compared whole, so a changed loss term names
    the field that changed instead of dumping both lists side by side.
    """
    from omegaconf import DictConfig, ListConfig

    out: dict = {}
    if isinstance(node, (DictConfig, dict)):
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(node, (ListConfig, list)):
        for i, v in enumerate(node):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = node
    return out


def compose(config_dir: Path, name: str):
    from hydra import compose as hydra_compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        return hydra_compose(config_name=name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", nargs="?", default="finetune")
    ap.add_argument("right", nargs="?", default="finetune_tail")
    ap.add_argument("--config-dir", type=Path, default=REPO / "bris" / "train")
    ap.add_argument("--expect-prefix", default="training.training_loss",
                    help="differences outside this prefix make the run non-zero")
    args = ap.parse_args()

    a = flatten(compose(args.config_dir, args.left))
    b = flatten(compose(args.config_dir, args.right))

    keys = sorted(set(a) | set(b))
    rows = [(k, a.get(k, MISSING), b.get(k, MISSING))
            for k in keys if a.get(k, MISSING) != b.get(k, MISSING)]

    if not rows:
        print(f"{args.left} and {args.right} compose to the same config. "
              "That is not an experiment.")
        return 1

    def show(v):
        return "<absent>" if v is MISSING else repr(v)

    width = max(len(k) for k, _, _ in rows)
    print(f"=== {len(rows)} differences between {args.left} and {args.right}\n")
    stray = []
    for k, va, vb in rows:
        mark = " " if k.startswith(args.expect_prefix) else "!"
        if mark == "!":
            stray.append(k)
        print(f"{mark} {k:<{width}}  {show(va)}  ->  {show(vb)}")

    if stray:
        print(f"\n{len(stray)} of them are outside {args.expect_prefix}, marked !.",
              file=sys.stderr)
        print("The arms are not comparable while those stand: a result could "
              "come from any of them.", file=sys.stderr)
        return 2

    print(f"\nAll differences are inside {args.expect_prefix}. The arms are comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

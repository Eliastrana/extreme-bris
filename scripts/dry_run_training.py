#!/usr/bin/env python3
"""Build everything a training run builds, on the CPU, and stop before the data.

    scripts/dry_run_training.py finetune_tail
    scripts/dry_run_training.py finetune

WHY. Getting a card on this cluster has taken anywhere from thirty seconds to
several days. Spending that wait to discover a missing config key is the worst
trade available, and it has happened repeatedly: the run died on the Weights
and Biases entity, then the mlflow tracking uri, then a loss class that does
not exist in this anemoi, then a run id that has to be set before a warm start
is looked for at all, then a checkpoint in the wrong format, then a logger that
is not installed. Every one of those is decided before the first batch, and
none of them needs a GPU.

WHAT IT DOES. It constructs anemoi's own trainer object and touches, in order,
the datamodule, the model, the callbacks and the loggers. Those are cached
properties, so reading them runs the same code a real run runs: the config is
resolved, the datasets are opened, the graph is read, the scalers are built,
the loss is instantiated, the published weights are loaded and the checkpoint's
variable ordering is compared against the datasets'.

WHAT IT CANNOT TELL YOU. Anything that needs a forward pass. The model holds
1.36 million nodes and the encoder's edge tensors run to tens of gigabytes, so
memory is decided on the card and only on the card. A clean dry run means the
next failure will be about size or speed rather than about spelling.

IT DOES NOT TRAIN AND DOES NOT WRITE. max_steps is forced to zero and the
accelerator to the CPU, so nothing is optimised and no checkpoint is produced.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO))
import _venv  # noqa: E402
import _compose  # noqa: E402

_venv.ensure("anemoi", "torch")

import logging  # noqa: E402
import os  # noqa: E402

# anemoi takes its base seed from the environment and asserts if neither
# ANEMOI_BASE_SEED nor SLURM_JOB_ID is set. A job always has one; a login shell
# does not. This must be the same value bris/slurm/finetune.sbatch uses, or the
# dry run composes a different run than the one it is standing in for, which is
# the one thing a dry run must never do.
BASE_SEED = "20260909"


# Everything a GPU would otherwise decide, pinned so this runs anywhere.
CPU_OVERRIDES = [
    "hardware.accelerator=cpu",
    "hardware.num_nodes=1",
    "hardware.num_gpus_per_node=1",
    "hardware.num_gpus_per_model=1",
    "hardware.num_gpus_per_ensemble=1",
    "training.max_steps=0",
]

# The order matters: the model needs the datamodule's variable indices, and the
# callbacks need the model. Reading them one at a time names which step failed.
STAGES = [
    ("datamodule", "opens the six zarrs, builds the cutout, counts variables"),
    ("model", "graph, scalers, loss, published weights, variable comparison"),
    ("callbacks", "checkpointing and plotting"),
    ("loggers", "wandb, mlflow, tensorboard"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_name", nargs="?", default="finetune_tail")
    ap.add_argument("--config-dir", type=Path, default=REPO.parent / "bris" / "train")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress anemoi's own INFO logging")
    ap.add_argument("--keep-gpu-settings", action="store_true",
                    help="do not force the CPU; only useful on a compute node")
    args = ap.parse_args()

    # xbris.losses is named by _target_ in the tail arm, so it must import.
    sys.path.insert(0, str(REPO.parent))

    # anemoi's own command line configures logging; constructing the trainer
    # directly does not, so every LOGGER.info it emits was being discarded.
    # That hid the very lines worth reading here, such as what each variable is
    # scaled by. Warnings came through regardless, which is why the tendency
    # warning was visible and nothing else was.
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not os.environ.get("ANEMOI_BASE_SEED"):
        os.environ["ANEMOI_BASE_SEED"] = BASE_SEED
        print(f"note: ANEMOI_BASE_SEED unset; using {BASE_SEED}, "
              "the same value the job script uses\n")

    overrides = [] if args.keep_gpu_settings else CPU_OVERRIDES
    print(f"=== composing {args.config_name}")
    for o in overrides:
        print(f"  override {o}")
    cfg = _compose.compose(args.config_dir, args.config_name, overrides)

    from anemoi.training.train.train import AnemoiTrainer

    print("\n=== constructing the trainer")
    started = time.time()
    try:
        trainer = AnemoiTrainer(cfg)
    except Exception:  # noqa: BLE001
        print("\nFAILED while constructing the trainer.\n", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(f"  ok in {time.time() - started:.1f}s")

    for name, what in STAGES:
        print(f"\n=== {name}: {what}")
        started = time.time()
        try:
            value = getattr(trainer, name)
        except Exception:  # noqa: BLE001
            print(f"\nFAILED at {name}.\n", file=sys.stderr)
            traceback.print_exc()
            print(f"\nThat is the same failure a queued job would have hit, "
                  f"found without one.", file=sys.stderr)
            return 1
        kind = type(value).__name__
        extra = ""
        if name == "model":
            n = sum(p.numel() for p in value.parameters())
            trainable = sum(p.numel() for p in value.parameters() if p.requires_grad)
            extra = f", {n / 1e6:.1f} M parameters, {trainable / 1e6:.1f} M trainable"
        elif isinstance(value, list):
            extra = f", {len(value)} of them"
        print(f"  ok in {time.time() - started:.1f}s  [{kind}{extra}]")

    print("\nEverything that is decided before the first batch is sound.")
    print("What is left for a card to decide is memory and speed, and nothing "
          "here\ncan tell you about either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Changes to anemoi applied identically in every training run.

Loaded by scripts/anemoi_train.py, which the job script runs in place of the
anemoi-training command, and by scripts/dry_run_training.py. Nothing here
changes what is computed, only where the memory goes, and both arms get exactly
the same patches, so the comparison between them is untouched.

MAPPER ACTIVATIONS ARE RECOMPUTED, NOT STORED.

The model has three parts. The encoder maps 1.36 million data nodes onto the
hidden mesh, the processor works on the mesh, and the decoder maps back. In
training every intermediate tensor is kept for the backward pass unless
something says otherwise. The processor says otherwise: anemoi runs its layers
under torch.utils.checkpoint, keeping only each layer's input and recomputing
the rest when gradients are needed. The two mappers do not, and they are the
parts that work on the full grid.

On one 140 GB card, with two ensemble members, that was the whole problem:

  * sixteen, thirty-two and sixty-four chunks all failed at the same line with
    nearly the same peak, so it was not one oversized temporary tensor;
  * offloading the processor made the attempt thirteen times slower and moved
    the peak by about two gigabytes, because the processor already stores
    almost nothing;
  * the extra twelve minutes that offload cost showed the processor had run
    before memory ran out, which puts the failure in the decoder, while the
    card still held everything the encoder had stored.

Recomputing the mapper blocks trades one extra forward pass of each mapper per
step for most of their stored activations. Offloading would have moved those
same tensors across PCIe twice per step, and the processor showed what that
costs.

The patch is found rather than named. The mapper block is the class whose
forward switches to NUM_CHUNKS_INFERENCE_MAPPER outside training, which is a
property of its code rather than of a class name that may move between anemoi
versions. If no such class exists the patch refuses, instead of silently doing
nothing and letting a run fail on memory for a reason already solved.
"""

from __future__ import annotations

import inspect
import os
import sys

_APPLIED: list[str] = []


def _mapper_block_classes() -> list[type]:
    from anemoi.models.layers import block

    found = []
    for obj in vars(block).values():
        if not inspect.isclass(obj) or obj.__module__ != block.__name__:
            continue
        forward = obj.__dict__.get("forward")
        if forward is None:
            continue
        try:
            source = inspect.getsource(forward)
        except (OSError, TypeError):
            continue
        if "NUM_CHUNKS_INFERENCE_MAPPER" in source:
            found.append(obj)
    return found


def checkpoint_mappers() -> list[str]:
    """Wrap each mapper block's forward in activation checkpointing."""
    import torch
    from torch.utils.checkpoint import checkpoint

    classes = _mapper_block_classes()
    if not classes:
        raise RuntimeError(
            "no mapper block found in anemoi.models.layers.block: nothing has a "
            "forward that reads NUM_CHUNKS_INFERENCE_MAPPER. This anemoi differs "
            "from the one the patch was written against; look before training."
        )

    for cls in classes:
        if getattr(cls.forward, "_xbris_checkpointed", False):
            continue
        original = cls.forward

        def forward(self, *args, _original=original, **kwargs):
            # Only when a backward pass will follow. Validation and inference
            # store nothing, so recomputing there would cost time for no gain.
            if self.training and torch.is_grad_enabled():
                return checkpoint(_original, self, *args, use_reentrant=False, **kwargs)
            return _original(self, *args, **kwargs)

        forward._xbris_checkpointed = True
        forward.__wrapped__ = original
        cls.forward = forward
        _APPLIED.append(f"{cls.__name__}.forward")
    return list(_APPLIED)


def apply() -> list[str]:
    """Apply every patch this project needs. Safe to call more than once.

    XBRIS_CHECKPOINT_MAPPERS=0 turns the mapper patch off, for measuring what it
    costs. Leave it on for both arms or off for both.
    """
    applied: list[str] = []
    if os.environ.get("XBRIS_CHECKPOINT_MAPPERS", "1") != "0":
        applied = checkpoint_mappers()
        print(f"xbris: recomputing activations in {', '.join(applied)}",
              file=sys.stderr)
    else:
        print("xbris: mapper checkpointing is OFF", file=sys.stderr)
    return applied

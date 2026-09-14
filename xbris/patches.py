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

RECOMPUTING THE WHOLE BLOCK WAS NOT ENOUGH, SO THE ATTENTION IS CHECKPOINTED
PER CHUNK AS WELL.

With the block recomputed, two members still ran out, at nearly the same peak,
but somewhere new: inside torch.utils.checkpoint's unpack_hook, recomputing the
mapper block during the backward pass. The forward pass had completed for the
first time. What was left is that recomputing a block rematerialises all of
that block's activations at once while its gradient is taken, and the mapper
splits its attention into num_chunks pieces only to keep every piece anyway.

So inside a mapper block, each call to the attention convolution is also
checkpointed. The backward pass then recomputes one chunk at a time, and the
peak is one chunk rather than one block. It applies only inside mapper blocks:
the processor already recomputes its layers, and checkpointing its attention
again would compute it three times per step for nothing.

Knowing whether a convolution is inside a mapper needs a flag set by the block
itself, and it has to be set during the recompute too, not only during the
original forward. The backward pass runs outside the wrapper that first called
the block, often on another thread, so the flag is set by the function that
checkpoint calls, which is also the function it calls again to recompute.

The patch is found rather than named. The mapper block is the class whose
forward switches to NUM_CHUNKS_INFERENCE_MAPPER outside training, which is a
property of its code rather than of a class name that may move between anemoi
versions. If no such class exists the patch refuses, instead of silently doing
nothing and letting a run fail on memory for a reason already solved.
"""

from __future__ import annotations

import contextvars
import inspect
import os
import sys

_APPLIED: list[str] = []

# True while a mapper block is running, including when checkpoint re-runs it
# during the backward pass. Read by the attention patch.
_IN_MAPPER: contextvars.ContextVar[bool] = contextvars.ContextVar("xbris_in_mapper",
                                                                  default=False)


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

        def run(self, *args, _original=original, **kwargs):
            # The flag is set here, in the function checkpoint calls, so that it
            # is also set when checkpoint calls it again to recompute.
            token = _IN_MAPPER.set(True)
            try:
                return _original(self, *args, **kwargs)
            finally:
                _IN_MAPPER.reset(token)

        def forward(self, *args, _run=run, _original=original, **kwargs):
            # Only when a backward pass will follow. Validation and inference
            # store nothing, so recomputing there would cost time for no gain.
            if self.training and torch.is_grad_enabled():
                return checkpoint(_run, self, *args, use_reentrant=False, **kwargs)
            return _run(self, *args, **kwargs)

        forward._xbris_checkpointed = True
        forward.__wrapped__ = original
        cls.forward = forward
        _APPLIED.append(f"{cls.__name__}.forward")
    return list(_APPLIED)


def checkpoint_mapper_attention() -> list[str]:
    """Checkpoint each attention call inside a mapper, so chunks recompute singly."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from anemoi.models.layers import conv

    cls = getattr(conv, "GraphTransformerConv", None)
    if cls is None or "forward" not in cls.__dict__:
        raise RuntimeError(
            "anemoi.models.layers.conv.GraphTransformerConv is not there. Every "
            "out-of-memory traceback so far ran through it; this anemoi differs "
            "from the one the patch was written against."
        )
    if getattr(cls.forward, "_xbris_checkpointed", False):
        return [f"{cls.__name__}.forward"]
    original = cls.forward

    def forward(self, *args, _original=original, **kwargs):
        if _IN_MAPPER.get() and self.training and torch.is_grad_enabled():
            return checkpoint(_original, self, *args, use_reentrant=False, **kwargs)
        return _original(self, *args, **kwargs)

    forward._xbris_checkpointed = True
    forward.__wrapped__ = original
    cls.forward = forward
    _APPLIED.append(f"{cls.__name__}.forward, inside mappers only")
    return [f"{cls.__name__}.forward"]


def apply() -> list[str]:
    """Apply every patch this project needs. Safe to call more than once.

    XBRIS_CHECKPOINT_MAPPERS=0 turns both mapper patches off, and
    XBRIS_CHECKPOINT_ATTENTION=0 only the per-chunk one, for measuring what each
    costs. Whatever is chosen, choose it for both arms.
    """
    if os.environ.get("XBRIS_CHECKPOINT_MAPPERS", "1") != "0":
        checkpoint_mappers()
        # Without the block patch there is no flag, so this would do nothing.
        if os.environ.get("XBRIS_CHECKPOINT_ATTENTION", "1") != "0":
            checkpoint_mapper_attention()
        print(f"xbris: recomputing activations in {'; '.join(_APPLIED)}",
              file=sys.stderr)
    else:
        print("xbris: mapper checkpointing is OFF", file=sys.stderr)
    return list(_APPLIED)

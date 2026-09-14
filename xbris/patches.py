"""Changes to anemoi applied identically in every training run.

Loaded by scripts/anemoi_train.py, which the job script runs in place of the
anemoi-training command, and by scripts/dry_run_training.py. Both arms get the
same patches, so the comparison between them is untouched.

ENSEMBLE MEMBERS GO THROUGH THE MODEL ONE AT A TIME.

anemoi's ensemble model stacks every member into one batch and runs encoder,
processor and decoder over all of them at once. The decoder writes back onto
the full data grid, 1.36 million nodes, so with two members it works on 2.72
million, and a single 1024-channel float32 tensor over those is

    2 * 1,359,281 * 1024 * 4 bytes = 10.37 GiB

That is, to the byte, the allocation that kept failing on a 140 GB H200. It
is a per-node tensor, which is why it was the same size at sixteen, thirty-two
and sixty-four attention chunks: chunks split the edges, and every chunk still
aggregates into a full-size output over the destination nodes.

Running members one after another halves every such tensor at two members,
and the saving grows with the ensemble. The mathematics is unchanged: the
layer norms are per sample, the noise injector draws per sample, and the
bounding is elementwise, so nothing couples one member to another inside the
forward pass. The loss still sees all members together, because the outputs
are joined along the ensemble axis before it is computed. Only the random
numbers drawn for the noise differ, in value rather than in distribution.

WHAT WAS TRIED FIRST, AND WHY IT IS GONE.

Two earlier patches wrapped the mapper block, and the attention convolution
inside it, in activation checkpointing. They made nothing fit, and the peak
rose from 134 to 141 GB. The reason is in encoder_processor_decoder.py:
anemoi's _run_mapper already wraps each whole mapper in torch.utils.checkpoint.
Checkpointing inside a checkpoint only adds recomputation. The failure was
never stored activations; it was the size of one member-stacked forward pass.

XBRIS_SEQUENTIAL_MEMBERS=0 turns the patch off, for measuring what it costs.
Whatever is chosen, choose it for both arms.
"""

from __future__ import annotations

import os
import sys

_APPLIED: list[str] = []


def ensemble_members_one_at_a_time() -> str:
    """Run AnemoiEnsModelEncProcDec.forward once per member and join the results."""
    import torch
    from anemoi.models.models import ens_encoder_processor_decoder as module

    cls = getattr(module, "AnemoiEnsModelEncProcDec", None)
    if cls is None or "forward" not in cls.__dict__:
        raise RuntimeError(
            "anemoi.models.models.ens_encoder_processor_decoder.AnemoiEnsModelEncProcDec "
            "is not there. This anemoi differs from the one the patch was written "
            "against; look before training."
        )
    label = f"{cls.__name__}.forward, one member at a time"
    if getattr(cls.forward, "_xbris_sequential", False):
        return label
    original = cls.forward

    def forward(self, x, *, fcstep, model_comm_group=None, grid_shard_shapes=None, **kwargs):
        # Input is (batch, time, ensemble, grid, vars); output is
        # (batch, ensemble, grid, vars). See the module docstring for why this
        # is exact.
        members = x.shape[2]
        if members <= 1:
            return original(self, x, fcstep=fcstep, model_comm_group=model_comm_group,
                            grid_shard_shapes=grid_shard_shapes, **kwargs)
        outputs = [
            original(self, x[:, :, i:i + 1], fcstep=fcstep, model_comm_group=model_comm_group,
                     grid_shard_shapes=grid_shard_shapes, **kwargs)
            for i in range(members)
        ]
        return torch.cat(outputs, dim=1)

    forward._xbris_sequential = True
    forward.__wrapped__ = original
    cls.forward = forward
    _APPLIED.append(label)
    return label


def apply() -> list[str]:
    """Apply every patch this project needs. Safe to call more than once."""
    if os.environ.get("XBRIS_SEQUENTIAL_MEMBERS", "1") != "0":
        ensemble_members_one_at_a_time()
        print(f"xbris: {'; '.join(_APPLIED)}", file=sys.stderr)
    else:
        print("xbris: sequential ensemble members OFF", file=sys.stderr)
    return list(_APPLIED)

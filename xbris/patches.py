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

TRAINING STATES CAN BE DRAWN WITH WEIGHTS (XBRIS_OVERSAMPLE).

anemoi draws each epoch's training samples as one permutation of every valid
date, so a state with heavy rain is seen exactly as often as a dry one. With
XBRIS_OVERSAMPLE set to a JSON file of {"weights": {date: weight}}, the
training dataset, and only that one, draws its samples with replacement in
proportion to those weights instead. A sample's weight is the largest weight
among its target dates, so a rollout sample counts as heavy if any step it is
scored on is. Dates the file does not list weigh 1. Validation and test are
untouched, so validation losses stay comparable with every other run.

It works by handing the dataset's own random generator to anemoi's __iter__
wrapped, so the one call that draws the permutation draws weighted indices
instead, from the same seeded generator. Everything else __iter__ does is
anemoi's own code, unchanged.
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


class _WeightedChoice:
    """A numpy Generator whose choice() over the valid dates draws by weight."""

    def __init__(self, rng, valid, p):
        self._rng, self._valid, self._p = rng, valid, p

    def choice(self, a, size=None, replace=True, p=None):
        import numpy as np

        if p is None and np.array_equal(np.asarray(a), self._valid):
            return self._rng.choice(self._valid, size=size, replace=True, p=self._p)
        return self._rng.choice(a, size=size, replace=replace, p=p)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def sample_weights(dates, valid, relative_date_indices, weights_by_date):
    """One probability per valid index: the largest weight over its target dates."""
    import numpy as np

    targets = list(relative_date_indices)[1:] or list(relative_date_indices)
    w = np.ones(len(valid), dtype=np.float64)
    for k, i in enumerate(valid):
        best = 1.0
        for r in targets:
            j = int(i) + int(r)
            if 0 <= j < len(dates):
                best = max(best, float(weights_by_date.get(dates[j], 1.0)))
        w[k] = best
    return w / w.sum()


def weighted_training_draws(cls, weights_by_date, label="train") -> str:
    """Make cls.__iter__ draw training samples with replacement, by weight."""
    if getattr(cls.__iter__, "_xbris_weighted", False):
        return f"{cls.__name__}: weighted training draws"
    original = cls.__iter__

    def __iter__(self):
        if getattr(self, "label", None) != label or not getattr(self, "shuffle", False):
            yield from original(self)
            return
        import numpy as np

        valid = np.asarray(self.valid_date_indices)
        if getattr(self, "_xbris_p", None) is None:
            raw = getattr(self.data, "dates", None)
            dates = [str(np.datetime_as_string(d, unit="s")) for d in raw] if raw is not None else []
            self._xbris_p = sample_weights(dates, valid, self.relative_date_indices, weights_by_date)
            heavy = float(self._xbris_p[self._xbris_p > self._xbris_p.min()].sum())
            print(f"xbris: weighted draws over {len(valid)} training samples, "
                  f"{int((self._xbris_p > self._xbris_p.min()).sum())} weighted up, "
                  f"expected share of draws {heavy:.3f}", file=sys.stderr)
        real = self.rng
        self.rng = _WeightedChoice(real, valid, self._xbris_p)
        gen = original(self)
        try:
            first = next(gen)
        except StopIteration:
            return
        finally:
            self.rng = real
        yield first
        yield from gen

    __iter__._xbris_weighted = True
    __iter__.__wrapped__ = original
    cls.__iter__ = __iter__
    label_text = f"{cls.__name__}: weighted training draws"
    _APPLIED.append(label_text)
    return label_text


def oversample_from_env() -> str | None:
    path = os.environ.get("XBRIS_OVERSAMPLE")
    if not path:
        return None
    import json

    with open(path) as f:
        weights = {str(k): float(v) for k, v in json.load(f)["weights"].items()}
    from anemoi.training.data.dataset.ensdataset import EnsNativeGridDataset

    return weighted_training_draws(EnsNativeGridDataset, weights)


def apply() -> list[str]:
    """Apply every patch this project needs. Safe to call more than once."""
    if os.environ.get("XBRIS_SEQUENTIAL_MEMBERS", "1") != "0":
        ensemble_members_one_at_a_time()
        print(f"xbris: {'; '.join(_APPLIED)}", file=sys.stderr)
    else:
        print("xbris: sequential ensemble members OFF", file=sys.stderr)
    if oversample_from_env():
        print(f"xbris: training states drawn by weight from {os.environ['XBRIS_OVERSAMPLE']}", file=sys.stderr)
    return list(_APPLIED)

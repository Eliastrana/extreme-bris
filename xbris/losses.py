"""Threshold-weighted CRPS, as a drop-in extra term in Bris's own loss.

WHAT THIS IS. Ordinary CRPS integrates the distance between the forecast
distribution and the observation over every threshold, weighting them all
equally. Rain of 2 mm and rain of 80 mm therefore count the same, and since
almost every state is near zero, the tail contributes almost nothing to the
gradient. Threshold-weighted CRPS puts a weight w(z) inside that integral so
chosen thresholds count more.

WHY IT IS A TRANSFORM AND NOT A NEW SCORE. For a weight w with antiderivative
v (so v' = w), the threshold-weighted CRPS of a forecast equals the ORDINARY
CRPS of the transformed forecast and the transformed observation:

    twCRPS(F, y; w) = CRPS(v(F), v(y))

for any non-decreasing v. That is the result in Allen, Ginsbourger and Ziegel
(2023) on transformed kernel scores, and it does two things for me. It keeps
the score proper, which a hand-rolled tail penalty would not, and it means I
do not reimplement the ensemble CRPS estimator at all. I transform the inputs
and hand them to the estimator anemoi already ships and Bris was trained with.

For the indicator weight w(z) = 1{z >= t}, the antiderivative is

    v(z) = max(z, t)

which collapses everything below the threshold onto a single point. Below t
the score becomes blind; above it, it is unchanged.

WHY IT IS AN EXTRA TERM, NOT A REPLACEMENT. A loss that is blind below the
threshold would happily wreck ordinary days, and ordinary days are most of
them. So the tail arm keeps both terms of Bris's original loss untouched and
adds this as a third, at a weight. The two arms of the experiment then differ
by exactly one entry in one list, which is the only way the comparison means
anything.

WHY THE THRESHOLD IS IN NORMALISED UNITS. The loss sees normalised values, not
millimetres. Normalisation is affine, and max() commutes with an increasing
affine map:

    max(a*z + b, a*t + b) = a * max(z, t) + b   for a > 0

so clipping at the normalised threshold is exactly clipping at the physical
one, up to the same affine scaling the rest of the loss already carries. There
is no approximation here, but there is a number to look up, and getting it
wrong is silent. scripts/tail_threshold.py prints it from the dataset's own
statistics. The config ships with null so that a forgotten lookup stops the
job rather than training on a threshold of zero.

WHY THE OTHER VARIABLES ARE ZEROED. This term is about precipitation. If the
transform left the other 97 channels alone, they would be scored twice, once
here and once in the main term, and the tail arm would quietly reweight
everything rather than the tail. Setting both prediction and target to zero on
those channels makes their contribution identically zero: |0 - 0| in the skill
term, and no spread in the spread term.

The cost of that is dilution. The estimator averages over all channels, so a
term that is nonzero on one of 98 lands about two orders of magnitude below
where it would land if scored alone. Do not guess around it. The first forward
passes log the raw value, and the weight belongs in the config once that
number and the main term's are both on screen.
"""

from __future__ import annotations

import inspect
import logging

import torch
from anemoi.training.losses import AlmostFairKernelCRPS

LOGGER = logging.getLogger(__name__)

# How many forward passes report their own value before going quiet. Enough to
# choose a weight from, few enough to not fill the log.
_REPORT_FIRST = 5


class TailWeightedKernelCRPS(AlmostFairKernelCRPS):
    """AlmostFairKernelCRPS scored on max(x, threshold) of one variable.

    Every argument the base class takes is passed straight through, so this is
    the same estimator with the same scalers, alpha and reduction. Only what
    reaches it differs.

    Parameters
    ----------
    tail_threshold:
        The threshold, in NORMALISED units, from scripts/tail_threshold.py.
        Required. There is no sensible default and a wrong one is invisible.
    tail_variable:
        Name of the variable to score, used to resolve and to cross-check the
        index. Defaults to precipitation.
    tail_index:
        Position of that variable in the model's output channels. Optional if
        anemoi passes data_indices, which is where the answer really lives;
        give both and they are checked against each other.
    tail_scale:
        A plain multiplier applied to the result, separate from the weight in
        CombinedLoss. Keep it at one and set the weight, unless you want the
        two knobs to mean different things in the run log.
    """

    def __init__(
        self,
        *args,
        tail_threshold: float | None = None,
        tail_variable: str = "tp",
        tail_index: int | None = None,
        tail_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if tail_threshold is None:
            raise ValueError(
                "tail_threshold is required and must be in NORMALISED units. "
                "Run scripts/tail_threshold.py against the training dataset to "
                "convert millimetres, then put the number in the config. "
                "Left unset it would default to something, and a threshold "
                "that is wrong by the width of the distribution looks exactly "
                "like a threshold that is right."
            )

        resolved = self._index_from_anemoi(kwargs, tail_variable)
        if resolved is not None and tail_index is not None and resolved != tail_index:
            raise ValueError(
                f"config says {tail_variable} is output channel {tail_index}, "
                f"the checkpoint's data_indices say {resolved}. One of them is "
                "stale. Weighting the wrong channel trains fine and proves "
                "nothing, so this refuses rather than picking a side."
            )
        index = resolved if resolved is not None else tail_index
        if index is None:
            raise ValueError(
                f"cannot tell which output channel is {tail_variable!r}: anemoi "
                "passed no data_indices and the config gave no tail_index. "
                "scripts/inspect_checkpoint.py prints the output ordering."
            )

        self.tail_threshold = float(tail_threshold)
        self.tail_variable = tail_variable
        self.tail_index = int(index)
        self.tail_scale = float(tail_scale)
        self._seen = 0

        # The base class's forward may name its arguments anything. Learn the
        # first two parameter names once, so transforming them works whether
        # the caller passes them positionally or by keyword.
        params = [
            p
            for p in inspect.signature(AlmostFairKernelCRPS.forward).parameters
            if p != "self"
        ]
        if len(params) < 2:
            raise RuntimeError(
                "AlmostFairKernelCRPS.forward takes fewer than two arguments; "
                f"this anemoi version does not match what was assumed: {params}"
            )
        self._pred_name, self._target_name = params[0], params[1]

        LOGGER.info(
            "tail term: %s at output channel %d, threshold %.6g normalised, "
            "scale %.6g",
            self.tail_variable,
            self.tail_index,
            self.tail_threshold,
            self.tail_scale,
        )

    @staticmethod
    def _index_from_anemoi(kwargs: dict, variable: str) -> int | None:
        """Read the output channel for `variable` out of anemoi's data_indices.

        Returns None rather than raising: older versions do not hand losses
        their indices, and the config can supply the number instead.
        """
        di = kwargs.get("data_indices")
        if di is None:
            return None
        for path in (("model", "output"), ("internal_model", "output")):
            node = di
            for step in path:
                node = getattr(node, step, None)
                if node is None:
                    break
            mapping = getattr(node, "name_to_index", None)
            if isinstance(mapping, dict) and variable in mapping:
                return int(mapping[variable])
        return None

    def _chain(self, x: torch.Tensor) -> torch.Tensor:
        """v(x): clip the tail variable at the threshold, zero everything else.

        Variables are the last dimension in every tensor anemoi hands a loss.
        Checked rather than assumed, because a silent broadcast over the wrong
        axis would train and converge and mean nothing.
        """
        if x.shape[-1] <= self.tail_index:
            raise IndexError(
                f"tail_index {self.tail_index} is outside the last dimension of "
                f"a tensor shaped {tuple(x.shape)}. Either the index is wrong or "
                "variables are not the last axis in this anemoi version."
            )
        out = torch.zeros_like(x)
        out[..., self.tail_index] = x[..., self.tail_index].clamp(min=self.tail_threshold)
        return out

    def forward(self, *args, **kwargs):
        args = list(args)
        for position, name in enumerate((self._pred_name, self._target_name)):
            if position < len(args):
                args[position] = self._chain(args[position])
            elif name in kwargs:
                kwargs[name] = self._chain(kwargs[name])
            else:
                raise TypeError(
                    f"neither positional argument {position} nor keyword {name!r} "
                    "was given, so there is nothing to transform"
                )

        value = super().forward(*args, **kwargs)
        result = self.tail_scale * value

        if self._seen < _REPORT_FIRST:
            self._seen += 1
            with torch.no_grad():
                LOGGER.info(
                    "tail term, forward %d of %d reported: %.6g "
                    "(before the CombinedLoss weight). Compare against the "
                    "other terms in the same log to choose loss_weights.",
                    self._seen,
                    _REPORT_FIRST,
                    float(result.detach().mean()),
                )
        return result

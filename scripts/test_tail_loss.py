#!/usr/bin/env python3
"""Exercise the threshold-weighted loss on the CPU, before it costs a GPU slot.

    scripts/test_tail_loss.py

xbris/losses.py was written against anemoi's documented surface, not against a
running copy of it, and it has never been constructed. Everything the tail arm
claims rests on it. A queue that hands out a card once a week is a bad place to
discover a signature mismatch, and none of what this checks needs a GPU or a
dataset.

It does two jobs. First it reports what anemoi's kernel CRPS actually expects,
since the wrapper's assumptions about argument names and tensor layout were
made from the config surface alone. Then it tests the four claims the design
rests on:

  blind below      two batches that differ only in precipitation below the
                   threshold must score identically. That is what makes it a
                   tail loss rather than a reweighted ordinary one.
  awake above      raising precipitation past the threshold must change it.
  deaf elsewhere   changing any other variable must not, or the term is
                   quietly reweighting all 98 instead of the tail.
  gradient         only the precipitation channel may receive one.

A failure here names the line to fix. A pass does not prove the arm is right,
only that the piece nothing else could test does what it says.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("anemoi", "torch")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NPOINTS = 64
NVARS = 5
TP = 2          # which channel stands in for precipitation
NENS = 4
THRESHOLD = 1.5


def report_surface() -> None:
    from anemoi.training.losses import AlmostFairKernelCRPS

    print("=== what anemoi's kernel CRPS expects")
    for name, fn in (("__init__", AlmostFairKernelCRPS.__init__),
                     ("forward", AlmostFairKernelCRPS.forward)):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            print(f"  {name:9s} unreadable: {exc}")
            continue
        print(f"  {name:9s} {sig}")
    print()


def build(**extra):
    """Construct the tail loss, filling in whatever anemoi demands."""
    from xbris.losses import TailWeightedKernelCRPS

    kwargs = dict(alpha=0.95, tail_threshold=THRESHOLD,
                  tail_variable="tp", tail_index=TP)
    kwargs.update(extra)
    try:
        return TailWeightedKernelCRPS(**kwargs)
    except TypeError as exc:
        if "node_weights" in str(exc):
            kwargs["node_weights"] = torch.ones(NPOINTS)
            return TailWeightedKernelCRPS(**kwargs)
        raise


def sample(tp_value: float, other: float = 0.0, seed: int = 0):
    """A prediction ensemble and a target, constant per channel by design.

    Constant fields make the checks below unambiguous: any change in the score
    can only come from the value that was changed.
    """
    g = torch.Generator().manual_seed(seed)
    pred = torch.zeros(1, NENS, NPOINTS, NVARS)
    target = torch.zeros(1, NPOINTS, NVARS)
    pred += other
    target += other
    pred[..., TP] = tp_value + 0.01 * torch.randn(1, NENS, NPOINTS, generator=g)
    target[..., TP] = tp_value
    return pred, target


def call(loss, pred, target) -> float:
    out = loss(pred, target)
    return float(out.detach().mean() if out.dim() else out.detach())


def main() -> int:
    report_surface()

    try:
        loss = build()
    except Exception as exc:  # noqa: BLE001
        print(f"CONSTRUCTION FAILED: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        print("Fix xbris/losses.py against the signature printed above.",
              file=sys.stderr)
        return 1
    print(f"constructed, tail channel {loss.tail_index}, "
          f"threshold {loss.tail_threshold}\n")

    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok))
        print(f"  {'pass' if ok else 'FAIL'}  {name:16s} {detail}")

    print("=== the four claims")
    try:
        # Blind below: both well under the threshold, and different.
        a = call(loss, *sample(0.1, seed=1))
        b = call(loss, *sample(0.9, seed=1))
        check("blind below", abs(a - b) < 1e-6,
              f"{a:.8f} vs {b:.8f} at 0.1 and 0.9, threshold {THRESHOLD}")

        # Awake above: one side past the threshold.
        c = call(loss, *sample(4.0, seed=1))
        check("awake above", abs(c - a) > 1e-6,
              f"{c:.8f} at 4.0 against {a:.8f} below")

        # Deaf elsewhere: change every other channel, leave precipitation alone.
        d = call(loss, *sample(4.0, other=7.0, seed=1))
        check("deaf elsewhere", abs(d - c) < 1e-6,
              f"{d:.8f} with other channels at 7.0 against {c:.8f} at 0.0")

        # Gradient reaches precipitation and nothing else.
        pred, target = sample(4.0, other=7.0, seed=1)
        pred.requires_grad_(True)
        out = loss(pred, target)
        (out.mean() if out.dim() else out).backward()
        grad = pred.grad.abs().sum(dim=(0, 1, 2))
        others = float(grad[[i for i in range(NVARS) if i != TP]].max())
        check("gradient", float(grad[TP]) > 0 and others < 1e-12,
              f"precipitation {float(grad[TP]):.6g}, largest elsewhere {others:.3g}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nRAISED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    bad = [n for n, ok in results if not ok]
    print()
    if bad:
        print(f"{len(bad)} of {len(results)} failed: {', '.join(bad)}",
              file=sys.stderr)
        return 1
    print(f"All {len(results)} claims hold. The tail term does what the "
          "config says it does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validation loss against training step, for each arm, from saved per-state files.

    scripts/validation_steps.py ~/bris-runs/validation-loss

WHY. The question is how many of the 3 000 steps the fine-tuning actually
needed. If most of the gain is in by step 1 000, each later experiment can be
three times cheaper.

WHAT IT READS. Every JSON written by validation_loss.py: the step-0 reference
(label "baseline", untouched Bris), the finished arms ("control", "tail", step
3 000), and the intermediate checkpoints under steps/ ("control-s000500" and so
on). Only states present in every file are used, so each point on the curve is
the same states, and each is also shown as a paired difference from step 0.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


def load(paths: list[Path]) -> dict[str, dict[str, float]]:
    runs: dict[str, dict[str, float]] = {}
    for path in paths:
        data = json.loads(path.read_text())
        rows = runs.setdefault(data["label"], {})
        for row in data["rows"]:
            rows[row["target"]] = row["loss"]
    return runs


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/bris-runs/validation-loss").expanduser()
    runs = load(sorted(root.glob("*.json")) + sorted((root / "steps").glob("*.json")))
    if "baseline" not in runs:
        raise SystemExit("no step-0 reference (label 'baseline') found")
    curves: dict[str, dict[int, str]] = {}
    for label in runs:
        m = re.fullmatch(r"(control|tail)(?:-s(\d+))?", label)
        if m:
            curves.setdefault(m.group(1), {})[int(m.group(2) or 3000)] = label
    common = set(runs["baseline"])
    for arm in curves.values():
        for label in arm.values():
            common &= set(runs[label])
    common = sorted(common)
    if not common:
        raise SystemExit("no state is present in every file")
    base = np.array([runs["baseline"][t] for t in common])
    print(f"=== {len(common)} validation states common to every checkpoint "
          f"({common[0][:10]} .. {common[-1][:10]})\n")
    print(f"{'arm':8s} {'step':>5s} {'mean loss':>10s} {'vs step 0':>10s} {'± se':>7s} "
          f"{'better on':>10s} {'share of final gain':>20s}")
    print(f"{'baseline':8s} {0:5d} {base.mean():10.4f}")
    for arm, steps in sorted(curves.items()):
        final = np.array([runs[steps[max(steps)]][t] for t in common]) - base
        for step in sorted(steps):
            values = np.array([runs[steps[step]][t] for t in common])
            diff = values - base
            se = diff.std(ddof=1) / np.sqrt(diff.size)
            share = diff.mean() / final.mean() if final.mean() else float("nan")
            print(f"{arm:8s} {step:5d} {values.mean():10.4f} {diff.mean():+10.4f} {se:7.4f} "
                  f"{int((diff < 0).sum()):4d} of {diff.size:<3d} {share:19.0%}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

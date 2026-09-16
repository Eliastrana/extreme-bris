#!/usr/bin/env python3
"""Validation loss of any checkpoint, state by state, on a fixed subset.

    scripts/validation_loss.py --label baseline --plan-only
    scripts/validation_loss.py --label control --checkpoint CKPT --cpu --shard 1/2
    scripts/validation_loss.py --compare ~/bris-runs/validation-loss/*.json

WHY. The control arm logged a validation loss of 2.13, and untouched Bris has
no number to set against it, so nobody can say whether fine-tuning moved it.
anemoi logs only the mean over the whole validation period, about 485 states,
which on CPU would take days per checkpoint.

WHAT IT DOES INSTEAD. Scores both checkpoints on the same evenly spaced subset
of validation states, one value per state, and compares them in pairs. Weather
varies far more from one state to the next than two nearby checkpoints do, so
the difference on the same state is a much sharper measure than two means
over the period.

THE SAME COMPUTATION. Composes finetune.yaml, the config the control arm
validated with, and runs Lightning's own validation step: the same loss,
scalers, three members one at a time, rollout and seed. Only the weights
change, through hardware.files.warm_start. On CPU it runs in full precision,
while the logged 2.13 was 16-mixed on a GPU, so compare the checkpoints with
each other here, not with 2.13.

THE TARGET. A sample starting at index i is scored on the state multi_step
later; each row records that target time, which is what the loss is about.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import _venv  # noqa: E402

_venv.ensure("anemoi.training", "numpy")

import numpy as np  # noqa: E402


def compare(paths: list[Path]) -> int:
    runs: dict[str, dict[str, float]] = {}
    for path in paths:
        data = json.loads(path.read_text())
        rows = runs.setdefault(data["label"], {})
        for row in data["rows"]:
            rows[row["target"]] = row["loss"]
    labels = sorted(runs)
    if len(labels) != 2:
        print(f"need exactly two labels to compare, found {labels}")
        return 1
    a, b = labels
    common = sorted(set(runs[a]) & set(runs[b]))
    print(f"=== {a} vs {b}: {len(common)} states in common "
          f"({len(runs[a])} and {len(runs[b])} scored)\n")
    print(f"{'target':20s} {a:>10s} {b:>10s} {b + ' - ' + a:>14s}")
    diffs = []
    for t in common:
        d = runs[b][t] - runs[a][t]
        diffs.append(d)
        print(f"{t:20s} {runs[a][t]:10.4f} {runs[b][t]:10.4f} {d:+14.4f}")
    if not diffs:
        return 1
    diffs = np.array(diffs)
    va = np.array([runs[a][t] for t in common])
    vb = np.array([runs[b][t] for t in common])
    se = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    print(f"\nmean {a}: {va.mean():.4f}   mean {b}: {vb.mean():.4f}")
    print(f"mean difference ({b} - {a}): {diffs.mean():+.4f}  "
          f"(standard error {se:.4f}, {diffs.mean() / se:+.1f} standard errors)")
    print(f"{b} lower on {int((diffs < 0).sum())} of {len(diffs)} states")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="name for this checkpoint in the output")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="weights to load; default is the config's warm start (untouched Bris)")
    ap.add_argument("--config-name", default="finetune")
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2025-07-31")
    # 21, not 20: samples are six hours apart, so every 20th is always the same
    # hour of day and the subset would see only noon. 21 walks through all four.
    ap.add_argument("--every", type=int, default=21, help="score every n-th validation sample")
    ap.add_argument("--shard", default=None, help="k/n: every n-th state of the subset, from the k-th")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", nargs="+", type=Path, default=None)
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare)
    if not args.label:
        ap.error("--label is required unless --compare is given")

    os.environ.setdefault("ANEMOI_BASE_SEED", "20260909")
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER", "32")

    from xbris import patches

    if not args.plan_only:
        patches.apply()
    import _compose

    overrides = [
        f"dataloader.validation.start={args.start}",
        f"dataloader.validation.end={args.end}",
        # One worker keeps samples in the order they are listed, which is how
        # each recorded loss is matched back to its state.
        "dataloader.num_workers.validation=1",
        "hardware.num_nodes=1",
        "hardware.num_gpus_per_node=1",
        "hardware.num_gpus_per_ensemble=1",
        "hardware.num_gpus_per_model=1",
    ]
    if args.checkpoint:
        overrides.append(f"hardware.files.warm_start={args.checkpoint.expanduser()}")
    if args.plan_only or args.cpu:
        overrides.append("hardware.accelerator=cpu")
    if args.cpu:
        import torch

        threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
        torch.set_num_threads(threads)
        print(f"  CPU run, {threads} threads, full precision")
    cfg = _compose.compose(REPO / "bris/train", args.config_name, overrides)
    print(f"=== {args.label}: weights {cfg.hardware.files.warm_start}")

    from anemoi.training.train.train import AnemoiTrainer

    trainer = AnemoiTrainer(cfg)
    dm = trainer.datamodule
    ds = dm.ds_valid
    multi_step = int(cfg.training.multistep_input)
    rel = [int(x) for x in ds.relative_date_indices]
    step = rel[1] - rel[0] if len(rel) > 1 else 1
    target_offset = rel[0] + multi_step * step
    dates = [str(d)[:19] for d in ds.data.dates]

    starts = sorted(int(i) for i in ds.valid_date_indices
                    if 0 <= int(i) + target_offset < len(dates))
    plan = starts[::args.every]
    print(f"  {len(starts)} validation samples, every {args.every}th: {len(plan)} states, "
          f"{dates[plan[0] + target_offset]} .. {dates[plan[-1] + target_offset]}")
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        plan = plan[k - 1::n]
        print(f"  shard {k} of {n}: {len(plan)} states")
    if args.plan_only:
        for s in plan:
            print(f"    start {dates[s]} -> target {dates[s + target_offset]}")
        # Building the model is what loads the weights; a checkpoint in the
        # wrong format should fail here, on the login node, not on a CPU node
        # an hour into a queue.
        n = sum(p.numel() for p in trainer.model.parameters())
        print(f"  model built, weights loaded: {n / 1e6:.1f} M parameters")
        return 0

    import pytorch_lightning as pl

    model = trainer.model
    records: list[float] = []
    combined_forward = model.loss.forward

    def record(*a, **k):
        out = combined_forward(*a, **k)
        records.append(float(out.detach().float().mean()))
        return out

    model.loss.forward = record
    ds.valid_date_indices = np.array(plan, dtype=np.int64)

    pl_trainer = pl.Trainer(
        accelerator=trainer.accelerator,
        strategy=trainer.strategy,
        devices=1,
        num_nodes=1,
        precision="32" if args.cpu else cfg.training.precision,
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        limit_val_batches=1.0,
        use_distributed_sampler=False,
        deterministic=cfg.training.deterministic,
    )
    started = time.time()
    pl_trainer.validate(model, datamodule=dm, verbose=False)
    elapsed = time.time() - started

    if len(records) != len(plan):
        print(f"WARNING: {len(records)} loss values for {len(plan)} states; "
              "the date matching cannot be trusted", file=sys.stderr)
    rows = [{"start": dates[s], "target": dates[s + target_offset], "loss": v}
            for s, v in zip(plan, records)]
    for r in rows:
        print(f"  {r['target']}  {r['loss']:.4f}")
    print(f"=== mean {np.mean([r['loss'] for r in rows]):.4f} over {len(rows)} states, "
          f"{elapsed / max(len(rows), 1):.0f} s per state")

    shard = (args.shard or "1/1").replace("/", "-of-")
    out = args.out or Path.home() / "bris-runs" / "validation-loss" / f"{args.label}-{shard}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label,
        "weights": str(cfg.hardware.files.warm_start),
        "precision": "32" if args.cpu else str(cfg.training.precision),
        "every": args.every, "shard": args.shard,
        "seconds": elapsed, "rows": rows,
    }, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

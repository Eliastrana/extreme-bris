#!/usr/bin/env python3
"""Measure what the tail term weighs, on the states where it is meant to matter.

    scripts/measure_tail_weight.py --plan-only        # login node, seconds
    sbatch bris/slurm/measure_tail_weight.sbatch      # one H200, about twenty minutes
    scripts/measure_tail_weight.py --cpu --limit 1    # timing test on a CPU node

ON A CPU. The two H200s on this cluster can be held by week-long jobs, and
nothing else has the memory. A measurement does not train, so it can run on a
CPU node with terabytes of memory instead, in full precision. The ratio
between the two loss terms does not depend on the precision the forward pass
used, only on the states and the weights.

WHY. The tail arm adds a threshold-weighted CRPS term to Bris's own loss, at a
weight that is still the placeholder 100. The only reading so far came from the
first five batches of a probe: a raw value near 1e-4 against a total near 1.9,
which would make the term half a percent of the loss. Those batches were
October 2023 states picked by nothing, and the tail term is built to be zero on
ordinary days. What it weighs on an ordinary day says nothing about the weight.
What it weighs on the days it exists for does.

WHAT IT DOES. Takes the extreme states the ranking flagged inside the training
window, draws the same number of ordinary states from the same window at
random, and runs the untouched warm-start model over each through anemoi's own
validation step: same precision, same strategy, same scalers, the same three
members one at a time. Each part of the combined loss is recorded on its own,
so the ordinary CRPS and the raw tail term can be compared state by state.

THE TARGET, NOT THE INPUT. A ranking date is the valid time of a state. A
sample starting at index i uses the states before its target as input and
scores the state multi_step later, so each date is mapped to the sample whose
TARGET is that state. Mapping it to the sample that starts there would score
the six hours after the storm and measure the wrong thing.

WHAT IT REPORTS.
  weight      mean main term over mean tail term on extreme states: the weight
              that makes the two terms equal where the tail matters
  quiet days  the tail term on ordinary states, which should be near zero
  exceedance  how many grid points pass the threshold on extreme states; if
              almost none do, the threshold is too high and no weight means
              anything
  share       the tail's share of the loss over the whole window at that weight,
              estimated from the two strata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import _venv  # noqa: E402

_venv.ensure("anemoi", "torch")

import numpy as np  # noqa: E402

MAIN = "AlmostFairKernelCRPS"
TAIL = "TailWeightedKernelCRPS"


def load_ranking(files: list[Path], start: str, end: str):
    extreme, ordinary = [], []
    for f in files:
        data = json.loads(f.read_text())
        for r in data["states"]:
            if r.get("missing") or not (start <= r["date"][:10] <= end):
                continue
            (extreme if r.get("extreme") else ordinary).append(r)
    return extreme, ordinary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-name", default="finetune_tail")
    ap.add_argument("--rankings", nargs="+", type=Path, default=[
        Path.home() / "bris-runs/extremes/meps-year3.json",
        Path.home() / "bris-runs/extremes/meps-year2.json",
    ])
    ap.add_argument("--window-start", default="2023-10-01")
    ap.add_argument("--window-end", default="2025-03-31")
    ap.add_argument("--ordinary", type=int, default=None,
                    help="ordinary states to draw (default: as many as extreme)")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--cpu", action="store_true",
                    help="run on the CPU, in full precision, with one thread per allocated core")
    ap.add_argument("--limit", type=int, default=None,
                    help="measure only the first N planned states, for timing")
    ap.add_argument("--plan-only", action="store_true",
                    help="map dates to samples and stop, without a model or a card")
    ap.add_argument("--out", type=Path,
                    default=Path.home() / "bris-runs/tail-weight/measure-local.json")
    args = ap.parse_args()

    os.environ.setdefault("ANEMOI_BASE_SEED", "20260909")
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER", "32")

    # ---- which states -------------------------------------------------------
    extreme, ordinary_pool = load_ranking(args.rankings, args.window_start, args.window_end)
    rng = np.random.default_rng(args.seed)
    n_ord = args.ordinary if args.ordinary is not None else len(extreme)
    ordinary = [ordinary_pool[i] for i in
                sorted(rng.choice(len(ordinary_pool), size=min(n_ord, len(ordinary_pool)),
                                  replace=False))]
    print(f"=== {len(extreme)} extreme and {len(ordinary)} ordinary states "
          f"from {args.window_start} to {args.window_end}")

    # ---- the same config the tail arm trains with, validating over the window
    from xbris import patches

    if not args.plan_only:
        patches.apply()
    import _compose

    overrides = [
        f"dataloader.validation.start={args.window_start}",
        f"dataloader.validation.end={args.window_end}",
        # One worker keeps samples in the order they are listed here, which is
        # how each recorded loss is matched back to its date.
        "dataloader.num_workers.validation=1",
        "hardware.num_nodes=1",
        "hardware.num_gpus_per_node=1",
        "hardware.num_gpus_per_ensemble=1",
        "hardware.num_gpus_per_model=1",
    ]
    if args.plan_only or args.cpu:
        overrides.append("hardware.accelerator=cpu")
    if args.cpu:
        import torch

        threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
        torch.set_num_threads(threads)
        print(f"  CPU run, {threads} threads, full precision")
    cfg = _compose.compose(REPO / "bris/train", args.config_name, overrides)

    from anemoi.training.train.train import AnemoiTrainer

    trainer = AnemoiTrainer(cfg)
    dm = trainer.datamodule
    ds = dm.ds_valid
    multi_step = int(cfg.training.multistep_input)
    rel = [int(x) for x in ds.relative_date_indices]
    step = rel[1] - rel[0] if len(rel) > 1 else 1
    target_offset = rel[0] + multi_step * step

    dates = [str(d)[:19] for d in ds.data.dates]
    position = {d: i for i, d in enumerate(dates)}
    valid = {int(i) for i in ds.valid_date_indices}
    n_window = sum(1 for i in valid if 0 <= i + target_offset < len(dates))

    plan, dropped = [], []
    for kind, records in (("extreme", extreme), ("ordinary", ordinary)):
        for r in records:
            d = position.get(r["date"][:19])
            start = None if d is None else d - target_offset
            if start is None or start not in valid:
                dropped.append((kind, r["date"]))
                continue
            plan.append({"kind": kind, "date": r["date"][:19], "start": start,
                         "rank_frac_exceed": r.get("frac_exceed"),
                         "rank_max_mm": r.get("max_mm"),
                         "rank_area_ge_20": r.get("area_ge_20")})

    print(f"  multi_step {multi_step}, relative indices {rel}, target is start + {target_offset}")
    print(f"  {len(dates)} states in the window, {n_window} usable as a target")
    print(f"  planned {sum(p['kind'] == 'extreme' for p in plan)} extreme, "
          f"{sum(p['kind'] == 'ordinary' for p in plan)} ordinary")
    for kind, date in dropped:
        print(f"  dropped {kind} {date}: no usable sample has it as target")
    if args.limit:
        plan = plan[:args.limit]
        print(f"  limited to the first {len(plan)} planned state(s)")
    check = plan[0]
    print(f"  check: {check['date']} -> sample start {dates[check['start']]}, "
          f"target {dates[check['start'] + target_offset]}")
    if args.plan_only:
        return 0

    # ---- record each part of the loss ---------------------------------------
    import pytorch_lightning as pl

    model = trainer.model
    records: list[dict] = []
    current: dict = {}
    combined = model.loss
    combined_forward = combined.forward

    def record_combined(*a, **k):
        current.clear()
        out = combined_forward(*a, **k)
        records.append(dict(current))
        return out

    combined.forward = record_combined
    for part in combined.losses:
        name = type(part).__name__
        original = part.forward

        def record_part(*a, _orig=original, _name=name, _part=part, **k):
            value = _orig(*a, **k)
            current[_name] = float(value.detach().float().mean())
            if hasattr(_part, "tail_threshold"):
                column = a[1][..., _part.tail_index]
                current["n_exceed"] = int((column > _part.tail_threshold).sum())
                current["n_points"] = int(column.numel())
            return value

        part.forward = record_part

    ds.valid_date_indices = np.array([p["start"] for p in plan], dtype=np.int64)

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
        print(f"WARNING: {len(records)} loss evaluations for {len(plan)} planned states; "
              "the date matching below cannot be trusted", file=sys.stderr)
    rows = []
    for p, r in zip(plan, records):
        row = dict(p)
        row["main"] = r.get(MAIN)
        row["tail"] = r.get(TAIL)
        row["n_exceed"] = r.get("n_exceed")
        row["n_points"] = r.get("n_points")
        rows.append(row)

    # ---- the four numbers ---------------------------------------------------
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else float("nan")

    ext = [r for r in rows if r["kind"] == "extreme"]
    ordn = [r for r in rows if r["kind"] == "ordinary"]
    main_e, tail_e = mean(r["main"] for r in ext), mean(r["tail"] for r in ext)
    main_o, tail_o = mean(r["main"] for r in ordn), mean(r["tail"] for r in ordn)
    weight = main_e / tail_e if tail_e > 0 else float("inf")
    ratios = [r["main"] / r["tail"] for r in ext if r["tail"]]
    n_ext_window = len(extreme)
    main_w = (n_ext_window * main_e + (n_window - n_ext_window) * main_o) / n_window
    tail_w = (n_ext_window * tail_e + (n_window - n_ext_window) * tail_o) / n_window

    def share(w):
        return w * tail_w / (main_w + w * tail_w)

    exceed_frac = mean(r["n_exceed"] / r["n_points"] for r in ext if r["n_points"])

    summary = {
        "states_measured": len(rows),
        "seconds": round(elapsed, 1),
        "main_extreme": main_e, "tail_extreme": tail_e,
        "main_ordinary": main_o, "tail_ordinary": tail_o,
        "weight_equal_on_extremes": weight,
        "weight_median_of_state_ratios": float(np.median(ratios)) if ratios else None,
        "tail_over_main_ordinary": tail_o / main_o if main_o else None,
        "mean_points_over_threshold_extreme": mean(r["n_exceed"] for r in ext),
        "mean_fraction_over_threshold_extreme": exceed_frac,
        "mean_points_over_threshold_ordinary": mean(r["n_exceed"] for r in ordn),
        "window_states": n_window, "window_extreme": n_ext_window,
        "share_of_loss_at_weight": share(weight) if np.isfinite(weight) else None,
        "share_of_loss_at_placeholder_100": share(100.0),
    }

    print(f"\n{'date':20s} {'kind':9s} {'main':>9s} {'tail':>11s} {'ratio':>9s} {'>thr':>8s}")
    for r in rows:
        ratio = r["main"] / r["tail"] if r["tail"] else float("inf")
        print(f"{r['date']:20s} {r['kind']:9s} {r['main']:9.4f} {r['tail']:11.3e} "
              f"{ratio:9.1f} {r['n_exceed']:8d}")

    median = summary["weight_median_of_state_ratios"]
    print(f"\n=== weight that makes the terms equal on extreme states: {weight:.1f}"
          + (f"   (median of per-state ratios {median:.1f})" if median is not None else ""))
    if ordn:
        print(f"=== tail term on ordinary states: {tail_o:.3e}, "
              f"{summary['tail_over_main_ordinary']:.2e} of the main term")
    print(f"=== points over the threshold on extreme states: "
          f"{summary['mean_points_over_threshold_extreme']:.0f} on average, "
          f"{exceed_frac:.2e} of the grid; ordinary states "
          f"{summary['mean_points_over_threshold_ordinary']:.0f}")
    print(f"=== tail share of the loss over the window: "
          f"{summary['share_of_loss_at_weight']:.1%} at the measured weight, "
          f"{summary['share_of_loss_at_placeholder_100']:.2%} at the placeholder 100")
    print(f"=== {len(rows)} states in {elapsed / 60:.1f} min, "
          f"{elapsed / max(len(rows), 1):.0f} s per state")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "states": rows}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

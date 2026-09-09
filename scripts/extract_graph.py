#!/usr/bin/env python3
"""Pull the stretched-grid graph out of the checkpoint instead of rebuilding it.

    ~/bris-env/.venv/bin/python scripts/extract_graph.py \
        $BRIS_CKPT ~/bris-runs/graphs/n320_2p5k_7p10.pt

WHY. MET's training config sets `graph.overwrite: True`, which tells anemoi to
construct the graph from scratch. For this model that is 1,359,281 data nodes
and 261,634 hidden nodes across three edge types, and building it costs hours
and a great deal of memory.

It does not have to be built. The inference checkpoint already carries the
exact graph the model was trained with, as `graph_data`. Extracting it and
pointing the config at the file gives a graph that is correct by construction
rather than one that merely ought to match.

THE TRAINING CHECKPOINT CANNOT BE READ HERE. It was written by a development
build of anemoi-core and unpickles into `anemoi.models.migrations`, a module
that does not exist in the released anemoi-models 0.8.1 that MET's own
pyproject pins. The inference checkpoint loads cleanly with the pinned
versions, and for fine-tuning with `load_weights_only: True` the optimiser
state in the training checkpoint is not needed anyway.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    import torch

    if not args.checkpoint.exists():
        print(f"ERROR: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    print(f"reading {args.checkpoint.name} ...", file=sys.stderr)
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    graph = getattr(model, "graph_data", None)
    if graph is None:
        print("ERROR: this checkpoint carries no graph_data", file=sys.stderr)
        return 2

    for nt in graph.node_types:
        print(f"  {nt:8s} {graph[nt].num_nodes:>9,} nodes")
    for et in graph.edge_types:
        n = graph[et].num_edges if hasattr(graph[et], "num_edges") else "?"
        print(f"  {'->'.join(et):40s} {n:>12,} edges")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, args.out)
    size = args.out.stat().st_size / 1024**2
    print(f"\nwrote {args.out}  ({size:.0f} MB)")
    print("Set graph.overwrite: False and point hardware.files.graph at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

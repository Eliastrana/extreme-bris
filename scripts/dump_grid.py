#!/usr/bin/env python3
"""Extract the grid definition from a Bris checkpoint.

Because `switch_graph: null`, the model uses the graph stored inside the
checkpoint, and an Anemoi graph carries explicit coordinates for every node.
The grid the input dataset must match is therefore already in the checkpoint —
it does not have to be obtained from MET or reverse-engineered from the loss
config.

    cd $BRIS_ENV_DIR && uv run python <repo>/scripts/dump_grid.py \
        ~/bris-models/bris-forecaster/bris-crpsfft_inference.ckpt \
        --npz ~/bris-runs/grid.npz

Prints, per node set: count, bounding box, and whether the coordinates look
like a regular lat/lon block or a reduced Gaussian grid. With --npz the raw
coordinates are saved for building or validating a dataset against.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


GRAPH_ATTRS = ("graph_data", "graph", "_graph_data", "_graph")


def find_graph(obj, _depth=0, _seen=None):
    """Locate the graph inside a loaded checkpoint.

    Bris inference checkpoints do not unpickle to a dict — torch.load returns an
    AnemoiModelInterface instance — so attributes have to be searched as well as
    mapping keys. Named attributes are tried first; the general scan skips
    nn.Module children, which are large and never hold the graph.
    """
    if _depth > 6:
        return None
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return None
    _seen.add(id(obj))

    if hasattr(obj, "node_types"):          # torch_geometric HeteroData
        return obj

    for attr in GRAPH_ATTRS:
        val = getattr(obj, attr, None) if not isinstance(obj, dict) else obj.get(attr)
        if val is not None:
            hit = find_graph(val, _depth + 1, _seen)
            if hit is not None:
                return hit

    if isinstance(obj, dict):
        for v in obj.values():
            hit = find_graph(v, _depth + 1, _seen)
            if hit is not None:
                return hit
        return None

    state = getattr(obj, "__dict__", None)
    if isinstance(state, dict):
        try:
            import torch.nn as nn
        except Exception:
            nn = None
        for k, v in state.items():
            if k.startswith("_parameters") or k.startswith("_buffers"):
                continue
            if nn is not None and isinstance(v, nn.Module):
                continue
            hit = find_graph(v, _depth + 1, _seen)
            if hit is not None:
                return hit
    return None


def describe(name, coords):
    """coords: (N, 2) array of [latitude, longitude] in radians or degrees."""
    import numpy as np

    n = len(coords)
    lat, lon = coords[:, 0], coords[:, 1]
    # Anemoi stores radians; convert if the range makes that unambiguous.
    if float(np.abs(lat).max()) <= math.pi / 2 + 1e-6:
        lat, lon = np.degrees(lat), np.degrees(lon)
        units = "radians -> degrees"
    else:
        units = "degrees"

    print(f"--- {name}: {n:,} nodes  ({units})")
    print(f"      lat  {lat.min():8.3f} .. {lat.max():8.3f}")
    print(f"      lon  {lon.min():8.3f} .. {lon.max():8.3f}")

    uniq_lat = len(np.unique(np.round(lat, 4)))
    uniq_lon = len(np.unique(np.round(lon, 4)))
    print(f"      unique lats {uniq_lat:,}   unique lons {uniq_lon:,}")

    if uniq_lat * uniq_lon == n:
        print(f"      -> regular grid: {uniq_lat} x {uniq_lon}")
    elif uniq_lat < n / 10:
        print(f"      -> looks reduced Gaussian: {uniq_lat} latitude rows")
    else:
        print("      -> irregular / projected grid; use the raw coordinates")
    print()
    return lat, lon


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--npz", type=Path, help="save raw coordinates per node set")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc.name} not available — this is not the Bris environment.",
              file=sys.stderr)
        print("Run it from the built env, with an explicit path:", file=sys.stderr)
        print("  cd ~/bris-env && uv run python <repo>/scripts/dump_grid.py <ckpt>",
              file=sys.stderr)
        print("If you used $BRIS_ENV_DIR and landed in $HOME, the variable was unset:",
              file=sys.stderr)
        print("  source ~/extreme-bris/scripts/env.sh", file=sys.stderr)
        return 3

    if not args.checkpoint.exists():
        print(f"ERROR: no such file: {args.checkpoint}", file=sys.stderr)
        return 1

    print(f"=== {args.checkpoint.name}\n")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)

    graph = find_graph(ckpt)
    if graph is None:
        print("No graph found in the checkpoint.\n", file=sys.stderr)
        print(f"Loaded object: {type(ckpt)}", file=sys.stderr)
        if isinstance(ckpt, dict):
            print("Keys:", list(ckpt.keys())[:30], file=sys.stderr)
        else:
            attrs = sorted(k for k in vars(ckpt)) if hasattr(ckpt, "__dict__") else []
            print("Attributes:", attrs[:40], file=sys.stderr)
            pub = [a for a in dir(ckpt) if not a.startswith("_")]
            print("Public names:", pub[:40], file=sys.stderr)
        print("\nPaste this output — it names where the graph actually lives.",
              file=sys.stderr)
        return 2

    try:
        node_types = list(graph.node_types)
    except Exception:
        node_types = [k for k in graph.keys()] if hasattr(graph, "keys") else []

    print(f"node sets: {', '.join(map(str, node_types))}\n")

    out = {}
    for nt in node_types:
        store = graph[nt]
        coords = None
        for attr in ("x", "coords", "latlons", "lat_lon"):
            val = getattr(store, attr, None) if not isinstance(store, dict) else store.get(attr)
            if val is not None and hasattr(val, "shape") and val.ndim == 2 and val.shape[1] >= 2:
                coords = np.asarray(val.detach().cpu() if hasattr(val, "detach") else val)
                break
        if coords is None:
            print(f"--- {nt}: no coordinate attribute found\n")
            continue
        lat, lon = describe(str(nt), coords[:, :2])
        out[f"{nt}_lat"], out[f"{nt}_lon"] = lat, lon

    if args.npz and out:
        np.savez_compressed(args.npz, **out)
        print(f"coordinates written to {args.npz}")

    print("The data node set is the grid an input dataset must reproduce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

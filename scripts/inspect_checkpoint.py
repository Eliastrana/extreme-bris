#!/usr/bin/env python3
"""Dump the Anemoi metadata embedded in a Bris checkpoint.

Anemoi stores the variable list, its index order, and the normalisation
statistics inside the checkpoint. Those define exactly what an input dataset
must contain and in what order, so this is the authoritative answer to
"what do I have to build?" — more so than the YAML configs, which only say
what MET happened to run with.

    python scripts/inspect_checkpoint.py path/to/bris-crpsfft_inference.ckpt
    python scripts/inspect_checkpoint.py <ckpt> --json out.json

Run it inside the built environment so anemoi and torch are importable:

    cd $BRIS_ENV_DIR && uv run python <repo>/scripts/inspect_checkpoint.py <ckpt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metadata(path: Path) -> dict:
    """Prefer anemoi's own reader; fall back to a raw torch load."""
    try:
        from anemoi.utils.checkpoints import load_metadata as _load

        return _load(str(path))
    except Exception as exc:  # noqa: BLE001 - we genuinely want any failure here
        print(f"  (anemoi.utils.checkpoints unavailable or failed: {exc})", file=sys.stderr)
        print("  falling back to torch.load — this reads the whole file", file=sys.stderr)

    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("metadata", "hyper_parameters", "config"):
            if key in obj:
                return {key: obj[key]}
        return {"top_level_keys": sorted(obj.keys())}
    return {"unrecognised_type": type(obj).__name__}


def find(node, *names):
    """Depth-first search for the first dict value under any of `names`."""
    if isinstance(node, dict):
        for n in names:
            if n in node:
                return node[n]
        for v in node.values():
            hit = find(v, *names)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for v in node:
            hit = find(v, *names)
            if hit is not None:
                return hit
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--json", type=Path, help="also write the full metadata as JSON")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: no such file: {args.checkpoint}", file=sys.stderr)
        return 1

    size_gb = args.checkpoint.stat().st_size / 1e9
    print(f"=== {args.checkpoint.name}  ({size_gb:.2f} GB)\n")

    meta = load_metadata(args.checkpoint)

    # --- variables, in index order -----------------------------------------
    name_to_index = find(meta, "name_to_index")
    if isinstance(name_to_index, dict) and name_to_index:
        ordered = sorted(name_to_index.items(), key=lambda kv: kv[1])
        print(f"--- variables ({len(ordered)}), in model index order")
        for name, idx in ordered:
            print(f"  {idx:3d}  {name}")
        print()
    else:
        variables = find(meta, "variables")
        if variables:
            print(f"--- variables ({len(variables)})")
            print("  " + ", ".join(map(str, variables)))
            print()

    # --- which of those are diagnostic vs prognostic ------------------------
    for key in ("diagnostic", "forcing", "prognostic"):
        vals = find(meta, key)
        if vals:
            print(f"--- {key} ({len(vals)}): {', '.join(map(str, vals))}\n")

    # --- normalisation statistics ------------------------------------------
    stats = find(meta, "statistics")
    if isinstance(stats, dict):
        keys = [k for k in ("mean", "stdev", "std", "minimum", "maximum") if k in stats]
        if keys:
            n = len(stats[keys[0]])
            print(f"--- normalisation statistics: {', '.join(keys)}  (length {n})")
            if isinstance(name_to_index, dict):
                ordered = sorted(name_to_index.items(), key=lambda kv: kv[1])
                print(f"  {'variable':<18}" + "".join(f"{k:>14}" for k in keys))
                for name, idx in ordered:
                    if idx >= n:
                        continue
                    row = "".join(f"{float(stats[k][idx]):>14.5g}" for k in keys)
                    print(f"  {name:<18}{row}")
            print()

    # --- grid / dataset shape ----------------------------------------------
    for key in ("resolution", "frequency", "grid_indices", "shape", "field_shape"):
        val = find(meta, key)
        if val is not None and not isinstance(val, (dict, list)):
            print(f"--- {key}: {val}")

    if args.json:
        args.json.write_text(json.dumps(meta, indent=2, default=str))
        print(f"\nFull metadata written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

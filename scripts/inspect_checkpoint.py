#!/usr/bin/env python3
"""Dump the Anemoi metadata embedded in a Bris checkpoint.

Anemoi stores the variable list, the index groups (prognostic / forcing /
diagnostic) and the normalisation statistics inside the checkpoint. Those define
exactly what an input dataset must contain and in what order, so this is the
authoritative answer to "what do I have to build?" — more so than the YAML
configs, which only record what MET happened to run with.

    python scripts/inspect_checkpoint.py path/to/bris-crpsfft_inference.ckpt
    python scripts/inspect_checkpoint.py <ckpt> --json out.json

Run inside the built environment so anemoi and torch are importable:

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
    except Exception as exc:  # noqa: BLE001
        print(f"  (anemoi.utils.checkpoints unavailable: {exc})", file=sys.stderr)
        print("  falling back to torch.load — reads the whole file", file=sys.stderr)

    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("metadata", "hyper_parameters", "config"):
            if key in obj:
                return {key: obj[key]}
        return {"top_level_keys": sorted(obj.keys())}
    return {"unrecognised_type": type(obj).__name__}


def dig(node, *path):
    """Follow an explicit key path. Returns None if any hop is missing.

    Preferred over a blind search: a bare search for "variables" happily finds
    the bounding config's variable list and reports two entries instead of 98.
    """
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def search(node, name, _depth=0):
    """Last-resort depth-first search, used only when the known paths miss."""
    if _depth > 12:
        return None
    if isinstance(node, dict):
        if name in node:
            return node[name]
        for v in node.values():
            hit = search(v, name, _depth + 1)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for v in node:
            hit = search(v, name, _depth + 1)
            if hit is not None:
                return hit
    return None


def variable_names(meta) -> list[str]:
    """The full ordered variable list."""
    for path in (("dataset", "variables"), ("data_indices", "variables")):
        v = dig(meta, *path)
        if isinstance(v, list) and len(v) > 20:
            return [str(x) for x in v]

    n2i = dig(meta, "dataset", "name_to_index") or search(meta, "name_to_index")
    if isinstance(n2i, dict) and len(n2i) > 20:
        return [n for n, _ in sorted(n2i.items(), key=lambda kv: kv[1])]

    v = search(meta, "variables")
    return [str(x) for x in v] if isinstance(v, list) else []


def index_groups(meta) -> dict[str, list[int]]:
    """prognostic / forcing / diagnostic index lists, from data_indices."""
    out: dict[str, list[int]] = {}
    for section in ("input", "output"):
        base = dig(meta, "data_indices", "data", section)
        if not isinstance(base, dict):
            continue
        for group, idx in base.items():
            if isinstance(idx, list) and all(isinstance(i, int) for i in idx):
                out[f"{section}.{group}"] = idx
    if out:
        return out
    for group in ("prognostic", "forcing", "diagnostic"):
        idx = search(meta, group)
        if isinstance(idx, list):
            out[group] = idx
    return out


def resolve(idx_list, names):
    """Map indices to names where possible; pass names through unchanged."""
    resolved = []
    for i in idx_list:
        if isinstance(i, int) and 0 <= i < len(names):
            resolved.append(names[i])
        else:
            resolved.append(str(i))
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--json", type=Path, help="also write the full metadata as JSON")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: no such file: {args.checkpoint}", file=sys.stderr)
        return 1

    print(f"=== {args.checkpoint.name}  ({args.checkpoint.stat().st_size / 1e9:.2f} GB)\n")

    meta = load_metadata(args.checkpoint)
    names = variable_names(meta)

    if names:
        print(f"--- variables ({len(names)}), in model index order")
        for i, n in enumerate(names):
            print(f"  {i:3d}  {n}")
        print()
    else:
        print("--- variables: NOT FOUND (inspect the --json dump by hand)\n")

    groups = index_groups(meta)
    for group, idx in sorted(groups.items()):
        shown = resolve(idx, names) if names else [str(i) for i in idx]
        print(f"--- {group} ({len(idx)}): {', '.join(shown)}\n")

    if groups and names:
        inp = groups.get("input.full") or groups.get("input.prognostic")
        print("--- summary")
        for g in ("input.prognostic", "input.forcing", "output.diagnostic",
                  "prognostic", "forcing", "diagnostic"):
            if g in groups:
                print(f"  {g:<20} {len(groups[g]):3d}")
        if inp:
            print(f"  {'model inputs':<20} {len(inp):3d}")
        print()

    # multistep_input decides whether one forecast needs one analysis time or two.
    for key in ("multistep_input", "multistep"):
        val = dig(meta, "config", "training", key) or search(meta, key)
        if isinstance(val, int):
            print(f"--- {key}: {val}   "
                  f"({'t0 only' if val == 1 else f'{val} states: t0 and {val - 1} earlier'})")
            break
    else:
        print("--- multistep_input: NOT FOUND — assume the Anemoi default of 2 and verify")

    stats = dig(meta, "dataset", "statistics") or search(meta, "statistics")
    if isinstance(stats, dict):
        keys = [k for k in ("mean", "stdev", "std", "minimum", "maximum") if k in stats]
        if keys and names:
            n = len(stats[keys[0]])
            print(f"\n--- normalisation statistics: {', '.join(keys)} (length {n})")
            print(f"  {'variable':<18}" + "".join(f"{k:>14}" for k in keys))
            for i, nm in enumerate(names):
                if i >= n:
                    break
                row = "".join(f"{float(stats[k][i]):>14.5g}" for k in keys)
                print(f"  {nm:<18}{row}")

    for key in ("resolution", "frequency"):
        val = dig(meta, "dataset", key) or search(meta, key)
        if val is not None and not isinstance(val, (dict, list)):
            print(f"\n--- {key}: {val}")

    if args.json:
        args.json.write_text(json.dumps(meta, indent=2, default=str))
        print(f"\nFull metadata written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

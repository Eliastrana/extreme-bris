#!/usr/bin/env python3
"""Say what was holding the card when it ran out.

    scripts/memory_snapshot_summary.py ~/bris-runs/oom.pickle
    scripts/memory_snapshot_summary.py ~/bris-runs/oom.pickle --depth 4 --top 30

Reads a snapshot written by xbris.memory and groups every live allocation by
the code that made it, largest first. The grouping key is the innermost few
frames inside anemoi or xbris, because a frame deep in torch says only that a
tensor was made, not which part of the model wanted it.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("torch")

GiB = 2 ** 30
OURS = ("anemoi/", "xbris/", "torch/nn/modules/", "torch_geometric/")


def describe(frames, depth: int) -> str:
    picked = []
    for fr in frames or []:
        name = str(fr.get("filename", ""))
        if any(tag in name for tag in OURS):
            short = name.split("site-packages/")[-1]
            picked.append(f"{short}:{fr.get('line')} {fr.get('name')}")
            if len(picked) >= depth:
                break
    return "  <-  ".join(picked) if picked else "(no model frames recorded)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    with open(args.snapshot.expanduser(), "rb") as f:
        snap = pickle.load(f)

    groups: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0, 0])
    total = reserved = 0
    for seg in snap.get("segments", []):
        reserved += seg.get("total_size", 0)
        for block in seg.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            size = block.get("requested_size") or block.get("size", 0)
            total += size
            g = groups[describe(block.get("frames"), args.depth)]
            g[0] += size
            g[1] += 1
            g[2] = max(g[2], size)

    print(f"live allocations {total / GiB:7.2f} GiB   reserved {reserved / GiB:7.2f} GiB\n")
    print(f"{'GiB':>8} {'count':>6} {'largest':>8}  made by")
    for key, (size, count, largest) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:args.top]:
        print(f"{size / GiB:8.2f} {count:6d} {largest / GiB:8.2f}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

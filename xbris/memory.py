"""Record every CUDA allocation, and write a snapshot the moment memory runs out.

    XBRIS_MEMORY_SNAPSHOT=~/bris-runs/oom.pickle sbatch bris/slurm/probe_memory.sbatch
    scripts/memory_snapshot_summary.py ~/bris-runs/oom.pickle

WHY. Fitting this model on one card has gone through chunking, offloading and
two layers of recomputation, and each round was chosen by reading a traceback
and inferring where the memory went. The last inference was wrong twice over:
anemoi already recomputes each whole mapper, so wrapping the mapper block again
did nothing, and the 10.37 GB allocation that fails is the same size at sixteen
and at thirty-two chunks, so it is not the per-chunk tensor it looked like.

A traceback says where the last allocation failed. It says nothing about the
130 GB that were already held. PyTorch can record every allocation together
with the Python stack that made it, and hand over the whole picture when an
allocation fails. That turns the next attempt from a guess into a reading.

Costs some speed and host memory while armed, so it is off unless asked for.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def record_until_oom(path: str | Path, max_entries: int = 200_000) -> None:
    import torch

    if not torch.cuda.is_available():
        print("xbris: no CUDA here, memory snapshot not armed", file=sys.stderr)
        return

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=max_entries)

    def observer(device, alloc, device_alloc, device_free):
        try:
            with open(path, "wb") as f:
                pickle.dump(torch.cuda.memory._snapshot(), f)
            print(f"xbris: out of memory asking for {alloc / 2**30:.2f} GiB with "
                  f"{device_free / 2**30:.2f} GiB free; snapshot written to {path}",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"xbris: snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    torch._C._cuda_attach_out_of_memory_observer(observer)
    print(f"xbris: recording CUDA allocations; snapshot on OOM goes to {path}",
          file=sys.stderr)

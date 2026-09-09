#!/usr/bin/env python3
"""anemoi-datasets create, with every 0.5.24 patch applied.

    ~/bris-data-env/bin/python scripts/anemoi_create.py <recipe.yaml> <out.zarr>

The patches themselves live in anemoi_patches.py, with the reasoning for each.
They used to be split across this file and build_dataset.py, which meant the
set you got depended on which wrapper ran - and a MARS build submitted through
the slurm job died on an accumulations bug that the other wrapper had fixed
days earlier. Both wrappers now apply the same list.

None of them change what is computed.
"""

from __future__ import annotations

# Hand over to an interpreter that has these, if this one does not. These
# scripts are run by path, so the shebang picks up whatever python3 is on
# PATH, and on the login node that one has none of the stack.
import sys as _sys, pathlib as _pathlib  # noqa: E401
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure('anemoi')


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anemoi_patches import apply_all                      # noqa: E402


def main() -> int:
    for line in apply_all():
        print(f"[patch] {line}", file=sys.stderr)

    from anemoi.datasets.__main__ import main as anemoi_main

    sys.argv = ["anemoi-datasets", "create", *sys.argv[1:]]
    return anemoi_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())

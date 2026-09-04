#!/usr/bin/env python3
"""Build an anemoi dataset, working around a bug in the class=od accumulations.

    ~/bris-data-env/bin/python scripts/build_dataset.py \
        bris/configs/od_n320_20250401.yaml \
        $BRIS_DATA_DIR/od-an-n320-2025-6h-v1.zarr

This is `anemoi-datasets create` with one method added at runtime. Building the
global side from the operational analysis (class od) instead of ERA5 dies in
anemoi-datasets 0.5.24 with:

    AttributeError: 'AccumulationFromLastStep' object has no attribute
                    'adjust_steps'

WHY IT ONLY BITES class od
--------------------------
`Accumulation.add` calls `self.adjust_steps` only when the GRIB reports
startStep == endStep, i.e. when the encoding gives one step rather than a
window. ERA5 fields come back with a real window, so the call never happens
and the missing method never shows - which is why the ERA5 build succeeded and
this one did not.

Of the three Accumulation subclasses, AccumulationFromStart and
AccumulationFromLastReset both define adjust_steps. AccumulationFromLastStep
does not, and it is the one the dispatcher picks for od: KWARGS holds
`("od", "oper"): dict(patch=_scda)` with no data_accumulation_period, so it
falls back to user_accumulation_period (6), which selects FromLastStep.

The recipe cannot steer this. Everything in the recipe's accumulations block
goes into the MARS request; data_accumulation_period is only reachable through
that hard-coded table.

WHAT THE PATCH RESTORES
-----------------------
AccumulationFromLastStep.compute already asserts

    endStep - startStep == self.frequency

so the window the class expects is not a guess: it is stated by the class
itself. adjust_steps just has to widen a collapsed step into that window.

This is a runtime patch rather than an edit to site-packages so that the
workaround travels with the repo and is visible in review, instead of living
in an environment nobody can reconstruct.

VERIFY THE RESULT. A patch to accumulation logic is exactly the kind of change
that produces a dataset which loads fine and carries quietly wrong
precipitation, so check tp against an independent source before building
anything on top of it:

    scripts/check_accumulations.py <zarr>
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def patch_accumulations() -> str:
    from anemoi.datasets.create.sources import accumulations as acc

    cls = acc.AccumulationFromLastStep
    if hasattr(cls, "adjust_steps"):
        return f"{cls.__name__}.adjust_steps already present - no patch applied"

    def adjust_steps(self, startStep: int, endStep: int):
        # Only ever called for a collapsed step (startStep == endStep). The
        # window is the one compute() asserts on.
        assert startStep == endStep, (startStep, endStep)
        return (endStep - self.frequency, endStep)

    cls.adjust_steps = adjust_steps
    return f"patched {cls.__name__}.adjust_steps (window = endStep - frequency)"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    print(f"[patch] {patch_accumulations()}", file=sys.stderr)

    script = Path(sys.executable).with_name("anemoi-datasets")
    if not script.exists():
        print(f"ERROR: {script} not found - wrong environment?", file=sys.stderr)
        return 1

    sys.argv = ["anemoi-datasets", "create", *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

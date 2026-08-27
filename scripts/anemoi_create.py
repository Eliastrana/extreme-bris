#!/usr/bin/env python3
"""anemoi-datasets create, with numpy integers made JSON-serialisable.

anemoi-datasets 0.5.24 writes dataset metadata through `json_tidy`, which
handles np.float32 and np.float64 but no numpy integer type at all. MEPS carries
int16 coordinate values, so the build gets all the way through reading the data
and computing statistics, then dies writing metadata:

    TypeError: np.int16(0) is not JSON serializable <class 'numpy.int16'>

That is a gap in the package rather than in the recipe — a bug worth reporting
upstream. Until then this wraps the CLI and widens json_tidy to cover numpy
integers, booleans and arrays, delegating everything else to the original.

It also fixes a second version-skew failure in the same run: patch.py's
fix_provenance calls .startswith on every module_versions value, while newer
anemoi-utils records dicts there, giving

    AttributeError: 'dict' object has no attribute 'startswith'

Both patches only touch how metadata is serialised, never what is computed.

    ~/bris-data-env/bin/python scripts/anemoi_create.py <recipe.yaml> <out.zarr>

Nothing else is changed: the patch only affects how already-computed metadata is
serialised, not what is computed.
"""

from __future__ import annotations

import sys


def main() -> int:
    import numpy as np
    import anemoi.datasets.create as adc

    original = adc.json_tidy

    def json_tidy(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return original(o)

    # update_metadata resolves json_tidy from module globals at call time, so
    # rebinding the module attribute is enough.
    adc.json_tidy = json_tidy

    # Same class of problem in the patch step: fix_provenance assumes every
    # module_versions value is a string and calls .startswith on it, while newer
    # anemoi-utils records dicts. Coerce them before the original runs, rather
    # than reimplementing the function.
    from anemoi.datasets.create import patch as adp

    _fix_provenance = adp.fix_provenance

    def fix_provenance(provenance):
        for section in ("module_versions", "git_versions"):
            versions = provenance.get(section)
            if isinstance(versions, dict) and section == "module_versions":
                for k, v in list(versions.items()):
                    if not isinstance(v, str):
                        versions[k] = str(v)
        return _fix_provenance(provenance)

    adp.fix_provenance = fix_provenance
    if "provenance" in getattr(adp, "FIXES", {}):
        adp.FIXES["provenance"] = fix_provenance
    for key in list(getattr(adp, "FIXES", {})):
        if adp.FIXES[key] is _fix_provenance:
            adp.FIXES[key] = fix_provenance

    from anemoi.datasets.__main__ import main as anemoi_main

    sys.argv = ["anemoi-datasets", "create", *sys.argv[1:]]
    return anemoi_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())

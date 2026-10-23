"""Join models from several hits_and_calibration.py caches into one set of cases.

Every test-year cache holds the same 259,178 station-days in the same order,
because they were all built by the same strict common-case rule from the same
observations. But "the same" is checked, not assumed: observations, dates and
station indices must agree exactly across files, or the join refuses. A model
is taken from the first file that has it.

    cases = load(["a-vs-pavo/cases.npz", "calibration-test/with-baseline-cases.npz"],
                 ["baseline", "control2000", "tailA2000", "tailC2000"])
    cases["obs"], cases["dates"], cases["station"], cases["meps"], cases["models"]["tailA2000"]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LEAD = 30


def load(paths: list[str | Path], labels: list[str]) -> dict:
    out: dict = {"models": {}}
    for path in paths:
        path = Path(path).expanduser()
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            shared = {
                "obs": data[f"obs_{LEAD}"],
                "dates": data[f"dates_{LEAD}"].astype("datetime64[D]"),
                "meps": data[f"meps_{LEAD}"] if f"meps_{LEAD}" in keys else None,
                "station": data[f"station_{LEAD}"] if f"station_{LEAD}" in keys else None,
                "stations": data["stations"] if "stations" in keys else None,
            }
            if "obs" not in out:
                out.update(shared)
            else:
                for name in ("obs", "dates", "station", "meps"):
                    a, b = out[name], shared[name]
                    if a is None or b is None:
                        if out[name] is None:
                            out[name] = b
                        continue
                    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                        raise SystemExit(f"{path}: {name} differs from the first file; these caches are not the same cases")
                if out["stations"] is None:
                    out["stations"] = shared["stations"]
            for label in labels:
                key = f"model_{label}_{LEAD}"
                if label not in out["models"] and key in keys:
                    out["models"][label] = data[key]
    missing = [l for l in labels if l not in out["models"]]
    if missing:
        raise SystemExit(f"no cache holds {', '.join(missing)}")
    if out["station"] is None or out["stations"] is None:
        raise SystemExit("none of the caches records stations; rebuild one with the current hits_and_calibration.py")
    return out

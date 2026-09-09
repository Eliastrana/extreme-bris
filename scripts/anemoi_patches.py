#!/usr/bin/env python3
"""Every runtime patch anemoi-datasets 0.5.24 needs, in one place.

    from anemoi_patches import apply_all
    for line in apply_all():
        print(f"[patch] {line}", file=sys.stderr)

WHY THIS MODULE EXISTS. The patches used to live in two wrappers -
anemoi_create.py had three, build_dataset.py had a different one - and which
ones you got depended on which wrapper happened to run. The slurm job calls
anemoi_create.py, so a MARS build submitted through it died on the
accumulations bug that build_dataset.py had fixed days earlier. Ten minutes of
MARS retrieval thrown away because a fix existed in the wrong file.

One module, applied by both wrappers, so a fix cannot be present in one path
and missing from another.

None of these change what is computed. Two are metadata serialisation, one
restores a method the class contract already specifies, and one makes a
function pair its inputs the way its own name says it does.
"""

from __future__ import annotations

# Hand over to an interpreter that has these, if this one does not. These
# scripts are run by path, so the shebang picks up whatever python3 is on
# PATH, and on the login node that one has none of the stack.
import sys as _sys, pathlib as _pathlib  # noqa: E401
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure('anemoi')


from typing import List


def patch_json_tidy() -> str:
    """Teach the metadata writer about numpy scalars.

    0.5.24's json_tidy handles np.float32/64 and no integer type at all. MEPS
    carries int16 coordinates, so a build reads every field, computes every
    statistic, then dies writing metadata with
    `TypeError: np.int16(0) is not JSON serializable`.
    """
    import numpy as np
    import anemoi.datasets.create as adc

    if getattr(adc.json_tidy, "_widened", False):
        return "json_tidy already widened"

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

    json_tidy._widened = True
    # update_metadata resolves json_tidy from module globals at call time, so
    # rebinding the module attribute is enough.
    adc.json_tidy = json_tidy
    return "widened json_tidy to numpy scalars"


def patch_fix_provenance() -> str:
    """Coerce module_versions values to strings before fix_provenance reads them.

    It calls .startswith on every value; newer anemoi-utils records dicts,
    giving `AttributeError: 'dict' object has no attribute 'startswith'` at
    step 7 of 9 - after the data and statistics are already written.
    """
    from anemoi.datasets.create import patch as adp

    if getattr(adp.fix_provenance, "_coerces", False):
        return "fix_provenance already coercing"

    original = adp.fix_provenance

    def fix_provenance(provenance):
        versions = provenance.get("module_versions")
        if isinstance(versions, dict):
            for k, v in list(versions.items()):
                if not isinstance(v, str):
                    versions[k] = str(v)
        return original(provenance)

    fix_provenance._coerces = True
    adp.fix_provenance = fix_provenance
    for key, value in list(getattr(adp, "FIXES", {}).items()):
        if value is original:
            adp.FIXES[key] = fix_provenance
    return "fix_provenance coerces module_versions to str"


def patch_accumulations() -> str:
    """Restore the missing adjust_steps on AccumulationFromLastStep.

    Accumulation.add calls self.adjust_steps only when the GRIB reports
    startStep == endStep, i.e. a collapsed step. ERA5 fields carry a real
    window so it never fires there; class od fields do not, so `tp`, `ssrd`
    and `strd` bring the whole build down.

    Of the three Accumulation subclasses only this one lacks the method, and
    it is the one the dispatcher selects for od. The window is not a guess:
    compute() in the same class asserts endStep - startStep == self.frequency.
    """
    from anemoi.datasets.create.sources import accumulations as acc

    cls = acc.AccumulationFromLastStep
    if hasattr(cls, "adjust_steps"):
        return f"{cls.__name__}.adjust_steps already present"

    def adjust_steps(self, startStep: int, endStep: int):
        assert startStep == endStep, (startStep, endStep)
        return (endStep - self.frequency, endStep)

    cls.adjust_steps = adjust_steps
    return f"patched {cls.__name__}.adjust_steps (window = endStep - frequency)"


def patch_iterate_patterns() -> str:
    """Give each resolved URL only the dates that produced it.

    The shipped version yields every resolved path paired with EVERY date. With
    a per-date URL template spanning two MEPS cycles, each cycle file is then
    searched for both valid times: the 00Z file is asked for 18:00, which
    begins six hours before that file exists, and 00:00 is found twice - as an
    analysis in one file and a +6 h forecast in the other. The build resolves
    that collision by writing NaN for everything.

    Pinning one cycle avoids the collision but makes t0 a forecast rather than
    an analysis, which corrupts exactly the tendency the second state carries.
    """
    from anemoi.datasets.create.sources import patterns as pat
    from anemoi.datasets.create.sources import xarray_support
    from earthkit.data.utils.patterns import Pattern

    if getattr(pat.iterate_patterns, "_per_date", False):
        return "iterate_patterns already per-date"

    expand = pat._expand

    def iterate_patterns(path, dates, **kwargs):
        given = path if isinstance(path, list) else [path]
        iso = [d.isoformat() for d in dates]
        for one in given:
            if not iso:
                for q in expand(Pattern(one).substitute(allow_extra=True, **kwargs)):
                    yield q, []
                continue
            buckets: dict[str, list[str]] = {}
            for d in iso:
                kw = dict(kwargs)
                kw["date"] = [d]
                for q in expand(Pattern(one).substitute(allow_extra=True, **kw)):
                    buckets.setdefault(q, []).append(d)
            for q, ds in buckets.items():
                yield q, ds

    iterate_patterns._per_date = True
    pat.iterate_patterns = iterate_patterns
    # xarray_support did `from ...patterns import iterate_patterns`, so it holds
    # its own reference; rebinding the source module alone would not reach it.
    xarray_support.iterate_patterns = iterate_patterns
    return "iterate_patterns gives each URL only its own dates"


# ECMWF retired the short cut-off stream on this date: the 06Z and 18Z cycles
# moved from `scda` into `oper`. Verified against MARS itself, one request per
# stream and month, asking only for a single grid box:
#
#   scda 18Z, April 2026     30 of 30      scda 06Z, May 2026   1-11 May
#   scda 18Z, May 2026       1-11 May      scda 18Z, Aug 2026   nothing
#   oper 06Z, May 2026      12-31 May      oper 18Z, May 2026  12-31 May
#   oper 06Z, Aug 2026       31 of 31      oper 18Z, Aug 2026   31 of 31
#
# The two streams meet exactly, with no overlap and no gap.
SCDA_RETIRED = 20260512


def patch_scda_retirement() -> str:
    """Choose the accumulation stream by date, not by time alone.

    anemoi's _scda sends every 06Z and 18Z accumulation request to the `scda`
    stream. That was right until 2026-05-12 and wrong after it, which is what
    killed a ten-hour build: MARS answered `Expected 90, got 33` for May 2026,
    because only the first eleven days of that month are in scda.

    The failure looked like missing data and was not. Declaring those months
    missing - the obvious reading of the error - would have thrown away four
    months of perfectly good analysis for a stream rename.

    The patch is date-aware because the archive is: requests before the
    cutover must still go to scda, which is the only place those days exist.
    `patch` is applied per request and each request carries one date, so this
    can decide per date rather than per month.
    """
    from anemoi.datasets.create.sources import accumulations as acc

    if getattr(acc._scda, "_date_aware", False):
        return "_scda already date-aware"

    def as_int(value) -> int | None:
        """The request encodes date as year*10000 + month*100 + day."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            digits = value.replace("-", "")
            return int(digits) if digits.isdigit() else None
        for attr in ("year",):
            if hasattr(value, attr):
                return value.year * 10000 + value.month * 100 + value.day
        return None

    def _scda(request):
        if request["time"] not in (6, 18, 600, 1800):
            request["stream"] = "oper"
            return request
        date = as_int(request.get("date"))
        # An unreadable date keeps the old behaviour rather than guessing.
        if date is None:
            request["stream"] = "scda"
        else:
            request["stream"] = "scda" if date < SCDA_RETIRED else "oper"
        return request

    _scda._date_aware = True
    # KWARGS is rebuilt inside accumulations() on every call and resolves
    # _scda from module globals, so rebinding the module attribute reaches it.
    acc._scda = _scda
    return f"_scda picks oper for 06Z/18Z from {SCDA_RETIRED}"


def apply_all() -> List[str]:
    """Apply every patch. Order does not matter; each is independent."""
    return [
        patch_json_tidy(),
        patch_fix_provenance(),
        patch_accumulations(),
        patch_iterate_patterns(),
        patch_scda_retirement(),
    ]

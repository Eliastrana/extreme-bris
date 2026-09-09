#!/usr/bin/env python3
"""Continue a dataset build instead of starting it over.

    ~/bris-data-env/bin/python scripts/anemoi_resume.py \
        recipe.yaml out.zarr --threads 4

WHY. anemoi already records which monthly groups are finished, in _build/flags
inside the dataset, and its load step skips the ones that are done. Nothing
used that. `anemoi-datasets create` refuses to touch a dataset that already
exists, and the slurm job deleted the target before every build, so a failure
at hour four cost four hours. That has now happened twice: once when a MARS
build died on a full disk after five hours, and once when three MEPS builds
were killed by the kernel.

WHAT IT DOES. If the dataset exists and the recipe stored inside it is byte
for byte the recipe being submitted, this runs anemoi's own step sequence
without `init`: load, finalise, additions, patch, cleanup, verify. The load
step then skips every group already flagged done.

WHY THE RECIPE COMPARISON IS NOT OPTIONAL. The load step reads its
configuration from the dataset, not from the file on disk. Resuming after
editing a recipe would therefore rebuild the missing groups using the OLD
recipe while the operator believes the new one is in effect, and the result
would carry two different definitions with nothing to show which state came
from which. If the recipes differ, this refuses and says so; wiping is then a
deliberate act, not a silent one.
"""

from __future__ import annotations

# Hand over to an interpreter that has these, if this one does not. These
# scripts are run by path, so the shebang picks up whatever python3 is on
# PATH, and on the login node that one has none of the stack.
import sys as _sys, pathlib as _pathlib  # noqa: E401
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure('anemoi')


import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anemoi_patches import apply_all                      # noqa: E402


def stored_recipe(path: Path):
    """The recipe anemoi saved inside the dataset, or None if there is none.

    A finished dataset has none: anemoi's cleanup step removes
    _create_yaml_config once the build completes. That is convenient rather
    than awkward, because it means the attribute is present exactly where a
    resume makes sense - a build that stopped partway - and absent where it
    does not. Do not "fix" a finished dataset by putting it back.
    """
    try:
        import zarr

        return zarr.open(str(path), mode="r").attrs.get("_create_yaml_config")
    except Exception:
        return None


def norm_dates(block):
    """The dates block with every timestamp reduced to a datetime.

    anemoi rewrites what it stores: `2024-09-08 00:00:00` on the way in comes
    back as `2024-09-08T00:00:00`. Comparing the strings therefore always
    fails, which is how the first version of this refused every resume it was
    ever offered - including the one it was written for, a build that lost
    four hours to a read timeout with ten of thirteen groups already done.
    """
    import datetime as dt

    def one(v):
        if isinstance(v, str):
            try:
                return dt.datetime.fromisoformat(v)
            except ValueError:
                return v
        if isinstance(v, list):
            return [one(x) for x in v]
        return v

    return {k: one(v) for k, v in (block or {}).items()
            if k in ("start", "end", "frequency", "missing")}


def same_recipe(stored, wanted) -> bool:
    """Do these two recipes build the same dataset from the same places?

    Not a whole-structure comparison. anemoi stores a normalised copy with its
    own defaults filled in - attribution, licence, chunking, order_by and more
    - so the stored recipe never equals the file even when nothing changed.
    What matters is what would be rebuilt and where it would come from: the
    dates and the input tree. Everything else is presentation or a default
    that does not change a single value in the array.
    """
    if not isinstance(stored, dict) or not isinstance(wanted, dict):
        return False
    if stored.get("input") != wanted.get("input"):
        return False
    return norm_dates(stored.get("dates")) == norm_dates(wanted.get("dates"))


def group_status(path: Path):
    """(done, total) monthly groups, or None when the dataset has no _build."""
    try:
        import zarr

        z = zarr.open(str(path), mode="r")
        if "_build" not in z:
            return None
        flags = z["_build"]["flags"][:]
        return int(flags.sum()), len(flags)
    except Exception:
        return None


def run(what: str, options: dict):
    from anemoi.datasets.create import creator_factory

    opts = {k: v for k, v in options.items() if v is not None}
    return creator_factory(what.replace("-", "_"), **opts).run()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("path", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="report whether a resume would run, and change nothing")
    args = ap.parse_args()

    for line in apply_all():
        print(f"[patch] {line}", file=sys.stderr)

    import yaml

    status = group_status(args.path)
    if status is None:
        print("no resumable dataset here (no _build group)", file=sys.stderr)
        return 4

    stored = stored_recipe(args.path)
    wanted = yaml.safe_load(args.recipe.read_text())
    if stored is None:
        print("dataset has no stored recipe; refusing to resume", file=sys.stderr)
        return 4
    if not same_recipe(stored, wanted):
        print("stored recipe differs from the one submitted; refusing to "
              "resume.\nThe load step reads its config from the dataset, so "
              "resuming would\nrebuild the missing groups from the old "
              "recipe. Delete the target to\nrebuild with the new one.",
              file=sys.stderr)
        return 4

    done, total = status
    if done == 0:
        print(f"nothing finished yet ({done} of {total} groups); "
              "a full build is no more expensive", file=sys.stderr)
        return 4

    print(f"resuming: {done} of {total} groups already built", file=sys.stderr)
    if args.dry_run:
        print("dry run; nothing done", file=sys.stderr)
        return 0

    options = {"path": str(args.path), "use_threads": True}

    futures = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for n in range(total):
            opt = dict(options, parts=f"{n + 1}/{total}")
            futures.append(pool.submit(run, "load", opt))
        for f in as_completed(futures):
            f.result()

    for step in ("finalise", "init-additions"):
        run(step, options)

    futures = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for n in range(total):
            opt = dict(options, parts=f"{n + 1}/{total}")
            futures.append(pool.submit(run, "load-additions", opt))
        for f in as_completed(futures):
            f.result()

    for step in ("finalise-additions", "patch", "cleanup", "verify"):
        run(step, options)

    print("resume complete", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

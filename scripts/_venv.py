"""Re-run the calling script under an environment that has what it needs.

These scripts live in a repo that is not installed and are run by path, so the
shebang picks up whatever python3 the login node puts on PATH. That one has
neither anemoi nor zarr, and the traceback it produces points at the import
rather than at the interpreter, which reads like a broken environment instead
of the wrong one.

There is more than one environment here. bris-env carries the training and
inference stack; bris-data-env carries the dataset build stack. They do not
hold the same packages, and which one a script needs depends on the script. So
rather than naming an interpreter, this asks each candidate whether it can
import what the caller asked for, and hands over to the first that says yes.

The handover is announced on stderr. A script that silently runs under an
interpreter you did not choose is worse than the traceback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def candidates() -> list[Path]:
    """Interpreters to try, in order, without duplicates."""
    home = Path.home()
    env_dir = Path(os.environ.get("BRIS_ENV_DIR", home / "bris-env"))
    ordered = [
        env_dir / ".venv" / "bin" / "python",
        env_dir / "bin" / "python",
        home / "bris-env" / ".venv" / "bin" / "python",
        home / "bris-data-env" / "bin" / "python",
        home / "bris-data-env" / ".venv" / "bin" / "python",
    ]
    seen: set[Path] = set()
    out = []
    for p in ordered:
        if p not in seen and p.exists():
            seen.add(p)
            out.append(p)
    return out


def can_import(python: Path, modules: tuple[str, ...]) -> bool:
    code = "import " + ", ".join(modules)
    try:
        return subprocess.run([str(python), "-c", code],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ensure(*modules: str) -> None:
    """Hand over to an interpreter that can import every named module."""
    import importlib.util

    wanted = modules or ("anemoi",)
    if all(importlib.util.find_spec(m) is not None for m in wanted):
        return

    # Set once, so a half-built environment cannot re-execute forever.
    if os.environ.get("XBRIS_REEXEC"):
        raise SystemExit(
            f"{sys.executable} still cannot import {', '.join(wanted)}. "
            "That interpreter was chosen because it claimed it could, so "
            "something is inconsistent; check it with `pip list`."
        )

    tried = candidates()
    for python in tried:
        if python == Path(sys.executable):
            continue
        if not can_import(python, wanted):
            continue
        print(f"note: {sys.executable} cannot import {', '.join(wanted)}; "
              f"re-running under {python}", file=sys.stderr)
        os.environ["XBRIS_REEXEC"] = "1"
        os.execv(str(python), [str(python), *sys.argv])

    listing = "\n  ".join(str(p) for p in tried) or "  (none found)"
    raise SystemExit(
        f"no interpreter here can import {', '.join(wanted)}.\n"
        f"Running under {sys.executable} (PATH python3 is {shutil.which('python3')}).\n"
        f"Tried:\n  {listing}\n"
        "Set BRIS_ENV_DIR, or add the environment to candidates() in this file."
    )

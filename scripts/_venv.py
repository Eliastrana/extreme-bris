"""Re-run the calling script under the project environment if anemoi is missing.

These scripts live in a repo that is not installed and are run by path, so the
shebang picks up whatever python3 the login node puts on PATH. That one has no
anemoi in it, and the traceback it produces points at the import rather than
at the interpreter, which reads like a broken environment instead of the wrong
one.

Rather than documenting a longer command, find the environment and hand over
to it. The handover is announced on stderr, because a script that silently
runs under an interpreter you did not choose is worse than the traceback.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def ensure(module: str = "anemoi") -> None:
    import importlib.util

    if importlib.util.find_spec(module) is not None:
        return

    # Set once, this stops a broken environment from re-executing forever.
    if os.environ.get("XBRIS_REEXEC"):
        raise SystemExit(
            f"{sys.executable} still cannot import {module}. The environment at "
            f"{os.environ.get('BRIS_ENV_DIR', '~/bris-env')} is not the one this "
            "expects; check it with `pip list | grep anemoi`."
        )

    env_dir = Path(os.environ.get("BRIS_ENV_DIR", Path.home() / "bris-env"))
    candidate = env_dir / ".venv" / "bin" / "python"
    if not candidate.exists():
        found = shutil.which("python3")
        raise SystemExit(
            f"cannot import {module}, and no interpreter at {candidate}.\n"
            f"Running under {sys.executable} (PATH python3 is {found}).\n"
            "Set BRIS_ENV_DIR, or run the script with the environment's own "
            "python directly."
        )

    print(f"note: {module} is absent from {sys.executable}; "
          f"re-running under {candidate}", file=sys.stderr)
    os.environ["XBRIS_REEXEC"] = "1"
    os.execv(str(candidate), [str(candidate), *sys.argv])

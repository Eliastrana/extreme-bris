#!/usr/bin/env python3
"""anemoi-training, with this project's patches applied first.

    python scripts/anemoi_train.py train --config-dir bris/train --config-name finetune

Identical to the anemoi-training command in every respect except that
xbris.patches.apply() runs before anemoi builds anything. The job script calls
this instead of the installed command, so both arms always get the same
patches, and so does the dry run, which applies them the same way.

Run it with the training environment's own Python. It does not hand itself
over to another interpreter, because a training job must never quietly start
under a different one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbris import patches  # noqa: E402

patches.apply()

from anemoi.training.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "anemoi-training"
    raise SystemExit(main())

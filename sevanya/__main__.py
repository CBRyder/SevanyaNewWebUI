"""Entry point: sort the requirements out, then start the server.

    python -m sevanya

This exists so the dependency check happens *before* anything that needs those
dependencies is imported. `python -m sevanya.server` imports fastapi at the top
of the module, so by the time any code of ours runs, a missing fastapi has
already ended the process with a traceback — the check can only ever confirm
what already worked. Here, nothing beyond the standard library is imported
until the requirements are known to be good, so a fresh clone, a new machine,
or a line added to requirements.txt all just work.

Set SEVANYA_SKIP_DEPS=1 to start without checking, for when you're offline or
managing the environment yourself.
"""

import os
import sys
from pathlib import Path

from . import deps


def main() -> int:
    root = Path.cwd().resolve()

    if os.environ.get("SEVANYA_SKIP_DEPS"):
        print("skipping the requirements check (SEVANYA_SKIP_DEPS is set)")
    else:
        # Only reaches for pip when something is actually wrong, so the usual
        # startup — everything already installed — costs a metadata scan and
        # nothing else.
        ok, report = deps.ensure(root, install_missing=True)
        print(f"requirements: {report}")
        if not ok:
            print(
                "\nNot starting: the requirements aren't right, and starting anyway "
                "would just fail on import with a less useful message.",
                file=sys.stderr,
            )
            return 1

    # Imported here, not at the top, which is the entire point of this file.
    from .server import main as serve

    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

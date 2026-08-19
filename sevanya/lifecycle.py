"""Restarting the process.

Lives here rather than in server.py because both the HTTP endpoint and the
`reload` tool need it, and tools.py importing server.py would be a circle —
server imports agent imports tools.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# How to start a fresh copy. `python -m sevanya` rather than
# `-m sevanya.server`, so a restart goes through the bootstrap and picks up a
# changed requirements.txt before importing anything that needs it — which is
# the case where restarting straight into the server would fail on import with
# nothing left listening to say why.
#
# Always a module entry point, whatever was typed originally: `python
# server.py` can't work, because the relative imports need the package context
# that -m provides. cwd and the environment come along with exec, so
# PROJECT_ROOT and SEVANYA_TOKEN are the same on the other side.
RELAUNCH = [sys.executable, "-m", "sevanya"]

# Set by server.py at import. The terminal REPL never sets it, so `reload` can
# say "there's no server to restart" instead of exec'ing one into existence
# underneath somebody typing at a prompt.
_under_server = False


def mark_server_running() -> None:
    global _under_server
    _under_server = True


def can_restart() -> bool:
    return _under_server


def relaunch() -> None:
    """Replace this process with a new one.

    execv rather than spawn-and-exit: same PID, same terminal, and no window
    where nothing is listening because the parent died before the child was
    up. Python marks its own file descriptors non-inheritable, so the listening
    socket closes as the image is replaced and the new process takes the port
    straight back.
    """
    os.execv(sys.executable, RELAUNCH)


def pull_latest(repo_dir: Path) -> tuple[bool, str]:
    """Fast-forward this checkout from its remote. Never merges, never discards.

    The point of this existing at all: editing index.html from a phone
    (through GitHub's own editor, say) has to get onto this machine's disk
    somehow before a restart can serve it — restart alone only re-execs the
    process, it never looks at the network.

    --ff-only refuses rather than merging or overwriting anything, so a real
    conflict — or local changes on this machine that were never pushed —
    comes back as a failure to report, never a silent loss of work.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only"],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        return False, "git isn't on PATH for this process"
    except subprocess.TimeoutExpired:
        return False, "git pull timed out after 30s — offline, or GitHub is slow"

    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, (output or "no output")


def schedule_restart(delay: float = 0.4) -> None:
    """Restart, but not until the answer has gone out.

    Calling execv inline would replace the process mid-request, so the client
    would see a dropped connection rather than a reply — which is exactly the
    state this is meant to fix, and so indistinguishable from it not working.
    """
    def run():
        time.sleep(delay)
        relaunch()

    threading.Thread(target=run, daemon=True).start()

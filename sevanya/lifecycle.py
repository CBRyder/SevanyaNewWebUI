"""Restarting the process.

Lives here rather than in server.py because both the HTTP endpoint and the
`reload` tool need it, and tools.py importing server.py would be a circle —
server imports agent imports tools.
"""

import os
import sys
import threading
import time

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

"""The terminal REPL, and the /rc command.

/rc exists because the two things happen in the same place: you're mid
conversation at the desk, and you want the phone to be able to reach her
without leaving the conversation to go and start a server.
"""

import sys

import pytest

from sevanya import main as repl


def test_rc_reports_a_server_that_is_already_up(monkeypatch):
    """Pressing it twice shouldn't start a second one fighting for the port."""
    started = []
    monkeypatch.setattr(repl, "_server_is_up", lambda port, timeout=1.0: True)
    monkeypatch.setattr(repl.subprocess, "Popen", lambda *a, **k: started.append(a))

    message = repl.start_web(port=8765)
    assert "already running" in message
    assert "8765" in message
    assert not started, "it launched another one anyway"


def test_rc_starts_the_server_from_this_directory(monkeypatch):
    """PROJECT_ROOT is wherever the process starts, and that's the only place
    she can read. Starting the server elsewhere would give the phone a
    different view of your files than the terminal has."""
    calls = {}

    class FakeProcess:
        pid = 4321
        def poll(self): return None

    def fake_popen(argv, **kwargs):
        calls["argv"] = argv
        calls["cwd"] = kwargs.get("cwd")
        return FakeProcess()

    ups = iter([False, True])
    monkeypatch.setattr(repl, "_server_is_up", lambda port, timeout=1.0: next(ups))
    monkeypatch.setattr(repl.subprocess, "Popen", fake_popen)

    message = repl.start_web(port=9999)
    assert calls["argv"] == [sys.executable, "-m", "sevanya"], (
        "-m sevanya, not -m sevanya.server: the bootstrap checks requirements "
        "before importing anything that needs them"
    )
    assert calls["cwd"] == str(repl.PROJECT_ROOT)
    assert "9999" in message and "4321" in message


def test_rc_says_so_when_the_server_dies_immediately(monkeypatch):
    """A silent failure here looks identical to a slow start."""
    class DeadProcess:
        pid = 1
        returncode = 1
        def poll(self): return 1

    monkeypatch.setattr(repl, "_server_is_up", lambda port, timeout=1.0: False)
    monkeypatch.setattr(repl.subprocess, "Popen", lambda *a, **k: DeadProcess())

    message = repl.start_web(port=9999, wait=2)
    assert "exited immediately" in message
    assert "python -m sevanya" in message, "it should say how to see the error"


def test_rc_gives_up_rather_than_hanging(monkeypatch):
    class SlowProcess:
        pid = 7
        def poll(self): return None

    monkeypatch.setattr(repl, "_server_is_up", lambda port, timeout=1.0: False)
    monkeypatch.setattr(repl.subprocess, "Popen", lambda *a, **k: SlowProcess())

    message = repl.start_web(port=9999, wait=0.5)
    assert "nothing answered" in message


def test_a_server_that_is_not_there_is_not_reported_as_up():
    """The probe has to actually probe. A port nothing is listening on."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert repl._server_is_up(free, timeout=0.5) is False


# --- what /rc must not do --------------------------------------------------


def test_the_repl_handles_rc_itself_and_passes_everything_else_on():
    """/show is hers; /rc is this program's.

    Swallowing every slash command would quietly break the one convention the
    prompt already tells her to honour.
    """
    source = (repl.__file__ and open(repl.__file__).read())
    handled = source.split("user_input.lower() in")[1].split("\n")[0]
    assert "/rc" in handled
    assert "/show" not in handled, "/show must reach her, not be eaten here"


def test_rc_is_documented_where_someone_would_look():
    assert "/rc" in (repl.__doc__ or "")

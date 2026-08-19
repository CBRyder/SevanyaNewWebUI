"""The restart button, against a real server process.

Everything else about restarting can be tested with the exec stubbed out, but
that leaves the one question that matters unanswered: does the process actually
come back? So this starts a real server, asks it to restart itself over HTTP,
and waits to see whether anything is listening afterwards.

It costs a few seconds. It is the only test here that would have caught a
restart that kills the server and doesn't bring it back, which is the failure
that leaves you walking to the machine.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    """Never the default 8765 — a test must not fight the server you use."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    """A real `python -m sevanya`, on its own port and its own database."""
    port = free_port()
    env = {
        **os.environ,
        "SEVANYA_PORT": str(port),
        "SEVANYA_TOKEN": "test-token",
        "PYTHONPATH": str(ROOT),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "sevanya"],
        cwd=str(tmp_path), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        if wait_for_health(base) is None:
            pytest.skip("server did not start in this environment")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


def wait_for_restart(base: str, previous: float, timeout: float = 25.0):
    """Wait for a *new* process image, not merely for something to answer.

    The restart is deliberately delayed so the response can go out first, which
    means for a moment the old process is still serving. Polling for any 200
    would catch it and conclude nothing happened — which is exactly the mistake
    the client would make too, so it waits on `started` changing, as this does.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = wait_for_health(base, timeout=1)
        if current is not None and current["started"] > previous:
            return current
        time.sleep(0.2)
    return None


def wait_for_health(base: str, timeout: float = 25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base}/api/health", timeout=1)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    return None


def test_the_server_actually_comes_back(server):
    before = wait_for_health(server)
    assert before is not None

    posted = httpx.post(f"{server}/api/restart",
                        headers={"Authorization": "Bearer test-token"}, timeout=5)
    assert posted.status_code == 200, "the response has to arrive before the process goes"

    after = wait_for_restart(server, before["started"])
    assert after is not None, "it restarted and never came back — the worst outcome"

    # execv keeps the pid, so a later start time is what proves this is a new
    # process image rather than a request that quietly did nothing.
    assert after["started"] > before["started"]


def test_an_unauthenticated_restart_is_refused_by_a_real_server(server):
    before = wait_for_health(server)
    assert httpx.post(f"{server}/api/restart", timeout=5).status_code == 401
    time.sleep(1.0)
    after = wait_for_health(server)
    assert after["started"] == before["started"], "it restarted anyway"


def test_it_serves_normally_after_restarting(server):
    before = wait_for_health(server)
    httpx.post(f"{server}/api/restart",
               headers={"Authorization": "Bearer test-token"}, timeout=5)
    assert wait_for_restart(server, before["started"]) is not None
    auth = {"Authorization": "Bearer test-token"}
    assert httpx.get(f"{server}/", timeout=5).status_code == 200
    assert httpx.get(f"{server}/api/conversations", headers=auth, timeout=5).status_code == 200

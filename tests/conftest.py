"""Shared fixtures.

Two rules the whole suite depends on:

Nothing here touches ~/.sevanya/sevanya.db. That file is the part of this
project that isn't replaceable — the journal is the accumulated record of how
you've been learning, and a test run that wipes it has destroyed the only copy.
Every Store in these tests is built on a tmp_path.

Nothing here talks to the Anthropic API. Tests that need a model get the stub
below, so the suite runs offline, costs nothing, and gives the same answer
every time.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sevanya.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A small fake project tree, with tools pointed at it.

    tools.PROJECT_ROOT is read at import time from the working directory, so
    it has to be patched rather than set — and patching it is also what keeps
    the path-containment tests honest, since 'outside the project' has to mean
    somewhere real.
    """
    from sevanya import tools

    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "code.py").write_text("MAX_STEPS = 12\nprint(MAX_STEPS)\n")
    (root / "notes.md").write_text("mentions max_steps in lowercase\n")
    (root / "pkg" / "__pycache__").mkdir()
    (root / "pkg" / "__pycache__" / "junk.py").write_text("MAX_STEPS everywhere\n")
    (root / "blob.bin").write_bytes(b"\x00\x01MAX_STEPS\xff\xfe")

    monkeypatch.setattr(tools, "PROJECT_ROOT", root)
    return root


class _Block:
    """Stands in for an SDK content block, including model_dump()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.__dict__.items()
                if not exclude_none or v is not None}


class _Response:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class FakeClient:
    """Replays a scripted list of turns instead of calling the API.

    Every call asserts the tool schemas are well-formed, because that is
    exactly the bug this suite exists to catch: a tool definition missing
    input_schema is rejected by the API on *every* request, not just requests
    that use that tool, so one typo takes the whole assistant offline.
    """

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.messages = self

    def _check(self, kwargs):
        self.calls.append(kwargs)
        for tool in kwargs["tools"]:
            assert "input_schema" in tool, f"{tool['name']} has no input_schema"

    def _next(self):
        content, stop = self.turns.pop(0) if self.turns else ([_Block(type="text", text="done")], "end_turn")
        return _Response(content, stop)

    def create(self, **kwargs):
        self._check(kwargs)
        return self._next()

    def stream(self, **kwargs):
        self._check(kwargs)
        response = self._next()

        class Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            @property
            def text_stream(self_inner):
                for block in response.content:
                    if block.type == "text":
                        yield block.text

            def get_final_message(self_inner):
                return response

        return Ctx()


@pytest.fixture
def fake_client():
    return FakeClient


@pytest.fixture
def block():
    return _Block

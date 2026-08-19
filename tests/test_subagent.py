"""Handing work to a helper, and why it's worth the machinery.

Not specialisation — context. Her fixed overhead is ~3,500 tokens before you
type, one read_file can add ten thousand, and a local model has 32k to play
with. Three file reads and the conversation is full.

So the test that matters isn't "does the helper answer" — it's "does what the
helper read stay out of the main conversation".
"""

import json

import httpx
import pytest

from sevanya import subagent, tools


def server(script):
    """A stand-in model that plays out a scripted list of turns."""
    state = {"n": 0, "sent": []}

    def handler(request):
        state["sent"].append(json.loads(request.content))
        turn = script[min(state["n"], len(script) - 1)]
        state["n"] += 1
        return httpx.Response(200, json=turn)

    handler.state = state
    return handler


def backend(handler):
    from sevanya import backends

    return backends.LocalBackend(url="http://x/v1", model="m",
                                 client=httpx.Client(transport=httpx.MockTransport(handler)))


def call(name, arguments):
    return {"choices": [{"message": {"tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        "finish_reason": "tool_calls"}]}


def says(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "game"
    (root / "src").mkdir(parents=True)
    # Something big enough that passing it back would defeat the purpose.
    (root / "src" / "save.py").write_text("# save system\n" + ("x = 1\n" * 3000))
    monkeypatch.setattr(tools, "PROJECT_ROOT", root)
    return root


# --- what it may do ---------------------------------------------------------


def test_it_gets_the_reading_tools_only():
    """A researcher, not a second Sevanya.

    Journal writes and the task list are her judgement about the person she's
    teaching; an errand-runner has no business forming those.
    """
    names = {t["name"] for t in subagent.sub_tools()}
    assert names == {"read_file", "list_files", "grep", "sync_repo", "fetch_url"}
    for forbidden in ("remember", "add_task", "complete_task", "notify_phone", "reload"):
        assert forbidden not in names


def test_it_cannot_delegate_further():
    """Otherwise a model with a taste for delegation spawns helpers until the
    step limit stops it, each one costing a model call."""
    assert "delegate" not in {t["name"] for t in subagent.sub_tools()}
    assert "delegate" in {t["name"] for t in tools.TOOLS}, "she should still have it"


# --- the point --------------------------------------------------------------


def test_what_the_helper_reads_stays_out_of_the_conversation(store, project):
    """The whole justification, in one assertion.

    The helper reads a 3,000-line file. The caller gets a sentence.
    """
    handler = server([
        call("read_file", {"path": "src/save.py"}),
        says("The save system writes JSON to save.dat; see src/save.py line 1."),
    ])
    answer = subagent.run("How does the save system work?", store=store,
                          backend=backend(handler))

    assert "save.dat" in answer
    assert len(answer) < 200, "a paragraph, not a transcript"
    assert "x = 1" not in answer, "it handed back the file it was sent to read"

    # And the file did reach the helper — it isn't small because nothing happened.
    conversation = json.dumps(handler.state["sent"])
    assert "x = 1" in conversation


def test_a_long_answer_is_cut_rather_than_passed_on(store, project, monkeypatch):
    """A helper that returns everything it read has moved the problem."""
    monkeypatch.setattr(subagent, "MAX_REPLY", 100)
    handler = server([says("y" * 5000)])
    answer = subagent.run("summarise everything", store=store, backend=backend(handler))

    assert len(answer) < 300
    assert "ask something narrower" in answer


def test_it_gives_up_rather_than_looping(store, project, monkeypatch):
    """Nobody is watching this run, so a loop is a GPU burning for nothing."""
    monkeypatch.setattr(subagent, "MAX_STEPS", 3)
    handler = server([call("list_files", {})])   # forever
    answer = subagent.run("go round in circles", store=store, backend=backend(handler))

    assert "ran out of steps" in answer
    assert handler.state["n"] <= 3


# --- what it leaves behind --------------------------------------------------


def test_its_transcript_is_kept_but_out_of_your_list(store, project):
    """When a delegated answer is wrong, reading what it did is the only way to
    find out why — but these aren't conversations you had."""
    handler = server([call("list_files", {}), says("there's one file, save.py")])
    subagent.run("what's in here?", store=store, backend=backend(handler))

    assert [r["title"] for r in store.list_conversations()] == [], "it cluttered the list"

    everything = store.list_conversations(kind=None)
    assert len(everything) == 1
    assert everything[0]["title"].startswith("sub:")

    kept = store.load_messages(everything[0]["id"])
    blocks = [b for m in kept if isinstance(m["content"], list) for b in m["content"]]
    kinds = {b.get("type") for b in blocks}
    # Both halves. Matching only the tool_use would pass with the results
    # thrown away, and the results are the half you'd actually want to read
    # when a delegated answer turns out to be wrong.
    assert "tool_use" in kinds, "the call it made wasn't kept"
    assert "tool_result" in kinds, "what the tool returned wasn't kept"


def test_the_helper_records_which_model_did_the_work(store, project):
    handler = server([says("done")])
    subagent.run("a task", store=store, backend=backend(handler))
    row = store.db.execute(
        "SELECT model FROM messages WHERE model IS NOT NULL LIMIT 1").fetchone()
    assert "m" in row["model"]


# --- through the tool -------------------------------------------------------


def test_delegate_refuses_an_empty_task(store):
    out, is_error = tools.dispatch("delegate", {"task": "   "},
                                   store=store, conversation_id=store.new_conversation())
    assert "empty" in out


def test_a_failing_helper_is_reported_not_raised(store, monkeypatch):
    """Like every other tool: something she can work around and mention."""
    def explode(*a, **k):
        raise RuntimeError("the local server died")

    monkeypatch.setattr(subagent, "run", explode)
    out, is_error = tools.dispatch("delegate", {"task": "find the save code"},
                                   store=store, conversation_id=store.new_conversation())
    assert not is_error
    assert "the helper failed" in out
    assert "the local server died" in out


def test_the_helper_can_run_on_a_different_model(monkeypatch):
    """The routing half: searching is not the work that needs your best model."""
    from sevanya import backends

    monkeypatch.setenv("SEVANYA_SUBAGENT_BACKEND", "local")
    monkeypatch.setenv("SEVANYA_LOCAL_MODEL", "cheap-and-quick")
    chosen = subagent.backend_for_subagents()
    assert isinstance(chosen, backends.LocalBackend)
    assert chosen.model == "cheap-and-quick"


def test_without_that_variable_it_uses_whatever_she_uses(monkeypatch):
    monkeypatch.delenv("SEVANYA_SUBAGENT_BACKEND", raising=False)
    monkeypatch.delenv("SEVANYA_BACKEND", raising=False)
    from sevanya import backends

    monkeypatch.setattr(backends, "AnthropicBackend", lambda: "the main one")
    assert subagent.backend_for_subagents() == "the main one"

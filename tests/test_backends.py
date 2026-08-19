"""Which model answers, and the translation that lets a local one do it.

The stored transcript stays Anthropic-shaped whichever backend wrote it. That
is the whole design: change the shape on disk and every conversation already in
the database breaks. So these tests are mostly about the boundary — a thread
Claude started must be answerable by a model on your own machine, and the other
way round.
"""

import json

import httpx
import pytest

from sevanya import backends


# --- tool definitions -------------------------------------------------------


def test_tools_become_openai_functions():
    from sevanya import tools

    converted = backends.tools_for_openai(tools.TOOLS)
    assert len(converted) == len(tools.TOOLS)
    for original, new in zip(tools.TOOLS, converted):
        assert new["type"] == "function"
        assert new["function"]["name"] == original["name"]
        assert new["function"]["parameters"] == original["input_schema"]
        assert new["function"]["description"]


# --- transcripts ------------------------------------------------------------


def test_a_plain_exchange_converts():
    out = backends.messages_for_openai("be helpful", [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ])
    assert out == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_a_tool_call_and_its_result_are_paired_across_conventions():
    """Anthropic puts results in a user turn; OpenAI gives them their own role.

    Getting this wrong doesn't error — the model simply never sees what its
    tool returned, and repeats the call forever until MAX_STEPS trips.
    """
    out = backends.messages_for_openai("system", [
        {"role": "user", "content": "where is MAX_STEPS?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "tu_1", "name": "grep", "input": {"pattern": "MAX_STEPS"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "agent.py:22", "is_error": False},
        ]},
    ])

    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Looking."
    call = assistant["tool_calls"][0]
    assert call["id"] == "tu_1"
    assert call["function"]["name"] == "grep"
    assert json.loads(call["function"]["arguments"]) == {"pattern": "MAX_STEPS"}

    result = out[3]
    assert result == {"role": "tool", "tool_call_id": "tu_1", "content": "agent.py:22"}


def test_thinking_blocks_from_claude_are_dropped():
    """A thread Claude started can be picked up by a local model.

    Thinking blocks mean nothing to it, and passing along a block the other
    side doesn't understand is worse than saying less.
    """
    out = backends.messages_for_openai("system", [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "long internal reasoning"},
            {"type": "text", "text": "the answer"},
        ]},
    ])
    assert out[1] == {"role": "assistant", "content": "the answer"}
    assert "long internal reasoning" not in json.dumps(out)


def test_an_assistant_turn_that_is_only_a_tool_call_has_no_content():
    out = backends.messages_for_openai("system", [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "list_files", "input": {}},
        ]},
    ])
    assert out[1]["content"] is None
    assert out[1]["tool_calls"]


def test_several_results_in_one_turn_become_several_messages():
    """Parallel tool calls come back together and have to be split."""
    out = backends.messages_for_openai("system", [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "one"},
            {"type": "tool_result", "tool_use_id": "b", "content": "two"},
        ]},
    ])
    assert [m["tool_call_id"] for m in out[1:]] == ["a", "b"]


# --- responses --------------------------------------------------------------


def test_a_text_answer_comes_back_as_a_text_block():
    reply = backends.reply_from_openai({"content": "the answer"}, "stop")
    assert reply.content == [{"type": "text", "text": "the answer"}]
    assert reply.stop_reason == "stop"
    assert reply.text() == "the answer"


def test_a_tool_call_comes_back_in_the_stored_shape():
    reply = backends.reply_from_openai({
        "content": None,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "grep", "arguments": '{"pattern": "MAX_STEPS"}'},
        }],
    }, "tool_calls")

    assert reply.stop_reason == "tool_use", "the agent loop keys off exactly this"
    assert reply.content == [{
        "type": "tool_use", "id": "call_1", "name": "grep",
        "input": {"pattern": "MAX_STEPS"},
    }]


def test_malformed_arguments_are_handed_on_rather_than_raising():
    """Small models produce broken JSON often enough that this must not throw.

    Passed to the tool layer as an argument it will reject, so the model sees
    an error it can correct instead of the process dying under it.
    """
    reply = backends.reply_from_openai({
        "tool_calls": [{"id": "c", "function": {"name": "grep", "arguments": "{not json"}}],
    }, "tool_calls")
    assert reply.content[0]["input"] == {"__malformed__": "{not json"}


# --- the local backend over HTTP -------------------------------------------


def local(handler, **kwargs):
    return backends.LocalBackend(
        url="http://localhost:1234/v1", model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_send_asks_the_local_server_properly():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello from the GPU"}, "finish_reason": "stop"}]
        })

    from sevanya import tools
    reply = local(handler).send("be helpful", [{"role": "user", "content": "hi"}], tools.TOOLS)

    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["tools"][0]["type"] == "function"
    assert reply.text() == "hello from the GPU"


def test_no_tools_key_when_there_are_no_tools():
    """Some servers reject an empty tools array outright."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    local(handler).send("system", [{"role": "user", "content": "hi"}], [])
    assert "tools" not in seen["body"]


def sse(*chunks):
    body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def drain(stream):
    """Run a backend stream to the end; return (text pieces, final Reply)."""
    pieces = []
    while True:
        try:
            pieces.append(next(stream))
        except StopIteration as done:
            return pieces, done.value


def test_streaming_text_arrives_in_pieces():
    def handler(request):
        return sse(
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " there"}}, ]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

    pieces, reply = drain(local(handler).stream("system", [{"role": "user", "content": "hi"}], []))
    assert pieces == ["Hello", " there"]
    assert reply.text() == "Hello there"
    assert reply.stop_reason == "stop"


def test_reasoning_tokens_announce_thinking_once_not_per_token():
    """A reasoning model's thinking can run to hundreds of tokens.

    One signal that she's working on it, not one event per thinking token —
    nothing downstream wants that many, and it's not the answer anyway.
    """
    def handler(request):
        return sse(
            {"choices": [{"delta": {"reasoning_content": "Let me "}}]},
            {"choices": [{"delta": {"reasoning_content": "think about this."}}]},
            {"choices": [{"delta": {"content": "The answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

    pieces, reply = drain(local(handler).stream("system", [{"role": "user", "content": "hi"}], []))
    assert pieces == [("thinking", "thinking…"), "The answer"]
    assert reply.text() == "The answer"


def test_reasoning_content_itself_never_reaches_the_transcript():
    """The thinking tokens are what she's working out, not what she said.

    Surfacing the raw reasoning as "text" would put her internal monologue
    in the conversation as if she'd said it out loud.
    """
    def handler(request):
        return sse(
            {"choices": [{"delta": {"reasoning_content": "hmm, let me consider the options here"}}]},
            {"choices": [{"delta": {"content": "Yes."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

    pieces, reply = drain(local(handler).stream("system", [{"role": "user", "content": "hi"}], []))
    assert "hmm, let me consider the options here" not in pieces
    assert reply.text() == "Yes."


def test_no_reasoning_means_no_thinking_event():
    """A non-reasoning model's stream is unaffected -- no field, no signal."""
    def handler(request):
        return sse(
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

    pieces, _ = drain(local(handler).stream("system", [{"role": "user", "content": "hi"}], []))
    assert pieces == ["Hi"]


def test_streamed_tool_calls_are_reassembled_from_fragments():
    """The arguments arrive split across chunks, indexed rather than named.

    Concatenate them in the wrong order, or key them by anything but index,
    and you get JSON that doesn't parse — from a model that did nothing wrong.
    """
    def handler(request):
        return sse(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "grep", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"pat'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'tern": "X"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        )

    pieces, reply = drain(local(handler).stream("system", [{"role": "user", "content": "hi"}], []))
    assert pieces == []
    assert reply.stop_reason == "tool_use"
    assert reply.content[0]["name"] == "grep"
    assert reply.content[0]["input"] == {"pattern": "X"}


def test_two_parallel_tool_calls_stay_separate():
    def handler(request):
        return sse(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "grep", "arguments": "{}"}},
                {"index": 1, "id": "b", "function": {"name": "list_files", "arguments": "{}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        )

    _, reply = drain(local(handler).stream("system", [], []))
    assert [b["name"] for b in reply.content] == ["grep", "list_files"]


def test_a_keep_alive_frame_does_not_kill_the_stream():
    def handler(request):
        return httpx.Response(200, text=(
            ": ping\n\n"
            'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            "data: [DONE]\n\n"))

    pieces, reply = drain(local(handler).stream("system", [], []))
    assert pieces == ["hi"]


def test_an_http_error_from_the_local_server_is_not_swallowed():
    def handler(request):
        return httpx.Response(503, text="model not loaded")

    with pytest.raises(httpx.HTTPStatusError):
        local(handler).send("system", [{"role": "user", "content": "hi"}], [])


# --- choosing ---------------------------------------------------------------


def test_anthropic_is_the_default(monkeypatch):
    monkeypatch.delenv("SEVANYA_BACKEND", raising=False)
    monkeypatch.setattr(backends, "AnthropicBackend", lambda: "anthropic-backend")
    assert backends.choose() == "anthropic-backend"


@pytest.mark.parametrize("name", ["local", "lmstudio", "lm-studio", "ollama"])
def test_the_local_names_all_work(monkeypatch, name):
    """Nobody should have to remember which word this project chose."""
    monkeypatch.setenv("SEVANYA_BACKEND", name)
    assert isinstance(backends.choose(), backends.LocalBackend)


def test_an_unknown_backend_says_so(monkeypatch):
    monkeypatch.setenv("SEVANYA_BACKEND", "gpt5-at-home")
    with pytest.raises(ValueError, match="unknown SEVANYA_BACKEND"):
        backends.choose()


def test_it_asks_the_server_which_model_is_loaded(monkeypatch):
    """Naming the model by hand is a step you can't take until it's loaded.

    Getting it wrong is a 404 from Ollama, and from a more forgiving server a
    silent surprise where it serves something else entirely.
    """
    monkeypatch.delenv("SEVANYA_LOCAL_MODEL", raising=False)
    seen = {}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "qwen3.5-9b-instruct"}]})
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = backends.LocalBackend(
        url="http://localhost:1234/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert backend.model == "qwen3.5-9b-instruct"
    backend.send("system", [{"role": "user", "content": "hi"}], [])
    assert seen["model"] == "qwen3.5-9b-instruct", "it discovered a name and then ignored it"


def test_a_name_you_gave_it_wins(monkeypatch):
    """Discovery is a convenience, not an override."""
    monkeypatch.setenv("SEVANYA_LOCAL_MODEL", "the-one-i-chose")

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "something-else"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = backends.LocalBackend(
        url="http://localhost:1234/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert backend.model == "the-one-i-chose"


def test_a_server_that_is_not_up_yet_does_not_break_anything(monkeypatch):
    """describe() runs at server startup, possibly before LM Studio is running."""
    monkeypatch.delenv("SEVANYA_LOCAL_MODEL", raising=False)

    def dead(request):
        raise httpx.ConnectError("nothing there")

    backend = backends.LocalBackend(
        url="http://localhost:1234/v1",
        client=httpx.Client(transport=httpx.MockTransport(dead)))
    assert backend.model == "local-model"
    assert "local-model" in backend.describe()


def test_a_failed_lookup_is_not_remembered(monkeypatch):
    """You start her, then start LM Studio. She should catch up.

    Caching "unknown" from before the server existed would mean restarting her
    to fix something that fixed itself.
    """
    monkeypatch.delenv("SEVANYA_LOCAL_MODEL", raising=False)
    state = {"up": False}

    def handler(request):
        if not state["up"]:
            raise httpx.ConnectError("not yet")
        return httpx.Response(200, json={"data": [{"id": "arrived-late"}]})

    backend = backends.LocalBackend(
        url="http://localhost:1234/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert backend.model == "local-model"
    state["up"] = True
    assert backend.model == "arrived-late", "it gave up permanently"


def test_the_local_url_and_model_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("SEVANYA_BACKEND", "local")
    monkeypatch.setenv("SEVANYA_LOCAL_URL", "http://100.64.0.3:11434/v1/")
    monkeypatch.setenv("SEVANYA_LOCAL_MODEL", "qwen2.5-coder")
    backend = backends.choose()
    assert backend.url == "http://100.64.0.3:11434/v1", "the trailing slash should go"
    assert backend.model == "qwen2.5-coder"
    assert "qwen2.5-coder" in backend.describe()


# --- the whole loop, on a local model --------------------------------------


def test_a_local_model_can_drive_the_full_tool_loop(store, tmp_path, monkeypatch):
    """The one that matters: think, call a tool, read the result, answer.

    Everything below the wire format is the real thing — the real agent loop,
    the real dispatch, the real store. Only the model is a stand-in.
    """
    from sevanya import tools
    from sevanya.agent import Agent

    project = tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "code.py").write_text("MAX_STEPS = 12\n")
    monkeypatch.setattr(tools, "PROJECT_ROOT", project)

    turns = [
        # first: ask for grep
        {"choices": [{"message": {"content": "Let me look.", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "grep", "arguments": '{"pattern": "MAX_STEPS"}'}}]},
            "finish_reason": "tool_calls"}]},
        # second: having seen the result, answer
        {"choices": [{"message": {"content": "It's in pkg/code.py, line 1."},
                      "finish_reason": "stop"}]},
    ]
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=turns[len(sent) - 1])

    conversation = store.new_conversation()
    agent = Agent(store, conversation, backend=local(handler))
    answer = agent.send("where is MAX_STEPS?")

    assert answer == "It's in pkg/code.py, line 1."

    # The second request must carry the tool's output back, in OpenAI's shape.
    followup = sent[1]["messages"]
    tool_message = [m for m in followup if m["role"] == "tool"]
    assert tool_message, "the tool result never reached the model"
    assert "pkg/code.py:1" in tool_message[0]["content"]
    assert tool_message[0]["tool_call_id"] == "call_1"

    # And the transcript on disk is in the shape everything else expects.
    stored = store.load_messages(conversation)
    kinds = [b["type"] for m in stored for b in (m["content"] if isinstance(m["content"], list) else [])]
    assert "tool_use" in kinds and "tool_result" in kinds


def test_streaming_through_the_agent_yields_text_then_runs_tools(store, tmp_path, monkeypatch):
    from sevanya import tools
    from sevanya.agent import Agent

    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.md").write_text("hello\n")
    monkeypatch.setattr(tools, "PROJECT_ROOT", project)

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return sse(
                {"choices": [{"delta": {"content": "Looking"}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "c1",
                     "function": {"name": "list_files", "arguments": "{}"}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            )
        return sse(
            {"choices": [{"delta": {"content": "There's notes.md."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

    agent = Agent(store, store.new_conversation(), backend=local(handler))
    events = list(agent.stream("what's in here?"))

    assert ("tool", "list_files") in events, "the tool call should be announced"
    text = "".join(payload for kind, payload in events if kind == "text")
    assert "Looking" in text and "There's notes.md." in text


def test_a_thinking_event_reaches_agent_stream_untouched(store, tmp_path, monkeypatch):
    """_tagged() has to pass an already-tagged tuple through, not re-tag it.

    Naively tagging everything as "text" would turn ("thinking", "thinking…")
    into ("text", ("thinking", "thinking…")) -- a tuple where the UI expects
    a string, and thinking mislabeled as an answer either way.
    """
    from sevanya import tools
    from sevanya.agent import Agent

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(tools, "PROJECT_ROOT", project)

    def handler(request):
        return sse(
            {"choices": [{"delta": {"reasoning_content": "considering it"}}]},
            {"choices": [{"delta": {"content": "Sure."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

    agent = Agent(store, store.new_conversation(), backend=local(handler))
    events = list(agent.stream("hi"))

    assert ("thinking", "thinking…") in events
    assert all(not (kind == "text" and isinstance(payload, tuple)) for kind, payload in events)
    text = "".join(payload for kind, payload in events if kind == "text")
    assert text == "Sure."


def test_a_thread_started_by_claude_can_be_continued_locally(store):
    """Switching backends mid-conversation is the point of keeping one shape.

    The history here contains a thinking block, which only Anthropic produces.
    """
    from sevanya.agent import Agent

    conversation = store.new_conversation()
    store.append_message(conversation, "user", "explain decorators")
    store.append_message(conversation, "assistant", [
        {"type": "thinking", "thinking": "internal reasoning Claude produced"},
        {"type": "text", "text": "A decorator is a function returning a function."},
    ])

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "picking up where that left off"},
                         "finish_reason": "stop"}]})

    agent = Agent(store, conversation, backend=local(handler))
    assert agent.send("go on") == "picking up where that left off"

    body = json.dumps(sent[0]["messages"])
    assert "A decorator is a function" in body, "it lost the history"
    assert "internal reasoning Claude produced" not in body, "it sent a thinking block"


# --- keeping the material to improve a local model with --------------------


def test_assistant_turns_record_which_model_wrote_them(store, tmp_path, monkeypatch):
    """Material you can't attribute is material you can't filter.

    The point of the column is choosing what to train on later — you want the
    turns a strong model produced, not an average of everything that ever
    answered.
    """
    from sevanya.agent import Agent

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "an answer"}, "finish_reason": "stop"}]})

    conversation = store.new_conversation()
    Agent(store, conversation, backend=local(handler)).send("a question")

    rows = store.db.execute(
        "SELECT role, model FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation,)).fetchall()
    assert rows[0]["model"] is None, "a user turn has no model"
    assert "test-model" in rows[1]["model"]


def test_older_rows_stay_honest_about_not_knowing(store):
    """Written before anyone tracked it. Guessing would poison the column."""
    conversation = store.new_conversation()
    store.append_message(conversation, "assistant", [{"type": "text", "text": "before"}])
    assert store.db.execute("SELECT model FROM messages").fetchone()["model"] is None


def test_export_writes_one_json_object_per_conversation(store, tmp_path):
    from sevanya import db as db_cli

    conversation = store.new_conversation(title="real")
    store.append_message(conversation, "user", "where is MAX_STEPS?")
    store.append_message(conversation, "assistant", [
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "t1", "name": "grep", "input": {"pattern": "MAX_STEPS"}},
    ], model="anthropic · claude-opus-5")
    store.append_message(conversation, "user", [
        {"type": "tool_result", "tool_use_id": "t1", "content": "agent.py:22"}])
    store.append_message(conversation, "assistant",
                         [{"type": "text", "text": "agent.py, line 22."}],
                         model="anthropic · claude-opus-5")
    store.close()

    out = tmp_path / "train.jsonl"
    assert db_cli.main(["--path", str(store._path), "export", str(out)]) == 0

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert [m["role"] for m in record["messages"]] == [
        "system", "user", "assistant", "tool", "assistant"]
    assert record["messages"][0]["content"], "the system prompt should be in there"


def test_export_leaves_out_the_automatic_check_ins(store, tmp_path):
    """Their opening turn is an instruction to her, not something you said.

    Training on it teaches a model that users talk like a cron job.
    """
    from sevanya import db as db_cli
    from sevanya.store import CHECKIN_MARKER

    nudge = store.new_conversation(title="check-in")
    store.append_message(nudge, "user", f"{CHECKIN_MARKER} nudge them")
    store.append_message(nudge, "assistant", [{"type": "text", "text": "come back to the parser"}],
                         model="claude")
    store.close()

    out = tmp_path / "train.jsonl"
    db_cli.main(["--path", str(store._path), "export", str(out)])
    assert out.read_text().strip() == "", "a check-in got exported"


def test_export_can_be_narrowed_to_one_model(store, tmp_path):
    """The useful case: take what Claude wrote and train a local model on it."""
    from sevanya import db as db_cli

    good = store.new_conversation(title="claude answered")
    store.append_message(good, "user", "a question")
    store.append_message(good, "assistant", [{"type": "text", "text": "a good answer"}],
                         model="anthropic · claude-opus-5")

    other = store.new_conversation(title="local answered")
    store.append_message(other, "user", "a question")
    store.append_message(other, "assistant", [{"type": "text", "text": "a rougher answer"}],
                         model="local · tiny-model")
    store.close()

    out = tmp_path / "train.jsonl"
    db_cli.main(["--path", str(store._path), "export", str(out), "--model", "claude"])
    body = out.read_text()
    assert "a good answer" in body
    assert "a rougher answer" not in body


def test_export_skips_threads_she_never_answered(store, tmp_path):
    from sevanya import db as db_cli

    lonely = store.new_conversation(title="no reply")
    store.append_message(lonely, "user", "hello?")
    store.append_message(lonely, "user", "anyone?")
    store.close()

    out = tmp_path / "train.jsonl"
    db_cli.main(["--path", str(store._path), "export", str(out)])
    assert out.read_text().strip() == ""


# --- the setup check --------------------------------------------------------
#
# The checker exists because the question "is this model good enough for her"
# has one answer that matters: does it pick the right tool out of thirteen,
# reliably. These tests are about it giving an honest verdict rather than a
# flattering one.


WANTED = {"README": "read_file", "files are in this project": "list_files",
          "MAX_STEPS": "grep", "last time": "recall"}


def fake_server(behaviour):
    """Stands in for a local model with a particular failing."""
    state = {"n": 0}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})

        body = json.loads(request.content)
        ask = body["messages"][-1]["content"]
        wanted = next((tool for phrase, tool in WANTED.items() if phrase in ask), "read_file")
        state["n"] += 1

        if behaviour == "words":
            return httpx.Response(200, json={"choices": [{
                "message": {"content": "I would need to look at that."},
                "finish_reason": "stop"}]})

        if behaviour == "malformed":
            arguments = "{not json"
        else:
            arguments = "{}"

        if behaviour == "flaky" and state["n"] % 2 == 0:
            wanted = "notify_phone"      # confidently the wrong tool

        return httpx.Response(200, json={"choices": [{
            "message": {"tool_calls": [{"id": "c", "type": "function",
                                        "function": {"name": wanted, "arguments": arguments}}]},
            "finish_reason": "tool_calls"}]})

    return handler


def check_against(handler, monkeypatch, capsys, runs=1):
    from sevanya import check

    backend = local(handler)
    monkeypatch.setattr(check.backends, "choose", lambda name=None: backend)

    def get(url, **kwargs):
        # The request has to be attached, or raise_for_status() raises
        # RuntimeError instead of doing nothing — which the checker would
        # report as "cannot reach it", from a server that answered fine.
        request = httpx.Request("GET", url)
        response = handler(request)
        response.request = request
        return response

    monkeypatch.setattr(check.httpx, "get", get)
    code = check.main(["--runs", str(runs)])
    return code, capsys.readouterr().out


def test_the_probe_offers_every_tool_she_has():
    """Picking one of thirteen is a different job from calling the only one.

    Testing with a single tool flatters a model that will disappoint you the
    moment it's real — which is exactly the mistake this file used to make.
    """
    from sevanya import check
    from sevanya import tools

    assert check.PROBE is tools.TOOLS or len(check.PROBE) == len(tools.TOOLS)
    assert len(check.ASKS) >= 3
    for _, expected in check.ASKS:
        assert expected in {t["name"] for t in tools.TOOLS}


def test_a_model_that_always_picks_correctly_passes(monkeypatch, capsys):
    code, out = check_against(fake_server("good"), monkeypatch, capsys, runs=2)
    assert code == 0
    assert "every time" in out


def test_a_model_that_only_talks_fails(monkeypatch, capsys):
    """The failure this exists to catch.

    A model that chats well and calls tools badly reads as *her* being worse,
    which sends you looking in entirely the wrong place.
    """
    code, out = check_against(fake_server("words"), monkeypatch, capsys)
    assert code == 1
    assert "never called a tool" in out
    assert "function calling" in out, "it should say what to look for instead"


def test_a_model_that_picks_the_wrong_tool_half_the_time_fails(monkeypatch, capsys):
    """Capability isn't the question with a small model — reliability is.

    One lucky call tells you nothing about the twentieth, which is why the
    probe repeats.
    """
    code, out = check_against(fake_server("flaky"), monkeypatch, capsys, runs=2)
    assert code == 1
    assert "unreliable" in out
    assert "called the right one:" in out


def test_it_really_does_ask_more_than_once(monkeypatch, capsys):
    """The premise: one lucky call tells you nothing about the twentieth.

    This model is perfect for exactly one round and wrong afterwards, so a
    checker that quietly ran a single pass would hand it a clean bill.
    """
    state = {"n": 0}
    asks = None

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "honeymoon"}]})
        body = json.loads(request.content)
        ask = body["messages"][-1]["content"]
        wanted = next((tool for phrase, tool in WANTED.items() if phrase in ask), "read_file")
        state["n"] += 1
        if state["n"] > len(asks):
            wanted = "notify_phone"
        return httpx.Response(200, json={"choices": [{
            "message": {"tool_calls": [{"id": "c", "type": "function",
                                        "function": {"name": wanted, "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}]})

    from sevanya import check
    asks = check.ASKS

    passed_one, _ = check_against(handler, monkeypatch, capsys, runs=1)
    state["n"] = 0
    caught, out = check_against(handler, monkeypatch, capsys, runs=3)

    assert passed_one == 0, "the first round really is clean"
    assert caught == 1, "asking repeatedly should have caught it"
    assert "unreliable" in out


def test_malformed_arguments_count_against_it(monkeypatch, capsys):
    code, out = check_against(fake_server("malformed"), monkeypatch, capsys)
    assert code == 1
    assert "not valid JSON" in out


def test_the_check_says_how_much_context_she_needs(monkeypatch, capsys):
    """The number that decides whether a model can run her at all."""
    _, out = check_against(fake_server("good"), monkeypatch, capsys)
    assert "overhead" in out
    assert "16k" in out


def test_an_unreachable_server_says_what_to_check(monkeypatch, capsys):
    def dead(request):
        raise httpx.ConnectError("connection refused")

    code, out = check_against(dead, monkeypatch, capsys)
    assert code == 2
    assert "Start Server" in out

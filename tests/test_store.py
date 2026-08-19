"""Persistence, and the one thing about it that isn't obvious.

Assistant content is a *list of blocks*, not a string. Store only the text and
you silently drop the tool_use blocks — then reloading gives you tool_result
blocks answering requests the model can no longer see, and the API rejects the
whole conversation. The round-trip test below is the guard on that.
"""

from sevanya.store import Store, _readable


class Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.__dict__.items()
                if not exclude_none or v is not None}


def test_a_tool_call_survives_a_reload(store):
    conversation = store.new_conversation()
    store.append_message(conversation, "user", "where is MAX_STEPS?")
    store.append_message(conversation, "assistant", [
        Block(type="text", text="Looking."),
        Block(type="tool_use", id="tu_1", name="grep", input={"pattern": "MAX_STEPS"}),
    ])
    store.append_message(conversation, "user", [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "found it", "is_error": False},
    ])

    reloaded = store.load_messages(conversation)
    assert len(reloaded) == 3

    blocks = reloaded[1]["content"]
    kinds = [b["type"] for b in blocks]
    assert "tool_use" in kinds, "the tool_use block was dropped — the API will reject this"

    tool_use = next(b for b in blocks if b["type"] == "tool_use")
    result = reloaded[2]["content"][0]
    assert result["tool_use_id"] == tool_use["id"], "the result no longer answers the request"


def test_plain_string_content_round_trips(store):
    conversation = store.new_conversation()
    store.append_message(conversation, "user", "just text")
    assert store.load_messages(conversation)[0]["content"] == "just text"


def test_conversation_exists_distinguishes_empty_from_gone(store):
    """An empty conversation and a missing one are different answers.

    The phone keeps a conversation id in localStorage; it needs to know
    whether to keep using it or drop it.
    """
    conversation = store.new_conversation()
    assert store.conversation_exists(conversation)
    assert not store.conversation_exists(conversation + 999)


def test_latest_conversation_and_listing(store):
    first = store.new_conversation(title="first")
    second = store.new_conversation(title="second")
    store.append_message(second, "user", "hello")
    assert store.latest_conversation() == second
    listed = {row["id"]: row for row in store.list_conversations()}
    assert listed[first]["title"] == "first"
    assert listed[second]["n"] == 1


# --- the journal -----------------------------------------------------------


def test_remember_and_recall(store):
    conversation = store.new_conversation()
    store.remember("python/decorators", "clicked once I saw it as a function returning a function", conversation)
    hits = store.recall("decorators")
    assert len(hits) == 1
    assert "clicked" in hits[0]["note"]


def test_recall_matches_the_note_body_too(store):
    store.remember("misc", "the parser bug was an off-by-one")
    assert store.recall("off-by-one")


def test_recent_journal_is_most_recent_first(store):
    for i in range(7):
        store.remember(f"topic{i}", f"note {i}")
    recent = store.recent_journal(limit=5)
    assert len(recent) == 5
    assert recent[0]["topic"] == "topic6"


# --- searching past conversations -----------------------------------------


def test_search_finds_text_in_past_conversations(store):
    conversation = store.new_conversation(title="yesterday")
    store.append_message(conversation, "user", "I keep hitting a segfault in the parser")
    hits = store.search_messages("segfault")
    assert len(hits) == 1
    assert hits[0]["title"] == "yesterday"
    assert "segfault" in hits[0]["excerpt"]


def test_search_ignores_tool_results(store):
    """Tool results are file dumps. Matching inside them is noise, not recall.

    Without this filter, asking about anything that appears in your source
    returns the source back at you instead of the conversation about it.
    """
    conversation = store.new_conversation()
    store.append_message(conversation, "user", [
        {"type": "tool_result", "tool_use_id": "t1",
         "content": "def segfault_handler(): ...", "is_error": False},
    ])
    assert store.search_messages("segfault") == []


def test_readable_skips_tool_plumbing():
    content = [
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "id": "x", "name": "grep", "input": {}},
    ]
    assert _readable(content) == "visible"


def test_each_thread_gets_its_own_connection(store, tmp_path):
    """SQLite connections can't cross threads, and the web server uses a pool.

    One shared connection works fine in the REPL and breaks the moment two
    HTTP requests overlap, which is the kind of bug that only shows up on the
    phone.
    """
    import threading

    conversation = store.new_conversation()
    errors = []

    def write():
        try:
            store.append_message(conversation, "user", "from another thread")
        except Exception as exc:  # pragma: no cover - only on regression
            errors.append(exc)

    thread = threading.Thread(target=write)
    thread.start()
    thread.join()

    assert not errors, f"cross-thread write failed: {errors}"
    assert len(store.load_messages(conversation)) == 1

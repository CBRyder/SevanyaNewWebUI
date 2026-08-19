"""The HTTP surface the phone talks to.

server.py builds its Store at import time, so every test here imports it fresh
with the class swapped for one pointed at a tmp database. That's a little
awkward — it's the cost of a module-level `store = Store()` — but it's better
than a test run that writes into the real journal.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

import anthropic  # noqa: E402
import sevanya.store as store_mod  # noqa: E402


def build(tmp_path, monkeypatch, turns, token=None):
    """A TestClient over a fresh server module, a tmp DB and a stubbed model."""
    from tests.conftest import FakeClient

    real_store = store_mod.Store

    class TmpStore(real_store):
        def __init__(self, path=None):
            super().__init__(path if path is not None else tmp_path / "server.db")

    monkeypatch.setattr(store_mod, "Store", TmpStore)

    # A class, not a lambda: agent.py annotates its client parameter as
    # `anthropic.Anthropic | None`, and that union is evaluated when the
    # module is imported. A function object has no `|`, so a lambda here
    # fails the import rather than the test.
    class StubAnthropic(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(list(turns))

    monkeypatch.setattr(anthropic, "Anthropic", StubAnthropic)

    if token is None:
        monkeypatch.delenv("SEVANYA_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SEVANYA_TOKEN", token)

    sys.modules.pop("sevanya.server", None)
    import sevanya.server as server

    return TestClient(server.app), server


@pytest.fixture
def blocks():
    from tests.conftest import _Block
    return _Block


def answer(blocks, text="the answer"):
    return [([blocks(type="text", text=text)], "end_turn")]


# --- /api/chat -------------------------------------------------------------


def test_chat_streams_and_reports_its_conversation_id(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200

    events = [line for line in response.text.splitlines() if line.startswith("data:")]
    kinds = [line for line in events]
    assert any('"conversation"' in line for line in kinds), "client never learns the id"
    assert any('"the answer"' in line for line in kinds)
    assert any('"done"' in line for line in kinds)


def test_chat_without_an_id_starts_a_new_thread_not_the_latest(tmp_path, monkeypatch, blocks):
    """Deliberately not "resume latest".

    Siri creates its own conversations. If a bare /api/chat grabbed the newest
    thread, opening the web UI would drop you into whatever you last asked out
    loud.
    """
    client, server = build(tmp_path, monkeypatch, answer(blocks) * 3)
    first = client.post("/api/chat", json={"message": "one"})
    second = client.post("/api/chat", json={"message": "two"})
    assert _conversation_id(first) != _conversation_id(second)


def test_chat_titles_the_thread_it_creates(tmp_path, monkeypatch, blocks):
    """Otherwise the thread picker is a list of "(untitled)"."""
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    client.post("/api/chat", json={"message": "why is my parser segfaulting?"})
    listed = client.get("/api/conversations").json()
    assert listed[0]["title"].startswith("why is my parser")


def test_chat_reports_errors_as_events_rather_than_dying(tmp_path, monkeypatch, blocks):
    """A failure mid-stream has to reach the browser as text.

    The response has already started, so there's no status code left to
    change — an exception here is a silent truncation on the phone.
    """
    client, _ = build(tmp_path, monkeypatch, answer(blocks))

    import sevanya.server as server

    def boom(self, text):
        raise RuntimeError("model exploded")
        yield  # pragma: no cover

    monkeypatch.setattr(server.Agent, "stream", boom)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 200
    assert '"error"' in response.text
    assert "model exploded" in response.text


def _conversation_id(response):
    import json
    for line in response.text.splitlines():
        if line.startswith("data:"):
            event = json.loads(line[5:])
            if event.get("type") == "conversation":
                return event["id"]
    raise AssertionError("no conversation event in the stream")


# --- /api/ask (Siri) -------------------------------------------------------


def test_ask_blocks_and_returns_one_blob(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks, "spoken answer"))
    body = client.post("/api/ask", json={"message": "what is a decorator?"}).json()
    assert body["reply"] == "spoken answer"
    assert "conversation_id" in body


def test_every_siri_request_starts_a_fresh_thread(tmp_path, monkeypatch, blocks):
    """Continuity comes from recall, not from dragging a transcript along."""
    client, _ = build(tmp_path, monkeypatch, answer(blocks) * 2)
    first = client.post("/api/ask", json={"message": "one"}).json()
    second = client.post("/api/ask", json={"message": "two"}).json()
    assert first["conversation_id"] != second["conversation_id"]


# --- transcripts -----------------------------------------------------------


def test_transcript_strips_tool_plumbing(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    conversation = server.store.new_conversation()
    server.store.append_message(conversation, "user", "question")
    server.store.append_message(conversation, "assistant", [
        blocks(type="text", text="answer"),
        blocks(type="tool_use", id="t1", name="grep", input={}),
    ])
    transcript = client.get(f"/api/conversations/{conversation}").json()
    assert [m["text"] for m in transcript] == ["question", "answer"]


def test_missing_conversation_is_a_404_not_an_empty_list(tmp_path, monkeypatch, blocks):
    """A phone holding a stale id has to tell "empty" from "gone"."""
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    assert client.get("/api/conversations/4242").status_code == 404


# --- auth ------------------------------------------------------------------


def test_endpoints_are_open_when_no_token_is_set(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token=None)
    assert client.get("/api/conversations").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret"}])
def test_a_set_token_is_actually_required(tmp_path, monkeypatch, blocks, headers):
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    assert client.post("/api/chat", json={"message": "hi"}, headers=headers).status_code == 401
    assert client.post("/api/ask", json={"message": "hi"}, headers=headers).status_code == 401
    assert client.get("/api/conversations", headers=headers).status_code == 401
    assert client.get("/api/modes", headers=headers).status_code == 401
    assert client.post("/api/mode", json={"name": "direct"}, headers=headers).status_code == 401


def test_the_right_token_gets_in(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    headers = {"Authorization": "Bearer secret"}
    assert client.get("/api/conversations", headers=headers).status_code == 200


# --- modes -------------------------------------------------------------------


def test_modes_lists_every_mode_and_defaults_to_teach(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    body = client.get("/api/modes").json()
    assert body["current"] == "teach"
    names = {m["name"] for m in body["modes"]}
    assert names == {"teach", "direct", "review", "quiz"}
    # Every mode needs both, or a picker built from this has blank rows.
    assert all(m["label"] and m["description"] for m in body["modes"])


def test_setting_an_unknown_mode_400s_and_changes_nothing(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    res = client.post("/api/mode", json={"name": "sarcastic"})
    assert res.status_code == 400
    assert client.get("/api/modes").json()["current"] == "teach"


def test_setting_a_known_mode_persists_and_shows_up_in_get(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    res = client.post("/api/mode", json={"name": "review"})
    assert res.status_code == 200
    assert res.json()["mode"] == "review"
    assert client.get("/api/modes").json()["current"] == "review"


def test_a_mode_change_is_logged_like_every_other_state_change(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    client.post("/api/mode", json={"name": "quiz"})
    kinds = [n["kind"] for n in client.get("/api/notifications").json()]
    assert "mode" in kinds


def test_mode_changes_what_the_next_chat_actually_sends(tmp_path, monkeypatch, blocks):
    """The contract server.py owes agent.py: system_extra carries the mode.

    Not a round trip through the stub model — Agent._system() is already
    covered elsewhere (see test_tasks.py). This checks the one new wire:
    that a POST /api/mode changes what the *next* request builds, without
    needing a restart.
    """
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    assert server._mode_extra() == ""  # teach: the null case

    client.post("/api/mode", json={"name": "direct"})
    assert "Mode: direct" in server._mode_extra()


def test_the_page_is_served(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    assert client.get("/").status_code == 200


# The UI here is hand-built, separately from the server — on a fresh checkout
# before manifest.json exists this has nothing to check, and skips rather than
# failing on a file nobody has written yet. Once manifest.json is added it
# starts running for real, with no change needed here.
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    not (_REPO_ROOT / "manifest.json").is_file(),
    reason="manifest.json not written yet — this is the UI's to add",
)
def test_the_manifest_and_its_icons_are_served(tmp_path, monkeypatch, blocks):
    """The manifest points at these paths; a 404 here is a blank home screen icon."""
    client, _ = build(tmp_path, monkeypatch, answer(blocks))
    manifest = client.get("/manifest.json").json()
    for icon in manifest["icons"]:
        assert client.get(icon["src"]).status_code == 200, f"{icon['src']} is missing"


def test_missing_index_is_a_404_not_a_crash(tmp_path, monkeypatch, blocks):
    """Mid-rewrite the file can be absent for a moment — that's a 404, not a 500."""
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    real = server.WEB_DIR / "index.html"
    moved = real.with_suffix(".html.bak")
    real.rename(moved)
    try:
        assert client.get("/").status_code == 404
    finally:
        moved.rename(real)


# --- health ----------------------------------------------------------------


def test_health_needs_no_token_even_when_one_is_set(tmp_path, monkeypatch, blocks):
    """The one endpoint that must answer without auth.

    The client has to tell "the server is gone" from "the server is fine and my
    token is wrong", and it can't if the check itself 401s in both cases.
    """
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["auth_required"] is True


def test_health_says_when_no_token_is_required(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token=None)
    assert client.get("/api/health").json()["auth_required"] is False


def test_health_reports_when_this_process_started(tmp_path, monkeypatch, blocks):
    """execv keeps the pid, so the pid can't show that a restart happened.

    Without this the UI cannot tell a server that came back from one that
    never went away.
    """
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    assert client.get("/api/health").json()["started"] == server.STARTED


# --- restart ---------------------------------------------------------------


def test_restart_requires_the_token_when_one_is_set(tmp_path, monkeypatch, blocks):
    """This one does something to the machine rather than reading from it."""
    client, server = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    called = []
    monkeypatch.setattr(server.lifecycle, "schedule_restart", lambda *a, **k: called.append(True))

    assert client.post("/api/restart").status_code == 401
    assert client.post("/api/restart", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert not called, "an unauthorised request must not have scheduled anything"


def test_restart_schedules_the_relaunch_and_answers_first(tmp_path, monkeypatch, blocks):
    """The response has to go out before the process is replaced.

    Restarting inline would drop the connection mid-request, which looks
    exactly like the button not working.
    """
    client, server = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    called = []
    monkeypatch.setattr(server.lifecycle, "schedule_restart", lambda *a, **k: called.append(True))
    monkeypatch.setattr(server.lifecycle, "pull_latest", lambda *a, **k: (True, "up to date"))

    body = client.post("/api/restart", headers={"Authorization": "Bearer secret"}).json()
    assert body["restarting"] is True
    assert body["pid"] == __import__("os").getpid()
    assert called == [True]


# --- restart: pulling first -------------------------------------------------


def test_restart_pulls_before_scheduling_the_relaunch(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    pulled_dir = []
    monkeypatch.setattr(server.lifecycle, "schedule_restart", lambda *a, **k: None)
    monkeypatch.setattr(
        server.lifecycle, "pull_latest",
        lambda directory: (pulled_dir.append(directory), (True, "Fast-forward to abc123"))[1],
    )

    body = client.post("/api/restart", headers={"Authorization": "Bearer secret"}).json()
    assert pulled_dir == [server.WEB_DIR], "restart must pull the repo it actually serves from"
    assert body["pulled"] == "Fast-forward to abc123"


def test_a_failed_pull_refuses_to_restart(tmp_path, monkeypatch, blocks):
    """An unrelated restart on stale code isn't what pressing this asked for."""
    client, server = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    called = []
    monkeypatch.setattr(server.lifecycle, "schedule_restart", lambda *a, **k: called.append(True))
    monkeypatch.setattr(
        server.lifecycle, "pull_latest",
        lambda *a, **k: (False, "CONFLICT (content): Merge conflict in index.html"),
    )

    res = client.post("/api/restart", headers={"Authorization": "Bearer secret"})
    assert res.status_code == 409
    assert not called, "a refused pull must not restart on the old code anyway"

    kinds = [n["kind"] for n in client.get("/api/notifications", headers={"Authorization": "Bearer secret"}).json()]
    assert "restart-failed" in kinds


def test_skip_pull_env_var_goes_straight_to_a_plain_restart(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    called = []
    monkeypatch.setenv("SEVANYA_SKIP_PULL", "1")
    monkeypatch.setattr(server.lifecycle, "schedule_restart", lambda *a, **k: called.append(True))

    def fail_if_called(*a, **k):
        raise AssertionError("pull_latest must not run when SEVANYA_SKIP_PULL is set")

    monkeypatch.setattr(server.lifecycle, "pull_latest", fail_if_called)

    res = client.post("/api/restart", headers={"Authorization": "Bearer secret"})
    assert res.status_code == 200
    assert called == [True]


def test_the_relaunch_uses_the_module_entry_point(tmp_path, monkeypatch, blocks):
    """`python server.py` cannot work — the relative imports need -m."""
    from sevanya import lifecycle

    # -m sevanya, not -m sevanya.server: a restart goes through the bootstrap
    # so a changed requirements.txt is installed before anything imports it.
    assert lifecycle.RELAUNCH[1:] == ["-m", "sevanya"]


def test_importing_the_server_marks_it_restartable(tmp_path, monkeypatch, blocks):
    """The reload tool needs to know whether there's a server to restart.

    Under the terminal REPL there isn't, and exec'ing one into existence
    underneath somebody typing at a prompt would be a surprise.
    """
    from sevanya import lifecycle

    build(tmp_path, monkeypatch, answer(blocks))
    assert lifecycle.can_restart()


# --- the task list, read only ----------------------------------------------


def test_tasks_are_readable(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    server.store.add_task("rewrite the parser loop")
    done = server.store.add_task("already handled")
    server.store.complete_task(done)

    rows = client.get("/api/tasks").json()
    assert [r["task"] for r in rows] == ["rewrite the parser loop", "already handled"]
    assert [r["done"] for r in rows] == [0, 1]


def test_tasks_need_the_token(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    assert client.get("/api/tasks").status_code == 401


def test_there_is_no_way_to_change_the_list_over_http(tmp_path, monkeypatch, blocks):
    """It's her list. A write endpoint would quietly make it the user's.

    Checked against the route table rather than by trying verbs, so adding one
    later fails here on purpose and has to be a deliberate decision.
    """
    _, server = build(tmp_path, monkeypatch, answer(blocks))
    task_routes = [
        (route.path, sorted(route.methods))
        for route in server.app.routes
        if getattr(route, "path", "").startswith("/api/tasks")
    ]
    assert task_routes, "the tasks endpoint disappeared"
    for path, methods in task_routes:
        assert set(methods) <= {"GET", "HEAD"}, f"{path} accepts {methods}"


# --- notifications ---------------------------------------------------------


def test_notifications_are_readable(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    server.store.notify("restart", "reloading — all good")
    rows = client.get("/api/notifications").json()
    assert rows[0]["message"] == "reloading — all good"
    assert rows[0]["kind"] == "restart"


def test_notifications_need_the_token(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    assert client.get("/api/notifications").status_code == 401


def test_the_notification_log_cannot_be_written_over_http(tmp_path, monkeypatch, blocks):
    """A log of what happened. There is nothing here for a client to change."""
    _, server = build(tmp_path, monkeypatch, answer(blocks))
    routes = [
        (route.path, sorted(route.methods))
        for route in server.app.routes
        if getattr(route, "path", "").startswith("/api/notifications")
    ]
    assert routes, "the notifications endpoint disappeared"
    for path, methods in routes:
        assert set(methods) <= {"GET", "HEAD"}, f"{path} accepts {methods}"


def test_starting_up_records_the_dependency_check(tmp_path, monkeypatch, blocks):
    """A restart into a broken environment has to leave evidence somewhere.

    Otherwise the only trace is a traceback in a terminal nobody is watching,
    which is precisely the situation when you're holding a phone.

    The notice has to carry what the check actually said — asserting only that
    some startup notice exists would pass with the check removed entirely.
    """
    from sevanya import deps

    monkeypatch.setattr(deps, "ensure",
                        lambda root, install_missing=True: (True, "checked-3-requirements"))
    client, _ = build(tmp_path, monkeypatch, answer(blocks))

    rows = client.get("/api/notifications").json()
    startup = [r for r in rows if r["kind"] == "startup"]
    assert startup, "nothing recorded at startup"
    assert "checked-3-requirements" in startup[0]["message"]


def test_the_startup_check_installs_rather_than_only_reporting(tmp_path, monkeypatch, blocks):
    """Asked for explicitly: cover the requirements on start, don't just moan."""
    from sevanya import deps

    seen = {}

    def ensure(root, install_missing=True):
        seen["install"] = install_missing
        return True, "fine"

    monkeypatch.delenv("SEVANYA_SKIP_DEPS", raising=False)
    monkeypatch.setattr(deps, "ensure", ensure)
    build(tmp_path, monkeypatch, answer(blocks))
    assert seen["install"] is True


# --- the automatic check-in ------------------------------------------------


def test_the_synthetic_prompt_is_hidden_from_the_transcript(tmp_path, monkeypatch, blocks):
    """It's an instruction to her, not something the user said.

    Showing it would put words in their mouth in their own transcript — and
    they'd be words about themselves in the third person.
    """
    from sevanya.store import CHECKIN_MARKER

    client, server = build(tmp_path, monkeypatch, answer(blocks))
    conversation = server.store.new_conversation(title="check-in")
    server.store.append_message(conversation, "user", f"{CHECKIN_MARKER} nudge her into writing")
    server.store.append_message(conversation, "assistant",
                                [blocks(type="text", text="Worth picking the parser back up.")])

    transcript = client.get(f"/api/conversations/{conversation}").json()
    assert [m["text"] for m in transcript] == ["Worth picking the parser back up."]


def test_an_ordinary_message_is_still_shown(tmp_path, monkeypatch, blocks):
    """The filter must not swallow anything the user actually typed."""
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    conversation = server.store.new_conversation()
    server.store.append_message(conversation, "user", "[sevanya] is a good name, right?")
    transcript = client.get(f"/api/conversations/{conversation}").json()
    assert transcript[0]["text"] == "[sevanya] is a good name, right?"


def test_the_checkin_thread_is_not_started_at_import(tmp_path, monkeypatch, blocks):
    """Importing the module for a test must not spawn a thread that calls the API."""
    import threading

    before = {t.name for t in threading.enumerate()}
    build(tmp_path, monkeypatch, answer(blocks))
    after = {t.name for t in threading.enumerate()}
    assert "sevanya-checkin" not in after - before


# --- maintenance from the phone --------------------------------------------


def seed_db(server, blocks):
    conversation = server.store.new_conversation(title="a chat")
    server.store.append_message(conversation, "user", "hello")
    server.store.append_message(conversation, "assistant", [blocks(type="text", text="hi")])
    server.store.remember("topic", "a note worth keeping", conversation)
    server.store.add_task("a task worth keeping")
    return conversation


def test_the_database_state_is_readable_over_http(tmp_path, monkeypatch, blocks):
    """Everything the CLI prints, as JSON.

    The point is deciding whether to touch it from a phone, rather than from a
    terminal on the machine it runs on.
    """
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    seed_db(server, blocks)

    body = client.get("/api/db").json()
    assert body["counts"]["conversations"] == 1
    assert body["schema_version"] == body["schema_expects"]
    assert body["drift"] == []
    assert body["pending_migrations"] == []
    assert body["unloadable"] == 0


def test_maintenance_endpoints_need_the_token(tmp_path, monkeypatch, blocks):
    client, _ = build(tmp_path, monkeypatch, answer(blocks), token="secret")
    assert client.get("/api/db").status_code == 401
    assert client.post("/api/db/backup").status_code == 401
    assert client.post("/api/db/clear-history", json={"confirm": True}).status_code == 401


def test_a_backup_can_be_taken_from_the_phone(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    seed_db(server, blocks)

    saved = client.post("/api/db/backup").json()
    assert saved["name"].startswith("sevanya-")
    assert saved["bytes"] > 0
    assert client.get("/api/db").json()["backups"][0]["name"] == saved["name"]


def test_clearing_requires_saying_so(tmp_path, monkeypatch, blocks):
    """Not implied by having made the request.

    A destructive endpoint that fires on an empty body is one stray tap, or one
    retried request, away from deleting the transcripts.
    """
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    seed_db(server, blocks)

    assert client.post("/api/db/clear-history", json={}).status_code == 400
    assert client.post("/api/db/clear-history", json={"confirm": False}).status_code == 400
    assert server.store.counts()["conversations"] == 1, "it deleted anyway"


def test_clearing_keeps_the_journal_and_the_list(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    seed_db(server, blocks)

    done = client.post("/api/db/clear-history", json={"confirm": True}).json()
    assert done["removed"]["conversations"] == 1
    assert done["counts"]["journal"] == 1
    assert done["counts"]["task_list"] == 1
    assert server.store.recall("note"), "her note should still be there"


def test_clearing_always_backs_up_first(tmp_path, monkeypatch, blocks):
    """There's deliberately no way to skip it over HTTP.

    On the command line you're standing at the machine and can insist. From a
    phone, one mis-tap shouldn't be the end of the transcripts.
    """
    import sqlite3

    client, server = build(tmp_path, monkeypatch, answer(blocks))
    seed_db(server, blocks)

    done = client.post("/api/db/clear-history", json={"confirm": True}).json()
    saved = tmp_path / "backups" / done["backup"]
    assert saved.exists()

    conn = sqlite3.connect(saved)
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    conn.close()


def test_recent_threads_can_be_kept_from_the_phone(tmp_path, monkeypatch, blocks):
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    old = server.store.new_conversation(title="last month")
    server.store.append_message(old, "user", "old")
    server.store.db.execute(
        "UPDATE conversations SET updated_at = datetime('now', '-40 days') WHERE id = ?", (old,))
    server.store.db.commit()
    recent = server.store.new_conversation(title="today")
    server.store.append_message(recent, "user", "new")

    client.post("/api/db/clear-history", json={"confirm": True, "keep_days": 7})
    assert [row["title"] for row in server.store.list_conversations()] == ["today"]


def test_clearing_is_recorded_in_the_notices(tmp_path, monkeypatch, blocks):
    """Destructive and remote. It should leave a trace naming the backup."""
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    seed_db(server, blocks)
    client.post("/api/db/clear-history", json={"confirm": True})

    notices = client.get("/api/notifications").json()
    cleared = [n for n in notices if n["kind"] == "clear-history"]
    assert cleared
    assert "backup:" in cleared[0]["message"]


def test_drift_is_reported_over_http_when_there_is_some(tmp_path, monkeypatch, blocks):
    """Asserting it's empty on a healthy database proves nothing.

    A hardcoded [] passes that. The database is drifted here after the store is
    already open, which is also the only way to reach this state at runtime —
    Store refuses to open a drifted file in the first place.
    """
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    server.store.db.execute("ALTER TABLE journal RENAME COLUMN note TO note_old")
    server.store.db.commit()

    body = client.get("/api/db").json()
    assert body["drift"] == ["journal.note is missing"]


def test_unloadable_rows_are_reported_over_http(tmp_path, monkeypatch, blocks):
    """Same reasoning: zero on a clean database proves nothing."""
    client, server = build(tmp_path, monkeypatch, answer(blocks))
    conversation = server.store.new_conversation()
    server.store.db.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?,?,?)",
        (conversation, "assistant", '{"old": "shape"}'))
    server.store.db.commit()

    body = client.get("/api/db").json()
    assert body["unloadable"] == 1
    assert "expected a string or a list of blocks" in body["unloadable_sample"][0]["why"]

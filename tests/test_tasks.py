"""The task list: add, complete, remove.

Distinct from the journal on purpose. The journal is what Sevanya noticed and
is written once; this has a lifecycle, and the interesting behaviour is all at
the edges — an id that doesn't exist, a task completed twice, the difference
between finishing something and deciding it never mattered.
"""

import pytest

from sevanya import tools


@pytest.fixture
def call(store):
    """Invoke a tool the way the agent loop does, through dispatch."""
    conversation = store.new_conversation()

    def invoke(name, **arguments):
        output, is_error = tools.dispatch(
            name, arguments, store=store, conversation_id=conversation
        )
        return output, is_error

    return invoke


# --- the table -------------------------------------------------------------


def test_a_task_survives_and_comes_back_open(store):
    task_id = store.add_task("write the iOS Shortcut")
    row = store.get_task(task_id)
    assert row["task"] == "write the iOS Shortcut"
    assert not row["done"]
    assert row["completed_at"] is None


def test_completing_stamps_the_time(store):
    task_id = store.add_task("read the streaming code")
    assert store.complete_task(task_id)
    row = store.get_task(task_id)
    assert row["done"]
    assert row["completed_at"] is not None


def test_completed_tasks_leave_the_open_list_but_not_the_table(store):
    kept = store.add_task("still to do")
    done = store.add_task("finished")
    store.complete_task(done)

    assert [r["id"] for r in store.open_tasks()] == [kept]
    assert {r["id"] for r in store.list_tasks(include_done=True)} == {kept, done}


def test_removing_is_permanent(store):
    task_id = store.add_task("never mind")
    assert store.remove_task(task_id)
    assert store.get_task(task_id) is None
    assert not store.remove_task(task_id), "removing twice should report nothing removed"


def test_completing_or_removing_an_unknown_id_reports_false(store):
    assert not store.complete_task(999)
    assert not store.remove_task(999)


def test_a_task_can_be_reopened(store):
    task_id = store.add_task("thought this was done")
    store.complete_task(task_id)
    assert store.reopen_task(task_id)
    row = store.get_task(task_id)
    assert not row["done"] and row["completed_at"] is None


def test_open_and_listed_rows_have_the_same_shape(store):
    """A narrower select turns any shared formatter into an IndexError.

    sqlite3.Row raises on a column that wasn't selected, so two queries whose
    rows get rendered by the same helper have to agree on their columns.
    """
    store.add_task("something")
    open_row = store.open_tasks()[0]
    listed_row = store.list_tasks()[0]
    assert set(open_row.keys()) <= set(listed_row.keys())
    for row in (open_row, listed_row):
        assert row["id"] is not None and row["task"] and row["done"] == 0


def test_the_list_is_oldest_first(store):
    """The order you'd work through them, not newest-shouting-loudest."""
    first = store.add_task("one")
    second = store.add_task("two")
    assert [r["id"] for r in store.open_tasks()] == [first, second]


def test_an_existing_database_gains_the_table(tmp_path):
    """CREATE TABLE IF NOT EXISTS means no migration for a database in use."""
    from sevanya.store import Store

    path = tmp_path / "existing.db"
    old = Store(path)
    old.remember("topic", "a note from before the task list existed")
    old.close()

    fresh = Store(path)
    assert fresh.add_task("works on the old database")
    assert fresh.recall("note"), "the existing journal should be untouched"


# --- the tools -------------------------------------------------------------


def test_add_returns_the_id_the_model_will_need(call):
    """Completing something later requires knowing its number."""
    out, is_error = call("add_task", task="write the iOS Shortcut")
    assert not is_error
    assert "1" in out and "iOS Shortcut" in out


def test_the_full_lifecycle_through_dispatch(call):
    call("add_task", task="one")
    call("add_task", task="two")

    out, _ = call("list_tasks")
    assert "[ ] 1. one" in out and "[ ] 2. two" in out

    out, is_error = call("complete_task", task_id=1)
    assert not is_error and "completed task 1" in out

    out, _ = call("list_tasks")
    assert "one" not in out and "two" in out

    out, _ = call("list_tasks", include_done=True)
    assert "[x] 1. one" in out and "[ ] 2. two" in out

    out, is_error = call("remove_task", task_id=2)
    assert not is_error and "removed task 2" in out
    assert "nothing open" in call("list_tasks")[0]


def test_a_wrong_id_shows_the_real_list_instead_of_failing(call):
    """The id was probably misread, and the open list is short.

    Answering "no such task" and nothing else invites another guess.
    """
    call("add_task", task="the actual task")
    out, is_error = call("complete_task", task_id=99)
    assert not is_error, "this is a correctable mistake, not a tool failure"
    assert "no task 99" in out
    assert "the actual task" in out


def test_completing_something_already_done_says_so(call):
    call("add_task", task="done once")
    call("complete_task", task_id=1)
    out, is_error = call("complete_task", task_id=1)
    assert not is_error
    assert "already done" in out


def test_an_empty_task_is_refused_without_writing_a_row(call, store):
    out, _ = call("add_task", task="   ")
    assert "empty" in out
    assert store.list_tasks(include_done=True) == []


def test_removing_reports_what_it_removed(call):
    """So the user can tell from the transcript what was thrown away."""
    call("add_task", task="the one that got deleted")
    out, _ = call("remove_task", task_id=1)
    assert "the one that got deleted" in out


# --- what she sees at the start of a conversation --------------------------


# The header of the block agent._system() injects, as distinct from the
# standing instructions in prompt.py that also mention the task list.
INJECTED = "On their task list, still open"


def build_agent(store):
    from sevanya.agent import Agent

    class Stub:
        def __init__(self, *a, **k):
            self.messages = None

    return Agent(store, store.new_conversation(), client=Stub())


def test_open_tasks_are_in_the_system_prompt_with_their_ids(store):
    """She has to be able to act on the list without a lookup first.

    Completing a task needs its id; if the ids weren't shown she'd have to
    call list_tasks before every single completion.
    """
    store.add_task("write the iOS Shortcut")
    prompt = build_agent(store)._system()
    # The injected section specifically — "task list" alone also appears in
    # the static prompt, so matching that would pass with nothing injected.
    assert INJECTED in prompt
    assert "1. write the iOS Shortcut" in prompt
    assert "task_id" in prompt


def test_completed_tasks_are_not_in_the_prompt(store):
    task_id = store.add_task("already handled")
    store.complete_task(task_id)
    assert "already handled" not in build_agent(store)._system()


def test_no_task_section_when_the_list_is_empty(store):
    assert INJECTED not in build_agent(store)._system()


def test_the_journal_still_gets_injected_alongside(store):
    """Adding tasks must not have displaced the journal."""
    store.remember("python/decorators", "clicked when he saw it as a function")
    store.add_task("a task")
    prompt = build_agent(store)._system()
    assert "decorators" in prompt and "a task" in prompt


def test_the_injected_list_is_capped(store):
    """The context window isn't free, and a 200-item list would eat it."""
    for i in range(40):
        store.add_task(f"task number {i}")
    prompt = build_agent(store)._system()
    assert "task number 9" in prompt
    assert "task number 39" not in prompt


# --- notifications ---------------------------------------------------------


def test_notifications_come_back_newest_first(store):
    """A log you scroll, not a queue you work through."""
    store.notify("startup", "server started")
    store.notify("restart", "reloading")
    rows = store.notifications()
    assert [r["message"] for r in rows] == ["reloading", "server started"]
    assert rows[0]["kind"] == "restart"


def test_the_log_is_trimmed_so_it_cannot_grow_forever(store):
    """It shares a database with the journal, which is the part that matters."""
    for i in range(30):
        store.notify("test", f"message {i}")
    store.trim_notifications(keep=10)
    rows = store.notifications()
    assert len(rows) == 10
    assert rows[0]["message"] == "message 29", "it kept the newest"
    assert rows[-1]["message"] == "message 20"


def test_trimming_leaves_the_journal_and_tasks_alone(store):
    store.remember("topic", "a note")
    store.add_task("a task")
    for i in range(5):
        store.notify("test", f"m{i}")
    store.trim_notifications(keep=1)
    assert store.recall("note")
    assert store.list_tasks()

"""Schema versioning, backups, and clearing history.

The thing this file is defending against is subtle: `CREATE TABLE IF NOT
EXISTS` builds a missing table and silently ignores an existing one whose
columns have changed. So a schema edit lands in the source, does nothing to the
database, and the first sign of trouble is an INSERT failing with "no such
column" some hours later, nowhere near the change that caused it.
"""

import shutil
import sqlite3

import pytest

from sevanya import migrations
from sevanya.store import SCHEMA, Store


@pytest.fixture
def raw(tmp_path):
    """A bare connection, for building databases in awkward states."""
    conn = sqlite3.connect(tmp_path / "raw.db")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# --- the thing IF NOT EXISTS doesn't do ------------------------------------


def test_if_not_exists_ignores_a_changed_definition(raw):
    """The premise. Pinned so nobody has to rediscover it.

    If this ever fails, SQLite has changed and the rest of this file is
    solving a problem that no longer exists.
    """
    raw.executescript("CREATE TABLE IF NOT EXISTS journal (id INTEGER PRIMARY KEY, note TEXT);")
    raw.executescript("CREATE TABLE IF NOT EXISTS journal (id INTEGER PRIMARY KEY, note TEXT, mood TEXT);")
    assert "mood" not in migrations.columns(raw, "journal")


def test_drift_notices_a_missing_column(raw):
    migrations.initialise(raw, SCHEMA)
    raw.execute("ALTER TABLE journal RENAME COLUMN note TO note_old")
    assert migrations.drift(raw) == ["journal.note is missing"]


def test_drift_notices_a_missing_table(raw):
    migrations.initialise(raw, SCHEMA)
    raw.execute("DROP TABLE task_list")
    assert any("task_list" in problem for problem in migrations.drift(raw))


def test_a_healthy_database_has_no_drift(raw):
    migrations.initialise(raw, SCHEMA)
    assert migrations.drift(raw) == []


def test_the_store_refuses_to_open_a_drifted_database(tmp_path):
    """Fail at startup with a clear message, not at INSERT with a puzzle.

    Opening anyway means the first symptom appears somewhere unrelated, at
    whichever write happens to touch the missing column first.
    """
    path = tmp_path / "drifted.db"
    Store(path).close()

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE journal RENAME COLUMN note TO note_old")
    conn.commit()
    conn.close()

    with pytest.raises(migrations.SchemaMismatch) as caught:
        Store(path)
    assert "journal.note is missing" in str(caught.value)
    assert "sevanya.db check" in str(caught.value), "the error should say what to run"


# --- versioning ------------------------------------------------------------


def test_a_fresh_database_is_stamped_at_the_current_version(raw):
    migrations.initialise(raw, SCHEMA)
    assert migrations.version(raw) == migrations.LATEST
    assert migrations.pending(raw) == []


def test_an_existing_database_gets_the_migrations_it_missed(raw, monkeypatch):
    """The case that matters: a database that predates versioning."""
    raw.executescript(SCHEMA)          # built the old way, never stamped
    raw.commit()
    assert migrations.version(raw) == 0

    def add_mood(conn):
        migrations.add_column(conn, "journal", "mood", "TEXT")

    monkeypatch.setattr(migrations, "MIGRATIONS", [(2, "journal.mood", add_mood)])
    monkeypatch.setattr(migrations, "LATEST", 2)

    applied = migrations.initialise(raw, SCHEMA)
    assert applied == ["2: journal.mood"]
    assert "mood" in migrations.columns(raw, "journal")
    assert migrations.version(raw) == 2


def test_migrations_do_not_run_twice(raw, monkeypatch):
    runs = []

    def once(conn):
        runs.append(True)
        migrations.add_column(conn, "journal", "mood", "TEXT")

    raw.executescript(SCHEMA); raw.commit()
    monkeypatch.setattr(migrations, "MIGRATIONS", [(2, "journal.mood", once)])
    monkeypatch.setattr(migrations, "LATEST", 2)

    migrations.initialise(raw, SCHEMA)
    migrations.initialise(raw, SCHEMA)
    assert len(runs) == 1


def test_add_column_is_safe_to_call_twice(raw):
    """SQLite has no ADD COLUMN IF NOT EXISTS, and repeating it is an error."""
    migrations.initialise(raw, SCHEMA)
    assert migrations.add_column(raw, "journal", "mood", "TEXT") is True
    assert migrations.add_column(raw, "journal", "mood", "TEXT") is False


def test_a_failing_migration_does_not_advance_the_version(raw, monkeypatch):
    """Otherwise the next start skips it and the database stays half-changed."""
    def broken(conn):
        raise sqlite3.OperationalError("no such table: nope")

    raw.executescript(SCHEMA); raw.commit()
    monkeypatch.setattr(migrations, "MIGRATIONS", [(2, "broken", broken)])
    with pytest.raises(sqlite3.OperationalError):
        migrations.apply_pending(raw)
    assert migrations.version(raw) == 0


# --- backups ---------------------------------------------------------------


def test_a_backup_includes_what_a_file_copy_would_miss(tmp_path):
    """The WAL trap.

    journal_mode is WAL, so recent commits can live in sevanya.db-wal rather
    than the main file. Copy the one file and the newest journal notes can be
    missing — silently, which is the worst way to lose them.
    """
    store = Store(tmp_path / "live.db")
    store.db.execute("PRAGMA wal_autocheckpoint = 0")   # keep it all in the WAL
    for i in range(50):
        store.remember(f"topic{i}", f"note number {i}")

    naive = tmp_path / "naive-copy.db"
    shutil.copy(tmp_path / "live.db", naive)

    proper = store.backup(tmp_path / "proper.db")

    def notes(path):
        """How many journal notes survived into this file.

        -1 means the copy isn't even a usable database: with the WAL never
        checkpointed, the schema itself is still in it, so the main file alone
        has no tables at all.
        """
        conn = sqlite3.connect(path)
        try:
            return conn.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
        except sqlite3.OperationalError:
            return -1
        finally:
            conn.close()

    assert notes(proper) == 50, "the proper backup should have everything"
    assert notes(naive) < 50, (
        "the file copy saw everything, so this test is no longer demonstrating "
        "the WAL problem — check the autocheckpoint pragma"
    )


def test_auto_backup_is_timestamped_and_lands_beside_the_database(tmp_path):
    store = Store(tmp_path / "live.db")
    store.remember("topic", "note")
    path = store.auto_backup()
    assert path.exists()
    assert path.parent == tmp_path / "backups"
    assert path.name.startswith("sevanya-")


# --- clearing history ------------------------------------------------------


@pytest.fixture
def populated(store):
    conversation = store.new_conversation(title="a real chat")
    store.append_message(conversation, "user", "how do decorators work?")
    store.append_message(conversation, "assistant", "…")
    store.remember("python/decorators", "clicked as a function returning a function", conversation)
    store.add_task("rewrite the parser loop", conversation)
    store.notify("startup", "server started")
    return store


def test_clearing_removes_the_chats(populated):
    removed = populated.clear_conversations()
    assert removed["conversations"] == 1
    assert removed["messages"] == 2
    assert populated.counts()["messages"] == 0


def test_clearing_keeps_the_journal_and_the_list(populated):
    """The entire point. She keeps what she learned; the transcripts go.

    It works because journal and task_list reference conversations ON DELETE
    SET NULL while messages is ON DELETE CASCADE — and only with foreign keys
    enforced on the connection.
    """
    populated.clear_conversations()
    counts = populated.counts()
    assert counts["journal"] == 1
    assert counts["task_list"] == 1
    assert counts["notifications"] == 1
    assert populated.recall("decorators"), "the note itself should still be readable"
    assert [row["task"] for row in populated.list_tasks()] == ["rewrite the parser loop"]


def test_the_link_back_to_a_deleted_chat_is_blanked_not_orphaned(populated):
    populated.clear_conversations()
    row = populated.db.execute("SELECT conversation_id FROM journal").fetchone()
    assert row["conversation_id"] is None


def test_recent_threads_can_be_kept(store):
    old = store.new_conversation(title="last month")
    store.append_message(old, "user", "old")
    store.db.execute("UPDATE conversations SET updated_at = datetime('now', '-40 days') WHERE id = ?",
                     (old,))
    store.db.commit()
    recent = store.new_conversation(title="today")
    store.append_message(recent, "user", "new")

    store.clear_conversations(keep_days=7)
    remaining = [row["title"] for row in store.list_conversations()]
    assert remaining == ["today"]


# --- old-shape rows --------------------------------------------------------


def test_valid_content_is_not_flagged(populated):
    assert populated.unloadable_messages() == []


@pytest.mark.parametrize("content,expected", [
    ("not json at all", "not valid JSON"),
    ('{"old": "shape"}', "expected a string or a list of blocks"),
    ('[{"no_type": true}]', "no 'type'"),
])
def test_old_shapes_are_found_before_they_break_a_reload(store, content, expected):
    """Content is stored as JSON and read back as whatever shape it went in as.

    Change the shape and old rows keep the old one, so reopening an old
    conversation fails in a way that has nothing to do with the new code.
    """
    conversation = store.new_conversation()
    store.db.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?,?,?)",
                     (conversation, "assistant", content))
    store.db.commit()

    broken = store.unloadable_messages()
    assert len(broken) == 1
    assert expected in broken[0]["why"]


def test_clearing_history_removes_the_unloadable_rows(store):
    conversation = store.new_conversation()
    store.db.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?,?,?)",
                     (conversation, "assistant", '{"old": "shape"}'))
    store.db.commit()
    assert store.unloadable_messages()

    store.clear_conversations()
    assert store.unloadable_messages() == []


# --- the command line ------------------------------------------------------


from sevanya import db as db_cli  # noqa: E402


def seed(path):
    store = Store(path)
    conversation = store.new_conversation(title="a chat")
    store.append_message(conversation, "user", "hello")
    store.remember("topic", "a note worth keeping", conversation)
    store.add_task("a task worth keeping")
    store.close()


def test_check_is_quiet_and_clean_on_a_healthy_database(tmp_path, capsys):
    path = tmp_path / "t.db"
    seed(path)
    assert db_cli.main(["--path", str(path), "check"]) == 0
    out = capsys.readouterr().out
    assert "schema drift: none" in out
    assert "all loadable" in out


def test_check_reports_a_bad_exit_when_something_is_wrong(tmp_path, capsys):
    """So it's usable in a script, not just readable by a person."""
    path = tmp_path / "t.db"
    seed(path)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (1,'assistant','{}')")
    conn.commit(); conn.close()

    assert db_cli.main(["--path", str(path), "check"]) == 1
    assert "can't load" in capsys.readouterr().out


def test_the_cli_explains_drift_rather_than_crashing(tmp_path, capsys):
    """Store raises on drift; the tool you'd reach for must not just traceback."""
    path = tmp_path / "t.db"
    seed(path)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE journal RENAME COLUMN note TO note_old")
    conn.commit(); conn.close()

    assert db_cli.main(["--path", str(path), "check"]) == 2
    assert "journal.note is missing" in capsys.readouterr().err


def test_clear_history_backs_up_before_deleting(tmp_path, capsys):
    """Irreversible by default is not a reasonable default."""
    path = tmp_path / "t.db"
    seed(path)
    assert db_cli.main(["--path", str(path), "clear-history", "--yes"]) == 0

    backups = list((tmp_path / "backups").glob("sevanya-*.db"))
    assert len(backups) == 1

    saved = sqlite3.connect(backups[0])
    assert saved.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    saved.close()

    store = Store(path)
    assert store.counts()["messages"] == 0
    assert store.counts()["journal"] == 1


def test_clear_history_asks_first(tmp_path, monkeypatch, capsys):
    path = tmp_path / "t.db"
    seed(path)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    assert db_cli.main(["--path", str(path), "clear-history"]) == 1
    assert "nothing done" in capsys.readouterr().out
    assert Store(path).counts()["messages"] == 1


def test_answering_yes_at_the_prompt_clears(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    seed(path)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert db_cli.main(["--path", str(path), "clear-history"]) == 0
    assert Store(path).counts()["messages"] == 0


def test_the_backup_can_be_skipped_for_people_who_insist(tmp_path):
    path = tmp_path / "t.db"
    seed(path)
    db_cli.main(["--path", str(path), "clear-history", "--yes", "--no-backup"])
    assert not (tmp_path / "backups").exists()


def test_backup_writes_where_it_is_told(tmp_path, capsys):
    path = tmp_path / "t.db"
    seed(path)
    destination = tmp_path / "somewhere" / "copy.db"
    assert db_cli.main(["--path", str(path), "backup", str(destination)]) == 0
    assert destination.exists()
    assert "WAL" in capsys.readouterr().out


def test_migrate_says_when_there_is_nothing_to_do(tmp_path, capsys):
    path = tmp_path / "t.db"
    seed(path)
    assert db_cli.main(["--path", str(path), "migrate"]) == 0
    assert "nothing to apply" in capsys.readouterr().out


def test_an_old_database_that_already_matches_gets_stamped(raw):
    """Yours, built before versioning existed.

    Without this it sits at 0 for good and reports itself out of date in every
    check, while being perfectly current — a false alarm that trains you to
    ignore the real one.
    """
    raw.executescript(SCHEMA)
    raw.commit()
    assert migrations.version(raw) == 0

    migrations.initialise(raw, SCHEMA)
    assert migrations.version(raw) == migrations.LATEST


def test_the_courtesy_stamp_does_not_fire_on_a_drifted_database(raw, monkeypatch):
    """The stamp that says "this old file already matches" must not lie.

    Two different facts share one number, so be precise about which is being
    guarded here: `version` records which migrations have run, and drift
    records whether the shape matches. A migration that genuinely ran is
    entitled to advance the version even if the file is refused afterwards for
    an unrelated reason — that's the next test. What must never happen is the
    end-of-initialise stamp declaring a drifted database current.
    """
    monkeypatch.setattr(migrations, "MIGRATIONS", [])   # nothing legitimately to run
    raw.executescript(SCHEMA)
    raw.execute("ALTER TABLE journal RENAME COLUMN note TO note_old")
    raw.commit()

    with pytest.raises(migrations.SchemaMismatch):
        migrations.initialise(raw, SCHEMA)
    assert migrations.version(raw) == 0, "it stamped a database it had just rejected"


def test_a_migration_that_ran_still_counts_even_if_the_file_is_refused(raw):
    """Recording it is honest: that ALTER really did happen.

    Pretending otherwise would re-run it on the next start, and a migration
    that isn't safe to repeat would then fail for a second, unrelated reason.
    """
    raw.executescript(SCHEMA)
    raw.execute("ALTER TABLE messages DROP COLUMN model")   # needs migration 2
    raw.execute("ALTER TABLE journal RENAME COLUMN note TO note_old")   # and is drifted
    migrations.set_version(raw, 1)
    raw.commit()

    with pytest.raises(migrations.SchemaMismatch, match="journal.note"):
        migrations.initialise(raw, SCHEMA)

    assert "model" in migrations.columns(raw, "messages"), "the migration didn't run"
    # LATEST rather than a literal: every pending migration ran, and hard-coding
    # the number here would mean editing this test for each new one, which is
    # how a test stops describing anything.
    assert migrations.version(raw) == migrations.LATEST, "they ran but weren't recorded"


def test_a_database_with_work_waiting_is_not_stamped_ahead(raw, monkeypatch):
    """Otherwise the pending migration is skipped forever."""
    def never_runs(conn):
        raise AssertionError("should not have been reached")

    raw.executescript(SCHEMA); raw.commit()
    monkeypatch.setattr(migrations, "LATEST", 5)
    monkeypatch.setattr(migrations, "MIGRATIONS", [(5, "future", never_runs)])
    # pending() is non-empty, so the stamp must not fire behind its back
    assert migrations.pending(raw)


# --- the crash that pre-empted the useful message --------------------------

ORIGINAL_TABLES = """
CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL
  REFERENCES conversations(id) ON DELETE CASCADE, role TEXT NOT NULL,
  content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE TABLE journal (id INTEGER PRIMARY KEY, topic TEXT NOT NULL, note TEXT NOT NULL,
  conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')));
"""


def test_a_changed_column_explains_itself_instead_of_crashing(tmp_path):
    """SCHEMA isn't purely additive — it creates indexes too.

    CREATE INDEX ... ON journal(topic) raises a bare "no such column" against a
    table whose columns have changed, and it does so before the drift check
    runs, so the message written to explain exactly this situation never got a
    chance to appear. Found by pointing the tool at a database shaped the way
    an overhaul would leave one.
    """
    path = tmp_path / "wrong.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE journal (id INTEGER PRIMARY KEY, note TEXT);")
    conn.commit(); conn.close()

    with pytest.raises(migrations.SchemaMismatch) as caught:
        Store(path)

    message = str(caught.value)
    assert "no such column" in message, "the underlying cause should still be shown"
    assert "journal.topic is missing" in message, "and what's actually wrong"
    assert "sevanya.db check" in message


def test_an_old_but_healthy_database_still_gains_new_tables(tmp_path):
    """The upgrade path this must not break.

    task_list and notifications were both added to an existing database this
    way. Refusing to open anything that doesn't already have them would turn
    every future addition into a manual migration for no reason.
    """
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(ORIGINAL_TABLES)
    conn.execute("INSERT INTO journal (topic, note) VALUES ('python', 'a real note')")
    conn.commit(); conn.close()

    store = Store(path)
    assert "task_list" in migrations.tables(store.db)
    assert "notifications" in migrations.tables(store.db)
    assert store.recall("real note"), "the existing journal must survive the upgrade"
    assert migrations.version(store.db) == migrations.LATEST


def test_the_cli_reports_a_wrong_shaped_database_rather_than_tracebacking(tmp_path, capsys):
    path = tmp_path / "wrong.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE journal (id INTEGER PRIMARY KEY, note TEXT);")
    conn.commit(); conn.close()

    assert db_cli.main(["--path", str(path), "check"]) == 2
    assert "journal.topic is missing" in capsys.readouterr().err

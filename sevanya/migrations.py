"""Schema versioning, because CREATE TABLE IF NOT EXISTS doesn't do it.

`IF NOT EXISTS` means *table*, not *schema*. SQLite sees the table is there and
stops reading — it never compares your definition against the real one. So the
moment a change adds or renames a column, store.py says one thing, the file on
disk says another, and nothing complains until an INSERT that looks obviously
correct dies with "no such column".

Two mechanisms here, and they do different jobs.

**Migrations** are the fix for a schema change: an explicit ALTER, recorded
against a version number, run once. Editing SCHEMA harder does nothing to a
database that already exists.

**The drift check** is the safety net for when someone forgets. It compares the
columns the code relies on against the columns that are actually there, at
startup, and refuses to run if they disagree — turning a confusing failure
hours later into a clear message before anything has been written.
"""

import sqlite3
from _collections_abc import Callable

# Bumped by adding to MIGRATIONS. A fresh database is stamped with this
# straight away, because SCHEMA already builds the current shape.
LATEST = 4


class SchemaMismatch(RuntimeError):
    """The database on disk isn't the shape the code expects."""


# What each table must have for the code to work. Not every column SQLite
# happens to hold — the ones something reads or writes. A missing entry here is
# how a drift check gives false confidence, so add to this in the same commit
# as the migration.
EXPECTED = {
    "conversations": {"id", "title", "kind", "created_at", "updated_at"},
    "messages": {"id", "conversation_id", "role", "content", "model", "created_at"},
    "journal": {"id", "topic", "note", "conversation_id", "created_at"},
    "task_list": {"id", "task", "done", "conversation_id", "created_at", "completed_at"},
    "notifications": {"id", "kind", "message", "created_at"},
    "settings": {"key", "value"},
}


# Each entry is (version, description, function(conn)). Write them to be safe
# to run twice — an old database that predates versioning gets all of them, and
# it may already have some of the changes.
#
#   def _add_mood(conn):
#       add_column(conn, "journal", "mood", "TEXT")
#
#   MIGRATIONS = [(2, "journal.mood", _add_mood)]
#
Migration = Callable[[sqlite3.Connection], None]


def _add_message_model(conn: sqlite3.Connection) -> None:
    """Record which model wrote each assistant turn.

    Added when the model became configurable. Existing rows keep NULL, which
    is the honest answer — they were written before anyone was tracking it,
    and guessing would poison the one thing the column is for.
    """
    add_column(conn, "messages", "model", "TEXT")


def _add_conversation_kind(conn: sqlite3.Connection) -> None:
    """Separate the conversations you had from errands she ran.

    Everything that already exists is yours by definition — sub-agents didn't
    exist when those rows were written — so the default backfills correctly
    without a second statement.
    """
    add_column(conn, "conversations", "kind", "TEXT NOT NULL DEFAULT 'chat'")


def _add_settings_table(conn: sqlite3.Connection) -> None:
    """A place for small global config, starting with the active teaching mode.

    A whole new table rather than a column, so — unlike add_column above —
    plain CREATE TABLE IF NOT EXISTS is correct here: a table that never
    existed before has no old shape to drift from.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


MIGRATIONS: list[tuple[int, str, Migration]] = [
    (2, "messages.model — which model wrote each turn", _add_message_model),
    (3, "conversations.kind — yours, or an errand she ran", _add_conversation_kind),
    (4, "settings — key/value, currently just the active mode", _add_settings_table),
]


# --- helpers for writing migrations ----------------------------------------


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def add_column(conn: sqlite3.Connection, table: str, name: str, declaration: str) -> bool:
    """ALTER TABLE ADD COLUMN, unless it's already there. Returns whether it ran.

    SQLite has no ADD COLUMN IF NOT EXISTS, and running it twice is an error,
    so every migration would otherwise need this check written out by hand —
    which is exactly the sort of thing that gets skipped once.
    """
    if name in columns(conn, table):
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
    return True


# --- version tracking ------------------------------------------------------


def version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def set_version(conn: sqlite3.Connection, value: int) -> None:
    # PRAGMA won't take a bound parameter, and `value` never comes from
    # anywhere but the MIGRATIONS list above.
    conn.execute(f"PRAGMA user_version = {int(value)}")


def is_empty(conn: sqlite3.Connection) -> bool:
    return not tables(conn)


# --- the two jobs ----------------------------------------------------------


def pending(conn: sqlite3.Connection) -> list[tuple[int, str, Migration]]:
    at = version(conn)
    return sorted((m for m in MIGRATIONS if m[0] > at), key=lambda m: m[0])


def apply_pending(conn: sqlite3.Connection) -> list[str]:
    """Run the migrations this database hasn't had. Returns their descriptions."""
    applied = []
    for number, description, run in pending(conn):
        run(conn)
        set_version(conn, number)
        conn.commit()
        applied.append(f"{number}: {description}")
    return applied


def drift(conn: sqlite3.Connection) -> list[str]:
    """What the code expects that the database doesn't have."""
    problems = []
    present = tables(conn)
    for table, expected in sorted(EXPECTED.items()):
        if table not in present:
            problems.append(f"table {table!r} is missing entirely")
            continue
        missing = expected - columns(conn, table)
        for column in sorted(missing):
            problems.append(f"{table}.{column} is missing")
    return problems


def initialise(conn: sqlite3.Connection, schema_sql: str) -> list[str]:
    """Build or upgrade the database, and refuse to run on one that's wrong.

    Order matters. SCHEMA first, so a fresh file gets everything; migrations
    second, since they assume the tables exist; the drift check last, because
    it's the one that catches a change made in SCHEMA alone and never migrated.
    """
    fresh = is_empty(conn)

    try:
        conn.executescript(schema_sql)
    except sqlite3.OperationalError as exc:
        # SCHEMA is not purely additive in practice: it also creates indexes,
        # and CREATE INDEX ... ON journal(topic) raises a bare "no such column"
        # against a table whose columns have changed. That happens *before* the
        # drift check below, so the useful message never gets a chance. Convert
        # it here, with the drift detail attached.
        problems = drift(conn)
        raise SchemaMismatch(
            f"the database couldn't be brought up to date: {exc}\n\n"
            + ("\n".join(f"  - {p}" for p in problems) + "\n\n" if problems else "")
            + "This is what CREATE TABLE IF NOT EXISTS can't tell you: it sees "
            "the table exists and never checks the columns.\n"
            "Back up and inspect it with:  python -m sevanya.db check"
        ) from exc

    if fresh:
        # SCHEMA is the current shape by definition, so there's nothing to
        # migrate — stamp it and move on.
        set_version(conn, LATEST)
        conn.commit()
        applied = []
    else:
        applied = apply_pending(conn)

    problems = drift(conn)
    if problems:
        raise SchemaMismatch(
            "the database doesn't match what the code expects:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nThis is what CREATE TABLE IF NOT EXISTS can't tell you: it "
            "sees the table exists and never checks the columns.\n"
            "Back up and inspect it with:  python -m sevanya.db check"
        )

    # A database built before versioning existed sits at 0 forever otherwise,
    # and reports itself as out of date in every check while being perfectly
    # current. Stamp it — but only here, after the drift check has confirmed
    # its shape really does match, and only when nothing is waiting to run.
    if version(conn) < LATEST and not pending(conn):
        set_version(conn, LATEST)
        conn.commit()

    return applied

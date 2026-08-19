"""Persistence: conversations, messages, and the learning journal.

Two things live in here, and they're different in kind.

The **conversation log** is mechanical — it's what lets you close the terminal
without losing your place, and later it's what lets phone-you and desktop-you
be in the same conversation.

The **journal** is the interesting one. It's what Sevanya has noticed about how
you're learning: what you worked on, where you got stuck, what clicked. That's
the part no off-the-shelf tool can have, because nobody else has your history.
It gets written by Sevanya itself through the `remember` tool, not by a script.
"""

import json
import shutil
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from . import migrations

# Lives in your home directory, not the project — Sevanya follows you across
# repos, and its memory of you shouldn't reset because you cd'd somewhere else.
DB_PATH = Path.home() / ".sevanya" / "sevanya.db"

# Marks the synthetic turn that opens an automatic check-in. Defined here as
# well as in checkin.py would be two places to get it wrong, so checkin.py
# imports nothing and this is the copy the SQL uses — they're asserted equal in
# the tests.
CHECKIN_MARKER = "[sevanya:auto]"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY,
    title      TEXT,
    -- 'chat' for yours, 'subagent' for work she delegated. Kept rather than
    -- discarded — when a delegated answer turns out to be wrong, reading what
    -- it actually did is the only way to find out why — but filtered out of
    -- the list, because they aren't conversations you had.
    kind       TEXT NOT NULL DEFAULT 'chat',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    -- JSON, because content is not always a string. An assistant turn is a
    -- *list of blocks* that can include tool_use and thinking blocks, and
    -- those have to survive a reload intact. See _jsonable below.
    content         TEXT NOT NULL,
    -- Which model produced this turn, for assistant turns. Null for anything
    -- written before the column existed, and for user turns. Worth keeping
    -- because the transcripts are training material for a local model, and
    -- material you can't attribute is material you can't filter.
    model           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS journal (
    id              INTEGER PRIMARY KEY,
    topic           TEXT NOT NULL,
    note            TEXT NOT NULL,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_journal_topic ON journal(topic);

-- What you've said you're going to do. Distinct from the journal: the journal
-- is what Sevanya noticed about how you're learning, and it's write-once. This
-- is a working list with a lifecycle — added, completed, or dropped when it
-- turns out not to matter.
CREATE TABLE IF NOT EXISTS task_list (
    id              INTEGER PRIMARY KEY,
    task            TEXT NOT NULL,
    done            INTEGER NOT NULL DEFAULT 0,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_list_open ON task_list(done, id);

-- A log of things that happened to Sevanya herself rather than in a
-- conversation: restarts, dependency installs, repos synced, errors. It exists
-- because most of these happen while nobody is looking at the terminal — the
-- whole point is that she runs on the PC and you're on the phone.
CREATE TABLE IF NOT EXISTS notifications (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,
    message    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_notifications_id ON notifications(id DESC);

-- Small key/value config, currently just which teaching mode is active.
-- Deliberately not conversation-scoped: like the model backend, this is one
-- global choice rather than something each thread remembers separately.
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _jsonable(content):
    """Make message content safe to store as JSON.

    A user turn's content is usually a plain string. An assistant turn's is a
    list of SDK block objects (pydantic models), which json.dumps can't handle.

    This matters more than it looks: if you stored only the *text* of an
    assistant turn, you'd silently drop its tool_use blocks. Reload that
    conversation and your tool_result blocks now answer requests the model
    can't see — and the API rejects the whole thing. Keep the blocks.
    """
    if isinstance(content, str):
        return content
    out = []
    for block in content:
        if hasattr(block, "model_dump"):
            # exclude_none keeps the stored rows clean; the SDK is happy to
            # receive dicts back in place of its own objects.
            out.append(block.model_dump(exclude_none=True))
        else:
            out.append(block)  # already a plain dict (our tool_result blocks)
    return out


def _readable(content) -> str:
    """Pull human text out of stored content, skipping tool plumbing.

    tool_result blocks are deliberately excluded: they hold whole files
    Sevanya read, and matching your search term inside someone's source dump
    is noise, not recall.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts)


def _window(text: str, query: str, pad: int = 120) -> str | None:
    """A slice of `text` around the first case-insensitive hit on `query`."""
    idx = text.lower().find(query.lower())
    if idx == -1:
        return None
    start, end = max(0, idx - pad), min(len(text), idx + len(query) + pad)
    excerpt = text[start:end].replace("\n", " ").strip()
    return ("…" if start else "") + excerpt + ("…" if end < len(text) else "")


class Store:
    """One SQLite connection per thread.

    SQLite connections are bound to the thread that opened them — hand one to
    another thread and it raises. The terminal REPL is single-threaded so this
    never mattered, but the web server runs each request in a threadpool, so a
    single shared connection breaks the moment two requests overlap.

    `self.db` is a property that hands back this thread's connection, creating
    it on first use. Every method below is unchanged as a result.
    """

    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._local = threading.local()
        # Build or upgrade, on whichever thread made the Store. Not a bare
        # executescript any more: that creates missing tables but silently
        # ignores a table whose columns have changed, so it can leave you with
        # a database the code can't actually use and no sign of it until an
        # INSERT fails. See migrations.py.
        self.applied = migrations.initialise(self.db, SCHEMA)
        self.db.commit()

    @property
    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # timeout: wait rather than erroring if another thread is mid-write.
            conn = sqlite3.connect(self._path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL lets readers work while a writer holds the file.
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    # --- conversations ------------------------------------------------------

    def new_conversation(self, title: str | None = None, kind: str = "chat") -> int:
        cur = self.db.execute(
            "INSERT INTO conversations (title, kind) VALUES (?, ?)", (title, kind))
        self.db.commit()
        return cur.lastrowid

    def latest_conversation(self) -> int | None:
        row = self.db.execute(
            "SELECT id FROM conversations ORDER BY updated_at DESC, id DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def conversation_exists(self, conversation_id: int) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return row is not None

    def list_conversations(self, limit: int = 10, kind: str | None = "chat") -> list[sqlite3.Row]:
        """Yours, newest first. Sub-agent runs are excluded by default.

        They're transcripts of errands, not conversations, and a list where
        every question you ask spawns three entries is a list you stop opening.
        """
        where = "WHERE c.kind = :kind" if kind else ""
        return self.db.execute(
            f"""SELECT c.id, c.title, c.updated_at, COUNT(m.id) AS n
               FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
               {where}
               GROUP BY c.id ORDER BY c.updated_at DESC LIMIT :limit""",
            {"limit": limit, "kind": kind},
        ).fetchall()

    def set_title(self, conversation_id: int, title: str) -> None:
        self.db.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
        )
        self.db.commit()

    # --- messages -----------------------------------------------------------

    def append_message(self, conversation_id: int, role: str, content,
                       model: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO messages (conversation_id, role, content, model) VALUES (?, ?, ?, ?)",
            (conversation_id, role, json.dumps(_jsonable(content)), model),
        )
        self.db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )
        self.db.commit()

    def load_messages(self, conversation_id: int) -> list[dict]:
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [{"role": r["role"], "content": json.loads(r["content"])} for r in rows]

    # --- journal ------------------------------------------------------------

    def remember(self, topic: str, note: str, conversation_id: int | None = None) -> None:
        self.db.execute(
            "INSERT INTO journal (topic, note, conversation_id) VALUES (?, ?, ?)",
            (topic, note, conversation_id),
        )
        self.db.commit()

    def recall(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        """Substring search over the journal.

        Deliberately dumb. Semantic search over a few hundred short notes
        would be more machinery than it's worth — revisit if LIKE actually
        starts failing you, not before.
        """
        like = f"%{query}%"
        return self.db.execute(
            """SELECT topic, note, created_at FROM journal
               WHERE topic LIKE ? OR note LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (like, like, limit),
        ).fetchall()

    def search_messages(self, query: str, limit: int = 6) -> list[dict]:
        """Substring search across past conversations.

        The journal is what Sevanya chose to write down; this is everything
        else. Most of what you'll want to refer back to ("that parser bug")
        never got a journal entry — it's just sitting in a transcript.

        Two filters keep the results useful rather than overwhelming:
        tool_result blocks are skipped (they're full file contents, and
        matching inside them tells you nothing), and each hit is trimmed to a
        window around the match instead of returning the whole message.
        """
        rows = self.db.execute(
            """SELECT m.conversation_id, m.role, m.content, m.created_at,
                      c.title
               FROM messages m JOIN conversations c ON c.id = m.conversation_id
               WHERE m.content LIKE ?
               ORDER BY m.id DESC LIMIT ?""",
            (f"%{query}%", limit * 4),  # over-fetch; filtering drops some
        ).fetchall()

        hits = []
        for row in rows:
            text = _readable(json.loads(row["content"]))
            if not text:
                continue
            window = _window(text, query)
            if window is None:
                continue  # matched only inside JSON scaffolding, not real text
            hits.append(
                {
                    "conversation_id": row["conversation_id"],
                    "title": row["title"],
                    "role": row["role"],
                    "when": row["created_at"],
                    "excerpt": window,
                }
            )
            if len(hits) >= limit:
                break
        return hits

    def recent_journal(self, limit: int = 5) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT topic, note, created_at FROM journal ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    # --- the task list ------------------------------------------------------
    #
    # CREATE TABLE IF NOT EXISTS means an existing ~/.sevanya/sevanya.db picks
    # this up the next time Sevanya starts. There's no migration to run.

    def add_task(self, task: str, conversation_id: int | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO task_list (task, conversation_id) VALUES (?, ?)",
            (task, conversation_id),
        )
        self.db.commit()
        return cur.lastrowid

    def get_task(self, task_id: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT id, task, done, created_at, completed_at FROM task_list WHERE id = ?",
            (task_id,),
        ).fetchone()

    def complete_task(self, task_id: int) -> bool:
        """Mark one done. False if there's no such task.

        Completing something already complete is not an error — it's a no-op
        with an honest answer, because the interesting failure is naming a task
        that doesn't exist.
        """
        cur = self.db.execute(
            "UPDATE task_list SET done = 1, completed_at = datetime('now') WHERE id = ?",
            (task_id,),
        )
        self.db.commit()
        return cur.rowcount > 0

    def reopen_task(self, task_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE task_list SET done = 0, completed_at = NULL WHERE id = ?",
            (task_id,),
        )
        self.db.commit()
        return cur.rowcount > 0

    def remove_task(self, task_id: int) -> bool:
        cur = self.db.execute("DELETE FROM task_list WHERE id = ?", (task_id,))
        self.db.commit()
        return cur.rowcount > 0

    def list_tasks(self, include_done: bool = False, limit: int = 50) -> list[sqlite3.Row]:
        """Open tasks oldest first — the order you'd work through them."""
        where = "" if include_done else "WHERE done = 0"
        return self.db.execute(
            f"""SELECT id, task, done, created_at, completed_at FROM task_list
                {where} ORDER BY done, id LIMIT ?""",
            (limit,),
        ).fetchall()

    def open_tasks(self, limit: int = 10) -> list[sqlite3.Row]:
        # Selects `done` as well, even though it's 0 for every row here, so
        # these rows have the same shape as list_tasks' and anything that
        # formats one can format the other. sqlite3.Row raises on a missing
        # column, so a narrower select turns a shared helper into a crash.
        return self.db.execute(
            """SELECT id, task, done, created_at FROM task_list
               WHERE done = 0 ORDER BY id LIMIT ?""",
            (limit,),
        ).fetchall()

    # --- when did anything last happen ---------------------------------------

    def last_human_message_at(self) -> str | None:
        """When the person last wrote something.

        Restricted to string content, which is what a typed message is — an
        assistant turn and a tool_result are both stored as JSON arrays — and
        excluding the synthetic prompt that opens an automatic check-in, or
        her own nudges would count as activity and she'd never nudge again.
        """
        row = self.db.execute(
            """SELECT MAX(created_at) AS latest FROM messages
               WHERE role = 'user'
                 AND content LIKE '"%'
                 AND content NOT LIKE ?""",
            (f'"{CHECKIN_MARKER}%',),
        ).fetchone()
        return row["latest"] if row and row["latest"] else None

    def last_checkin_at(self) -> str | None:
        row = self.db.execute(
            "SELECT MAX(created_at) AS latest FROM notifications WHERE kind = 'checkin'"
        ).fetchone()
        return row["latest"] if row and row["latest"] else None

    # --- notifications ------------------------------------------------------

    def notify(self, kind: str, message: str) -> int:
        cur = self.db.execute(
            "INSERT INTO notifications (kind, message) VALUES (?, ?)",
            (kind, message),
        )
        self.db.commit()
        return cur.lastrowid

    def notifications(self, limit: int = 200) -> list[sqlite3.Row]:
        """Newest first — this is a log you scroll, not a queue you work."""
        return self.db.execute(
            "SELECT id, kind, message, created_at FROM notifications ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def trim_notifications(self, keep: int = 500) -> int:
        """Drop the oldest once the log gets long.

        Unbounded it would grow forever for something nobody reads twice, and
        it shares a database with the journal, which is the part that matters.
        """
        cur = self.db.execute(
            """DELETE FROM notifications WHERE id NOT IN (
                   SELECT id FROM notifications ORDER BY id DESC LIMIT ?
               )""",
            (keep,),
        )
        self.db.commit()
        return cur.rowcount

    # --- settings -------------------------------------------------------------

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.db.execute(
            """INSERT INTO settings (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, value),
        )
        self.db.commit()

    # --- looking after the file ----------------------------------------------

    def backup(self, destination: Path) -> Path:
        """Copy the database somewhere safe, WAL included.

        Not shutil.copy. journal_mode is WAL, so recent commits can still be
        sitting in sevanya.db-wal rather than the main file — copying the one
        file can quietly lose the newest journal notes, which are the part
        worth having. sqlite3's own backup walks the live database and folds
        the WAL in, and it's safe to run while something else is writing.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(destination)
        try:
            self.db.backup(target)
        finally:
            target.close()
        return destination

    def auto_backup(self, directory: Path | None = None) -> Path:
        """A timestamped backup, for taking before anything destructive."""
        directory = Path(directory) if directory else self._path.parent / "backups"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.backup(directory / f"sevanya-{stamp}.db")

    def list_backups(self, directory: Path | None = None, limit: int = 10) -> list[dict]:
        """Recent backups, newest first. Shown so you can see one was taken."""
        directory = Path(directory) if directory else self._path.parent / "backups"
        if not directory.is_dir():
            return []
        found = sorted(directory.glob("sevanya-*.db"), key=lambda p: p.name, reverse=True)
        return [{"name": p.name, "bytes": p.stat().st_size} for p in found[:limit]]

    def counts(self) -> dict[str, int]:
        """How much of each thing there is. For showing before and after."""
        out = {}
        for table in ("conversations", "messages", "journal", "task_list", "notifications"):
            out[table] = self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return out

    def clear_conversations(self, keep_days: int = 0) -> dict[str, int]:
        """Delete chat history. Keep everything she learned from it.

        The journal and the task list both reference conversations with
        ON DELETE SET NULL, and messages with ON DELETE CASCADE — so removing a
        conversation takes its messages and leaves her notes and your tasks
        standing, with the link blanked. That's the whole reason those two
        clauses differ, and it only works with foreign keys enforced, which is
        set on every connection.

        keep_days keeps recent threads; 0 clears the lot.
        """
        before = self.counts()

        if keep_days > 0:
            self.db.execute(
                "DELETE FROM conversations WHERE updated_at < datetime('now', ?)",
                (f"-{int(keep_days)} days",),
            )
        else:
            self.db.execute("DELETE FROM conversations")
        self.db.commit()

        after = self.counts()
        return {table: before[table] - after[table] for table in before}

    def unloadable_messages(self) -> list[dict]:
        """Rows that load_messages would choke on.

        Content is stored as JSON and read back as whatever shape it was
        written in. Change that shape and old rows keep the old one, so
        reopening an old conversation breaks in a way that has nothing to do
        with the new code. This finds them before they find you.
        """
        broken = []
        for row in self.db.execute(
            "SELECT id, conversation_id, role, content FROM messages ORDER BY id"
        ):
            try:
                content = json.loads(row["content"])
            except Exception as exc:
                broken.append({"id": row["id"], "conversation_id": row["conversation_id"],
                               "why": f"not valid JSON ({type(exc).__name__})"})
                continue

            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                broken.append({"id": row["id"], "conversation_id": row["conversation_id"],
                               "why": f"content is {type(content).__name__}, expected a string or a list of blocks"})
                continue
            for block in content:
                if not isinstance(block, dict) or "type" not in block:
                    broken.append({"id": row["id"], "conversation_id": row["conversation_id"],
                                   "why": "a content block has no 'type'"})
                    break
        return broken

    def conversations_for_export(self, model: str | None = None,
                                 min_messages: int = 2) -> list[int]:
        """Conversation ids worth training on.

        `model` narrows it to threads a particular model answered — which is
        the useful case: taking what Claude wrote and teaching a local model
        to do the same. Short threads are dropped because a one-line exchange
        teaches a model nothing except how to be brief.
        """
        if model:
            rows = self.db.execute(
                """SELECT conversation_id FROM messages
                   WHERE model LIKE ?
                   GROUP BY conversation_id
                   HAVING COUNT(*) >= 1""",
                (f"%{model}%",),
            ).fetchall()
            candidates = {row["conversation_id"] for row in rows}
        else:
            candidates = None

        rows = self.db.execute(
            """SELECT conversation_id, COUNT(*) AS n FROM messages
               GROUP BY conversation_id HAVING n >= ? ORDER BY conversation_id""",
            (min_messages,),
        ).fetchall()
        return [r["conversation_id"] for r in rows
                if candidates is None or r["conversation_id"] in candidates]

    def close(self) -> None:
        """Close this thread's connection. Other threads keep their own."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

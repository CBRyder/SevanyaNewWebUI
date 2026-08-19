"""Looking after the database.

    python -m sevanya.db check           what's in there, and what's wrong with it
    python -m sevanya.db backup          a WAL-safe copy
    python -m sevanya.db migrate         apply any pending schema changes
    python -m sevanya.db clear-history   delete chats, keep the journal and the list

The one worth running before any overhaul is `backup`. It takes a second and it
makes everything after it reversible.
"""

import argparse
import json
import sys
from pathlib import Path

from . import backends, migrations
from .prompt import SYSTEM
from .store import CHECKIN_MARKER, DB_PATH, Store


def check(store: Store) -> int:
    print(f"database: {store._path}")
    print(f"schema version: {migrations.version(store.db)} (code expects {migrations.LATEST})")

    waiting = migrations.pending(store.db)
    if waiting:
        print("\npending migrations:")
        for number, description, _ in waiting:
            print(f"  {number}: {description}")
        print("  run: python -m sevanya.db migrate")
    else:
        print("pending migrations: none")

    # Store refuses to open on drift, so reaching here with problems means
    # someone has been editing tables by hand — which is exactly when you'd
    # run this command.
    problems = migrations.drift(store.db)
    if problems:
        print("\nSCHEMA DRIFT — the file doesn't match the code:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("schema drift: none")

    print("\ncontents:")
    for table, count in store.counts().items():
        print(f"  {table:<14} {count}")

    broken = store.unloadable_messages()
    if broken:
        plural = "message is" if len(broken) == 1 else "messages are"
        print(f"\n{len(broken)} {plural} in a shape the code can't load:")
        for row in broken[:10]:
            print(f"  message {row['id']} (conversation {row['conversation_id']}): {row['why']}")
        if len(broken) > 10:
            print(f"  ... and {len(broken) - 10} more")
        print("  they're in old conversations — clear-history removes them")
    else:
        print("\nmessage shapes: all loadable")

    return 1 if (problems or broken) else 0


def backup(store: Store, destination: str | None) -> int:
    path = store.backup(Path(destination)) if destination else store.auto_backup()
    print(f"backed up to {path} ({path.stat().st_size:,} bytes)")
    print("this folds in the WAL, so it includes commits a file copy would miss")
    return 0


def migrate(store: Store) -> int:
    # Store.__init__ has already run them by the time we get here — opening the
    # database is what applies them. This reports what that did.
    if store.applied:
        print("applied:")
        for line in store.applied:
            print(f"  {line}")
    else:
        print(f"nothing to apply — already at version {migrations.version(store.db)}")
    return 0


def clear_history(store: Store, keep_days: int, assume_yes: bool, skip_backup: bool) -> int:
    counts = store.counts()
    print(f"database: {store._path}")
    print(f"  {counts['conversations']} conversations, {counts['messages']} messages"
          + (f" — keeping anything touched in the last {keep_days} days" if keep_days else ""))
    print(f"  keeping: {counts['journal']} journal notes, {counts['task_list']} tasks, "
          f"{counts['notifications']} notices")

    if not assume_yes:
        # Deleting someone's history on a mistyped command is not a thing to
        # be casual about, so the default is to ask.
        answer = input("\ndelete the conversations? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("nothing done")
            return 1

    if not skip_backup:
        path = store.auto_backup()
        print(f"backed up to {path} first")

    removed = store.clear_conversations(keep_days=keep_days)
    print(f"removed {removed['conversations']} conversations and {removed['messages']} messages")

    remaining = store.counts()
    print(f"kept {remaining['journal']} journal notes and {remaining['task_list']} tasks")
    if removed["journal"] or removed["task_list"]:
        # Should be impossible: both reference conversations ON DELETE SET
        # NULL. If it ever isn't, say so loudly rather than in a diff nobody
        # reads.
        print("WARNING: something was removed from the journal or the task list —"
              " restore the backup and check the foreign keys", file=sys.stderr)
        return 1
    return 0


def export(store: Store, destination: str, model: str | None, min_messages: int) -> int:
    """Write the conversations out as training data.

    Deliberately reuses the same converter that talks to a local model, so
    what comes out is in the shape the tools that fine-tune one already read —
    and so there's one definition of "this transcript as chat messages"
    rather than two that drift.

    Automatic check-ins are left out. Their opening turn is an instruction to
    her, not something you said, and training on it would teach a model that
    users talk like a cron job.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    ids = store.conversations_for_export(model=model, min_messages=min_messages)
    written = skipped = 0

    with path.open("w", encoding="utf-8") as out:
        for conversation_id in ids:
            messages = [
                m for m in store.load_messages(conversation_id)
                if not (isinstance(m["content"], str) and m["content"].startswith(CHECKIN_MARKER))
            ]
            if len(messages) < min_messages:
                skipped += 1
                continue
            converted = backends.messages_for_openai(SYSTEM, messages)
            if not any(m["role"] == "assistant" for m in converted):
                skipped += 1      # nothing to learn from a thread she never answered
                continue
            out.write(json.dumps({"messages": converted}, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} conversations to {path}")
    if skipped:
        print(f"skipped {skipped} (too short, or she never answered)")
    if model:
        print(f"filtered to threads answered by {model!r}")
    print("one JSON object per line, OpenAI chat format — what fine-tuning tools read")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sevanya.db", description=__doc__)
    parser.add_argument("--path", type=Path, default=DB_PATH, help="database file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="version, drift, contents, and any unloadable rows")

    backup_parser = sub.add_parser("backup", help="a WAL-safe copy")
    backup_parser.add_argument("destination", nargs="?", help="where to write it")

    sub.add_parser("migrate", help="apply pending schema changes")

    clear_parser = sub.add_parser("clear-history",
                                  help="delete conversations, keep the journal and the list")
    clear_parser.add_argument("--keep-days", type=int, default=0,
                              help="keep threads touched in the last N days")
    clear_parser.add_argument("--yes", action="store_true", help="don't ask")
    clear_parser.add_argument("--no-backup", action="store_true",
                              help="skip the automatic backup (don't)")

    export_parser = sub.add_parser("export",
                                   help="write conversations out as training data (JSONL)")
    export_parser.add_argument("out", help="file to write")
    export_parser.add_argument("--model",
                               help="only threads answered by this model, e.g. 'claude'")
    export_parser.add_argument("--min-messages", type=int, default=2,
                               help="skip threads shorter than this (default 2)")

    args = parser.parse_args(argv)

    try:
        store = Store(args.path)
    except migrations.SchemaMismatch as exc:
        # The one case where refusing to open is the whole point.
        print(f"{exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "check":
            return check(store)
        if args.command == "backup":
            return backup(store, args.destination)
        if args.command == "migrate":
            return migrate(store)
        if args.command == "clear-history":
            return clear_history(store, args.keep_days, args.yes, args.no_backup)
        if args.command == "export":
            return export(store, args.out, args.model, args.min_messages)
    finally:
        store.close()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Terminal REPL.

    python -m sevanya.main           resume the most recent conversation
    python -m sevanya.main --new     start a fresh one
    python -m sevanya.main --list    show recent conversations
    python -m sevanya.main --id 7    resume a specific one

Type /rc mid-conversation to bring the web server up without leaving.
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .agent import Agent
from .store import DB_PATH, Store
from .tools import PROJECT_ROOT

# Read straight from the environment rather than importing server.py, which
# pulls in fastapi and builds a Store just to learn one number.
PORT = int(os.environ.get("SEVANYA_PORT", "8765"))


def _server_is_up(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def start_web(port: int = PORT, wait: float = 25.0) -> str:
    """Start the web server in its own process. Returns a line to print.

    Launched from this directory on purpose: PROJECT_ROOT is wherever the
    process starts, and that's the only place she can read. Starting the
    server somewhere else would quietly give the phone a different view of
    your files than the terminal has.

    `python -m sevanya`, not `-m sevanya.server`, so it checks requirements
    before importing anything that needs them.
    """
    if _server_is_up(port):
        return f"already running — http://localhost:{port}"

    if os.name == "nt":
        # Its own console window, so it doesn't scribble over this
        # conversation and can be stopped without killing the REPL.
        creation = subprocess.CREATE_NEW_CONSOLE
        process = subprocess.Popen([sys.executable, "-m", "sevanya"],
                                   cwd=str(PROJECT_ROOT), creationflags=creation)
    else:
        process = subprocess.Popen([sys.executable, "-m", "sevanya"],
                                   cwd=str(PROJECT_ROOT), start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.time() + wait
    while time.time() < deadline:
        if _server_is_up(port):
            return f"web server up — http://localhost:{port}  (pid {process.pid})"
        if process.poll() is not None:
            return (f"the server exited immediately (code {process.returncode}). "
                    f"Run `python -m sevanya` here to see why.")
        time.sleep(0.4)

    return f"started (pid {process.pid}) but nothing answered on :{port} within {wait:g}s"


def main() -> int:
    parser = argparse.ArgumentParser(prog="sevanya")
    parser.add_argument("--new", action="store_true", help="start a new conversation")
    parser.add_argument("--list", action="store_true", help="list conversations and exit")
    parser.add_argument("--id", type=int, help="resume a specific conversation")
    args = parser.parse_args()

    store = Store()

    if args.list:
        for row in store.list_conversations():
            title = row["title"] or "(untitled)"
            print(f"{row['id']:>4}  {row['updated_at']}  {row['n']:>3} msgs  {title}")
        return 0

    if args.new:
        conversation_id = store.new_conversation()
    elif args.id is not None:
        conversation_id = args.id
    else:
        conversation_id = store.latest_conversation() or store.new_conversation()

    agent = Agent(store, conversation_id)

    print(f"Sevanya — conversation {conversation_id}, reading from {PROJECT_ROOT}")
    print(f"memory: {DB_PATH}")
    if agent.messages:
        print(f"resumed with {len(agent.messages)} messages of history")
    print("ctrl-c or 'exit' to quit\n")

    # First user message doubles as the conversation title, so `--list` is
    # readable instead of a wall of "(untitled)".
    needs_title = not agent.messages

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"exit", "quit"}:
            return 0

        # Handled here rather than sent to her: it's an instruction to this
        # program, not a question for her. Every other slash command — /show
        # among them — goes through untouched, because those are hers.
        if user_input.lower() in {"/rc", "/web"}:
            print(f"\n{start_web()}\n")
            continue

        if needs_title:
            store.set_title(conversation_id, user_input[:60])
            needs_title = False

        try:
            reply = agent.send(user_input)
        except Exception as exc:
            # Keep the session alive on a transient API failure — losing the
            # whole conversation to one network blip is infuriating. The
            # transcript is already on disk, so nothing is lost either way.
            print(f"\n[error: {type(exc).__name__}: {exc}]\n", file=sys.stderr)
            continue

        print(f"\nsevanya> {reply}\n")


if __name__ == "__main__":
    raise SystemExit(main())

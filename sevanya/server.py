"""HTTP server — the piece your phone talks to.

Two endpoints, deliberately different in shape:

  POST /api/chat   streams tokens as they arrive. For the web UI, where you
                   want to watch it think.

  POST /api/ask    blocks, returns one blob of text. For Siri, which cannot
                   consume a stream — a Shortcut sends one request, waits, and
                   speaks whatever comes back.

Everything intelligent stays here on the PC. The phone is a thin client.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import backends, checkin, deps, lifecycle, migrations, push
from .agent import Agent
from .prompt import DEFAULT_MODE, MODES
from .store import CHECKIN_MARKER, Store
from .tools import PROJECT_ROOT

# The UI lives at the repo root here, not in a sevanya/web/ subpackage —
# it's built by hand, separately from everything else in this file, and this
# just serves whatever ends up there. Nothing under WEB_DIR is written by
# this codebase.
WEB_DIR = Path(__file__).resolve().parent.parent

# Set SEVANYA_TOKEN to require a bearer token. Over Tailscale nothing else can
# reach this anyway, but a token costs nothing and means one misconfigured
# firewall rule isn't the only thing standing between the internet and a shell
# on your machine.
TOKEN = os.environ.get("SEVANYA_TOKEN")

# 8765 unless told otherwise. Configurable mostly so a second copy — a test,
# or a throwaway one pointed at another project — doesn't have to fight the
# one you actually use for the port.
PORT = int(os.environ.get("SEVANYA_PORT", "8765"))

# Appended to the system prompt for /ask only. Siri reads the reply aloud, and
# anything longer than a few sentences is unbearable through a speaker.
SPOKEN = """
You are being read aloud by a voice assistant. Answer in two or three sentences
of plain prose. No code blocks, no bullet lists, no file paths unless the answer
is meaningless without one. If the honest answer needs more room than that, give
the short version and say the detail is worth looking at on a screen.
"""

# When this process image started. execv keeps the PID, so the pid alone can't
# tell "it restarted" from "nothing happened" — this can, and the restart
# button uses it to confirm the server really came back rather than never
# having gone away.
STARTED = time.time()

# Which model is answering. Resolved once, at startup, so the page can show it
# without every request paying for the lookup — and so a misconfigured backend
# is a startup error rather than a surprise on the first question.
try:
    BACKEND = backends.choose().describe()
except Exception as exc:  # pragma: no cover - only on a bad config
    BACKEND = f"unavailable ({type(exc).__name__}: {exc})"

app = FastAPI(title="Sevanya")
store = Store()

# Tells the `reload` tool there's a server here to restart. The terminal REPL
# never imports this module, so there it correctly reports nothing to do.
lifecycle.mark_server_running()

# Check the requirements on the way up and record the answer, so a start that
# lands in a broken environment leaves a note saying so — otherwise the only
# evidence is a traceback in a terminal nobody is looking at.
#
# This is the backstop. `python -m sevanya` has already done it before
# importing anything, which is the only place a genuinely missing dependency
# can be fixed; this catches the case where someone started the server module
# directly, and anything that went missing since.
# SEVANYA_SKIP_DEPS means skip the check, not "check but don't fix" — a flag
# that half-applies is worse than no flag.
if os.environ.get("SEVANYA_SKIP_DEPS"):
    _deps_ok, _deps_report = True, "requirements not checked (SEVANYA_SKIP_DEPS)"
else:
    _deps_ok, _deps_report = deps.ensure(PROJECT_ROOT, install_missing=True)
    if push.is_public():
        # Not an error, but not something to discover later either.
        store.notify(
            "push",
            "notifications are going through the public ntfy.sh — anyone who "
            "guesses the topic can read them. See deploy/ntfy-compose.yml to "
            "self-host.",
        )

    if not _deps_ok:
        # A server that came up wrong is exactly the thing you'd otherwise
        # only discover by walking to the machine.
        push.send_and_log(store, _deps_report.splitlines()[0],
                          title="Sevanya started with problems", priority="high")
store.notify("startup", f"server started — {_deps_report}")
store.trim_notifications()

# App icons, referenced by manifest.json and the apple-touch-icon link. iOS
# will happily "Add to Home Screen" without them and give you a blurry
# screenshot of the page as the icon, which looks like something half-built.
#
# Mounted only if the directory exists — the UI here is built by hand and
# static/ may not exist yet on a fresh checkout. Failing to mount would be a
# missing icon; failing to *start* over a directory nobody has made yet is
# the wrong trade.
if (WEB_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


class ChatIn(BaseModel):
    message: str
    conversation_id: int | None = None


class AskIn(BaseModel):
    message: str


def _auth(authorization: str | None) -> None:
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="bad or missing token")


class ModeIn(BaseModel):
    name: str


def _current_mode() -> str:
    name = store.get_setting("mode", DEFAULT_MODE)
    # Falls back rather than KeyErrors on every request if a stored value
    # ever names a mode that's since been renamed or dropped.
    return name if name in MODES else DEFAULT_MODE


def _mode_extra() -> str:
    return MODES[_current_mode()]["addendum"]


@app.post("/api/chat")
def chat(body: ChatIn, authorization: str | None = Header(default=None)):
    """Streaming chat for the web UI.

    Server-Sent Events: one JSON object per line, prefixed `data: `. The browser
    reads them as they arrive rather than waiting for the whole reply.
    """
    _auth(authorization)

    # No id supplied means a new thread. Deliberately *not* "latest" — Siri
    # creates its own conversations, and grabbing the newest would drop you
    # into a voice thread the moment you opened the web UI. The browser
    # remembers its own conversation id instead (localStorage) and sends it.
    conversation_id = body.conversation_id
    if conversation_id is None:
        # Title it from the opening message, the way the REPL does. Without
        # this every thread the browser starts is "(untitled)" in the picker,
        # which makes the picker useless for the one job it has.
        conversation_id = store.new_conversation(title=body.message[:60])
    agent = Agent(store, conversation_id, system_extra=_mode_extra())

    def events():
        # Tell the client which conversation this is, so a phone that didn't
        # specify one can keep using the same thread on its next message.
        yield f"data: {json.dumps({'type': 'conversation', 'id': conversation_id})}\n\n"
        try:
            for kind, payload in agent.stream(body.message):
                yield f"data: {json.dumps({'type': kind, 'text': payload})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': f'{type(exc).__name__}: {exc}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ask")
def ask(body: AskIn, authorization: str | None = Header(default=None)):
    """One-shot endpoint for Siri.

    Every request starts a **fresh conversation**, by design. Continuity comes
    from `recall` searching the journal and past transcripts when you refer to
    something — not from dragging a whole transcript along. That keeps a voice
    question from polluting the context of a coding session you have open at
    your desk.
    """
    _auth(authorization)

    conversation_id = store.new_conversation(title=f"siri: {body.message[:50]}")
    # Mode first, brevity rule last — the voice constraint sits closest to
    # the end of the prompt regardless of which mode is active.
    agent = Agent(store, conversation_id, system_extra=_mode_extra() + SPOKEN)
    return {"reply": agent.send(body.message), "conversation_id": conversation_id}


@app.get("/api/health")
def health():
    """Is the server there, and does it want a token?

    Deliberately the one endpoint that doesn't require auth. The client needs
    to tell "the server is gone" apart from "the server is fine and my token is
    wrong", and it can't do that if the check itself returns 401 in both cases.
    It reveals only whether a token is needed, which anyone who can reach this
    port finds out from their first request anyway.
    """
    return {"ok": True, "auth_required": bool(TOKEN), "started": STARTED,
            "backend": BACKEND}


@app.post("/api/restart")
def restart(authorization: str | None = Header(default=None)):
    """Pull the latest commit, then restart the server process onto it.

    Requires the token when one is set — this is the one endpoint that does
    something to the machine rather than reading from it. With no token set,
    anything that can reach the port can bounce the server; that's the same
    exposure as every other endpoint here, but it's worth knowing.

    Editing index.html away from this machine — GitHub's own editor on a
    phone, say — only ever changes GitHub. This is the step that gets it
    onto disk here: a fast-forward pull, immediately before the restart that
    serves it. If it can't fast-forward — a real conflict, or this machine
    offline — nothing restarts. Restarting anyway would mean pressing this
    button and getting a restart with none of what you asked for in it,
    which is worse than the button just refusing and saying why.

    In-flight streams die with the old process. The client polls /api/health
    and picks its thread back up from the database, which is on disk and
    unaffected.

    SEVANYA_SKIP_PULL goes back to a plain restart — for tests, or a machine
    that was never meant to track a remote at all.
    """
    _auth(authorization)

    if os.environ.get("SEVANYA_SKIP_PULL"):
        pulled = "skipped (SEVANYA_SKIP_PULL)"
    else:
        ok, message = lifecycle.pull_latest(WEB_DIR)
        if not ok:
            store.notify("restart-failed", f"pull failed, not restarting: {message}")
            raise HTTPException(
                status_code=409, detail=f"pull failed, not restarting: {message}"
            )
        pulled = message

    store.notify("restart", f"restarting — {pulled}")
    lifecycle.schedule_restart()
    return {"restarting": True, "pid": os.getpid(), "pulled": pulled}


class ClearIn(BaseModel):
    # Not a default of True, and not inferred from the request having been
    # made. Deleting history needs something that can only have come from
    # someone deciding to.
    confirm: bool = False
    keep_days: int = 0


@app.get("/api/db")
def database(authorization: str | None = Header(default=None)):
    """The state of the database, for deciding whether to touch it.

    Everything `python -m sevanya.db check` prints, as JSON — so the same
    decision can be made from a phone instead of from a terminal on the machine
    it's running on.
    """
    _auth(authorization)
    unloadable = store.unloadable_messages()
    return {
        "backend": BACKEND,
        "path": str(store._path),
        "schema_version": migrations.version(store.db),
        "schema_expects": migrations.LATEST,
        "pending_migrations": [f"{n}: {d}" for n, d, _ in migrations.pending(store.db)],
        "drift": migrations.drift(store.db),
        "counts": store.counts(),
        "unloadable": len(unloadable),
        "unloadable_sample": unloadable[:5],
        "backups": store.list_backups(),
    }


@app.post("/api/db/backup")
def backup(authorization: str | None = Header(default=None)):
    _auth(authorization)
    path = store.auto_backup()
    store.notify("backup", f"backed up to {path.name}")
    return {"name": path.name, "bytes": path.stat().st_size}


@app.post("/api/db/clear-history")
def clear_history(body: ClearIn, authorization: str | None = Header(default=None)):
    """Delete conversations. Keep the journal, the task list and the notices.

    Takes a backup first, always — there's no flag to skip it here. On the
    command line you're standing at the machine and can argue; over HTTP,
    from a phone, one mis-tap should not be the end of the transcripts.
    """
    _auth(authorization)
    if not body.confirm:
        raise HTTPException(status_code=400, detail="confirm must be true")

    saved = store.auto_backup()
    removed = store.clear_conversations(keep_days=body.keep_days)
    store.notify(
        "clear-history",
        f"cleared {removed['conversations']} conversations and "
        f"{removed['messages']} messages (backup: {saved.name})",
    )
    return {"removed": removed, "backup": saved.name, "counts": store.counts()}


@app.get("/api/notifications")
def notifications(authorization: str | None = Header(default=None)):
    """What's happened to her lately — restarts, installs, errors.

    Read-only, like the task list: this is a log of things that happened, and
    there is nothing here for a client to change.
    """
    _auth(authorization)
    return [dict(row) for row in store.notifications()]


@app.get("/api/modes")
def modes(authorization: str | None = Header(default=None)):
    """How she's teaching right now, and what else she could do instead.

    Unlike the task list and notifications, this isn't read-only — it's what
    a mode picker in the UI needs: the current choice plus every option,
    in one call.
    """
    _auth(authorization)
    return {
        "current": _current_mode(),
        "modes": [
            {"name": name, "label": info["label"], "description": info["description"]}
            for name, info in MODES.items()
        ],
    }


@app.post("/api/mode")
def set_mode(body: ModeIn, authorization: str | None = Header(default=None)):
    """Change how she teaches, globally — one active mode, like the model.

    Takes effect on the next message, not this one: /api/chat builds a fresh
    Agent per request and reads the current mode when it does, so there's
    nothing to restart.
    """
    _auth(authorization)
    if body.name not in MODES:
        raise HTTPException(
            status_code=400,
            detail=f"no such mode {body.name!r} — have: {', '.join(MODES)}",
        )
    store.set_setting("mode", body.name)
    store.notify("mode", f"switched to {body.name}")
    return {"mode": body.name}


@app.get("/api/tasks")
def tasks(include_done: bool = True, authorization: str | None = Header(default=None)):
    """The task list, for reading.

    Read-only by design: this is Sevanya's list of what she thinks you should
    do, not a to-do app you fill in. There's no endpoint to add or tick one
    off from the phone, because the list means nothing if it's yours.
    """
    _auth(authorization)
    return [dict(row) for row in store.list_tasks(include_done=include_done)]


@app.get("/api/conversations")
def conversations(authorization: str | None = Header(default=None)):
    _auth(authorization)
    return [dict(r) for r in store.list_conversations(limit=25)]


@app.get("/api/conversations/{conversation_id}")
def history(conversation_id: int, authorization: str | None = Header(default=None)):
    """Readable transcript, for a phone rejoining a conversation.

    Tool plumbing is stripped — the phone wants the conversation, not the
    tool_use blocks that made it work.
    """
    _auth(authorization)

    # Say plainly that it's gone, rather than returning []. A phone holding a
    # stale id in localStorage needs to be able to tell "empty conversation"
    # from "that conversation no longer exists" so it can drop the id.
    if not store.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="no such conversation")

    out = []
    for message in store.load_messages(conversation_id):
        content = message["content"]
        if isinstance(content, str):
            # The prompt that opens an automatic check-in is an instruction to
            # her, not something the user said. Showing it would put words in
            # their mouth in their own transcript.
            if content.startswith(CHECKIN_MARKER):
                continue
            text = content
        else:
            text = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if text.strip():
            out.append({"role": message["role"], "text": text})
    return out


@app.get("/")
def index():
    path = WEB_DIR / "index.html"
    if not path.is_file():
        # Same reasoning as /manifest.json: while the UI is being built by
        # hand this file may not exist yet, or may be mid-rewrite — a 404
        # says so plainly instead of a 500 from FileResponse's stat().
        raise HTTPException(status_code=404, detail="no index.html yet")
    return FileResponse(path)


@app.get("/manifest.json")
def manifest():
    path = WEB_DIR / "manifest.json"
    if not path.is_file():
        # A missing manifest means no home-screen icon, not a broken server —
        # a plain 404 rather than the 500 FileResponse would raise on a stat()
        # of a path that isn't there yet.
        raise HTTPException(status_code=404, detail="no manifest.json yet")
    return FileResponse(path)


def main() -> None:
    import uvicorn

    print(f"Sevanya server — reading from {PROJECT_ROOT}")
    print(f"auth: {'bearer token required' if TOKEN else 'OPEN (set SEVANYA_TOKEN to lock)'}")
    print(f"listening on :{PORT}")

    # Started here rather than at import, so importing the module for a test
    # doesn't spawn a thread that talks to the API.
    if checkin.start(store):
        print(f"check-in: after {checkin.interval_hours():g}h of quiet")
    # 0.0.0.0 so your phone can reach it over Tailscale. On a machine that is
    # NOT on a private network, this listens on every interface — set a token.
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()

# Sevanya

The backend from `A.I.-Sevanya`, ported here so it sits next to a UI built
by hand from scratch. This file only covers the backend — how to run it,
what it exposes, how to point it at a local model. Nothing here describes
or dictates `index.html`; that's yours.

## Layout

```
sevanya/       the server, ported as-is from A.I.-Sevanya (see below)
tests/         its test suite, ported the same way
index.html     yours — the server serves whatever is here
manifest.json  yours, optional — served at /manifest.json if present
static/        yours, optional — served at /static/* if present
deploy/        self-hosted ntfy compose file, for push notifications
```

`sevanya/server.py` serves `index.html` from the repo root and mounts
`/static` only if that directory exists, so a fresh checkout with just your
markup in progress still starts — `GET /` 404s cleanly instead of crashing
if `index.html` is ever mid-rewrite.

## What was ported, and what wasn't

Every backend module — `server.py`, `store.py`, `backends.py`, `tools.py`,
`agent.py`, `subagent.py`, `push.py`, `checkin.py`, `migrations.py`,
`deps.py`, `net.py`, `db.py`, `lifecycle.py`, `check.py`, `prompt.py` — came
over unchanged except for where `server.py` looks for the UI files
(`WEB_DIR`, now the repo root instead of a `sevanya/web/` subpackage). The
test suite came over the same way, minus one file: `tests/test_web.py`,
which asserted the exact markup and JS of the *old* UI (panel structure,
specific function names, specific CSS) — that's a spec of a design you're
replacing, not a backend contract, so it wasn't brought over. It's still in
`A.I.-Sevanya` if any of its ideas (the id/markup-mismatch check in
particular) are worth adapting to your own markup later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run it from the directory you want it to be able to read (tools like
`read_file` and `grep` are sandboxed to the working directory the server
started in, plus a `repos/` cache for anything pulled with `sync_repo`):

```bash
python -m sevanya
```

This checks `requirements.txt` and installs anything missing before
importing anything that needs it — a fresh clone just works. Set
`SEVANYA_SKIP_DEPS=1` to skip that check.

## Talking to a local model

```bash
export SEVANYA_BACKEND=local
export SEVANYA_LOCAL_URL=http://localhost:1234/v1   # LM Studio, Ollama, etc.
export SEVANYA_LOCAL_MODEL=...                       # optional — discovered
                                                      # from /v1/models if unset
export SEVANYA_LOCAL_KEY=not-needed                  # optional
```

`SEVANYA_BACKEND` defaults to `anthropic` (needs `ANTHROPIC_API_KEY`).
`python -m sevanya.check --runs 5` runs the tool-calling harness against
whatever backend is configured — the way to tell if a model can actually
drive Sevanya's tools before trusting it day to day.

## Environment variables

| | |
|---|---|
| `SEVANYA_TOKEN` | bearer token required on every endpoint but `/api/health` |
| `SEVANYA_PORT` | default `8765` |
| `SEVANYA_BACKEND` | `anthropic` (default) or `local` |
| `SEVANYA_LOCAL_URL` / `SEVANYA_LOCAL_MODEL` / `SEVANYA_LOCAL_KEY` | local (OpenAI-compatible) backend |
| `SEVANYA_SUBAGENT_BACKEND` | override the model the `delegate` tool hands reading work to |
| `SEVANYA_CHECKIN` | set `0` to disable the 24h-quiet check-in |
| `SEVANYA_CHECKIN_BACKEND` | override the model used for check-ins |
| `SEVANYA_NTFY_SERVER` / `SEVANYA_NTFY_TOPIC` / `SEVANYA_NTFY_TOKEN` | push notifications — self-host with `deploy/ntfy-compose.yml` |
| `SEVANYA_SKIP_DEPS` | skip the startup requirements check |

## API surface

```
POST /api/chat                  streaming (SSE) chat — the web UI
POST /api/ask                   one-shot, blocking — Siri Shortcuts
GET  /api/health                no auth required
POST /api/restart                restarts the server process (execv)
GET  /api/db                     schema/backup/migration status
POST /api/db/backup
POST /api/db/clear-history       conversations only — journal/tasks kept
GET  /api/notifications          read-only
GET  /api/tasks                  read-only — her list, not yours
GET  /api/conversations
GET  /api/conversations/{id}
GET  /
GET  /manifest.json               404 until you add one
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

332 tests, all backend/API — nothing here checks `index.html`'s markup or
JS. That's worth having once your UI settles; see the note above about
`test_web.py` in `A.I.-Sevanya` for the shape such a test file could take.

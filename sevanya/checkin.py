"""The nudge, when you've been away a day.

Sevanya writes it herself rather than filling in a template — she can see her
open list and what you were last doing, and a check-in that reads like a cron
job is one you learn to ignore.

The awkward part is that a conversation has to open with a user turn; the API
rejects one that starts with the assistant. So the check-in thread begins with
a synthetic prompt marked with MARKER, which the transcript endpoint hides. The
model still sees it — it's the instruction — but you don't get a message in
your own voice that you didn't write.
"""

import os
import threading
import time
from datetime import datetime, timedelta

from . import push

# Marks the synthetic opening turn. Kept short and unmistakable: it's matched
# against in SQL and filtered out of the transcript.
MARKER = "[sevanya:auto]"

PROMPT = (
    f"{MARKER} It has been a day since they last wrote to you. This is an "
    f"automatic nudge — they haven't asked you anything, and they aren't at "
    f"the screen. Write them a short check-in, two or three sentences. Look at "
    f"what's open on your list and at what they were last working on, and pick "
    f"the one thing most worth coming back to. If nothing is genuinely "
    f"outstanding, say so briefly rather than inventing work."
)

EXTRA = """
You are writing an unprompted check-in that will arrive as a notification. Keep
it short enough to read on a lock screen. Don't greet them at length, don't ask
how they've been, and don't apologise for interrupting — say the useful thing.
"""


def _int_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def interval_hours() -> float:
    return _int_env("SEVANYA_CHECKIN_HOURS", 24)


def enabled() -> bool:
    return os.environ.get("SEVANYA_CHECKIN", "1") != "0"


def due(store, now: datetime | None = None) -> bool:
    """Is a check-in owed?

    Two clocks, both required. Quiet for long enough since *they* last wrote —
    her own check-ins don't count, or she'd be talking to herself — and long
    enough since the last nudge, so being away for a week produces a nudge a
    day rather than one every poll.
    """
    if not enabled():
        return False

    now = now or datetime.utcnow()
    cutoff = now - timedelta(hours=interval_hours())

    last_human = store.last_human_message_at()
    if last_human is None:
        return False  # nothing has ever been said; there's nothing to nudge about
    if _parse(last_human) > cutoff:
        return False

    last_nudge = store.last_checkin_at()
    return last_nudge is None or _parse(last_nudge) <= cutoff


def _parse(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")


def backend_for_checkin():
    """Which model writes the nudge.

    SEVANYA_CHECKIN_BACKEND exists because this runs unattended, once a day,
    and a two-sentence "you left the parser half-done" is not the work that
    needs your best model. Unset, it uses whatever the conversation uses.
    """
    from . import backends

    name = os.environ.get("SEVANYA_CHECKIN_BACKEND")
    return backends.choose(name) if name else None


def run_once(store, make_agent=None) -> str | None:
    """Write one check-in. Returns the text, or None if it couldn't.

    Every failure is caught and recorded. This runs on a background thread
    with nobody watching, so an exception here would otherwise be a thread
    that quietly stopped and a nudge that never came again.
    """
    from .agent import Agent

    conversation_id = store.new_conversation(title="check-in")
    try:
        agent = (make_agent or Agent)(store, conversation_id, system_extra=EXTRA,
                                      backend=backend_for_checkin())
        text = agent.send(PROMPT)
    except Exception as exc:
        store.notify("checkin-failed", f"couldn't write the check-in ({type(exc).__name__}: {exc})")
        return None

    # Recorded before the push, and deliberately not depending on it. The
    # notice is what `due` reads to decide it has already nudged, so if it were
    # ever skipped because the phone was unreachable she'd try again in ten
    # minutes, and again, and again. send_and_log swallows its own failures, so
    # this order is belt as well as braces rather than the only thing holding
    # it up.
    store.notify("checkin", text)
    push.send_and_log(store, text, title="Sevanya")
    return text


def start(store, poll_seconds: int = 600) -> threading.Thread | None:
    """Watch the clock on a background thread. Returns None when disabled."""
    if not enabled():
        return None

    def loop():
        while True:
            time.sleep(poll_seconds)
            try:
                if due(store):
                    run_once(store)
            except Exception as exc:  # pragma: no cover - belt and braces
                try:
                    store.notify("checkin-failed", f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass

    thread = threading.Thread(target=loop, daemon=True, name="sevanya-checkin")
    thread.start()
    return thread

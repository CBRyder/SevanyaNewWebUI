"""The nudge after a day of quiet.

The behaviour that needs pinning isn't "does it send" — it's when it *doesn't*.
A check-in that counts its own message as activity never fires twice; one that
doesn't record itself before pushing fires every ten minutes forever.
"""

from datetime import datetime, timedelta

import pytest

from sevanya import checkin, push
from sevanya.store import CHECKIN_MARKER


def age(store, hours: float, table: str = "messages", where: str = "1=1") -> None:
    """Backdate rows, so a test doesn't have to wait a day."""
    stamp = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    store.db.execute(f"UPDATE {table} SET created_at = ? WHERE {where}", (stamp,))
    store.db.commit()


class FakeAgent:
    """Stands in for the model, and records the transcript the way Agent does."""

    written = "You left the parser loop half-rewritten — worth picking that up."

    def __init__(self, store, conversation_id, system_extra="", backend=None):
        self.store = store
        self.conversation_id = conversation_id
        self.system_extra = system_extra
        # Accepted because the real Agent takes it: the check-in can be routed
        # to a cheaper model than the conversation uses.
        self.backend = backend

    def send(self, text):
        self.store.append_message(self.conversation_id, "user", text)
        self.store.append_message(self.conversation_id, "assistant", self.written)
        return self.written


@pytest.fixture
def quiet(store, monkeypatch):
    """A real conversation from 30 hours ago, and pushes that go nowhere."""
    monkeypatch.delenv("SEVANYA_CHECKIN", raising=False)
    monkeypatch.delenv("SEVANYA_CHECKIN_HOURS", raising=False)
    monkeypatch.setattr(push, "send", lambda *a, **k: (True, "sent"))
    monkeypatch.setattr(push, "configured", lambda: True)

    conversation = store.new_conversation(title="real chat")
    store.append_message(conversation, "user", "why is my parser segfaulting?")
    age(store, 30)
    return store


# --- the marker ------------------------------------------------------------


def test_the_marker_matches_the_one_the_sql_uses():
    """Two copies of a magic string is two places to get it wrong."""
    assert checkin.MARKER == CHECKIN_MARKER
    assert CHECKIN_MARKER in checkin.PROMPT


# --- when it's owed --------------------------------------------------------


def test_owed_after_a_day_of_quiet(quiet):
    assert checkin.due(quiet)


def test_not_owed_while_they_are_still_talking(store, monkeypatch):
    monkeypatch.delenv("SEVANYA_CHECKIN", raising=False)
    conversation = store.new_conversation()
    store.append_message(conversation, "user", "still here")
    assert not checkin.due(store)


def test_not_owed_when_nothing_was_ever_said(store, monkeypatch):
    """A fresh install shouldn't be nudged about a conversation that never happened."""
    monkeypatch.delenv("SEVANYA_CHECKIN", raising=False)
    assert not checkin.due(store)


def test_not_owed_again_straight_after_nudging(quiet):
    checkin.run_once(quiet, make_agent=FakeAgent)
    assert not checkin.due(quiet), "it would nudge again on the next poll, and forever"


def test_owed_again_a_day_later(quiet):
    """Asked for: keep nudging until they actually come back."""
    checkin.run_once(quiet, make_agent=FakeAgent)
    age(quiet, 30, table="notifications", where="kind = 'checkin'")
    assert checkin.due(quiet)


def test_replying_stops_it(quiet):
    checkin.run_once(quiet, make_agent=FakeAgent)
    age(quiet, 30, table="notifications", where="kind = 'checkin'")
    quiet.append_message(quiet.new_conversation(), "user", "ok I'm back")
    assert not checkin.due(quiet)


def test_her_own_check_in_does_not_count_as_them_talking(quiet):
    """The synthetic prompt is stored with role 'user'.

    If it counted as activity she'd nudge once, see her own message, and
    conclude they'd come back — silence forever after.
    """
    checkin.run_once(quiet, make_agent=FakeAgent)
    age(quiet, 30, table="notifications", where="kind = 'checkin'")
    assert checkin.due(quiet), "she counted her own nudge as them replying"


def test_it_can_be_turned_off(quiet, monkeypatch):
    monkeypatch.setenv("SEVANYA_CHECKIN", "0")
    assert not checkin.due(quiet)
    assert checkin.start(quiet) is None


def test_the_interval_is_configurable(store, monkeypatch):
    monkeypatch.delenv("SEVANYA_CHECKIN", raising=False)
    monkeypatch.setenv("SEVANYA_CHECKIN_HOURS", "1")
    conversation = store.new_conversation()
    store.append_message(conversation, "user", "hello")
    age(store, 2)
    assert checkin.due(store)
    assert checkin.interval_hours() == 1


def test_a_nonsense_interval_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("SEVANYA_CHECKIN_HOURS", "soon")
    assert checkin.interval_hours() == 24


# --- writing it ------------------------------------------------------------


def test_she_writes_it_into_its_own_thread(quiet):
    text = checkin.run_once(quiet, make_agent=FakeAgent)
    assert text == FakeAgent.written
    titles = [row["title"] for row in quiet.list_conversations()]
    assert "check-in" in titles


def test_the_prompt_tells_her_it_is_automatic(quiet):
    """She should know nobody asked, so she doesn't answer a question that wasn't put."""
    assert "automatic" in checkin.PROMPT.lower()
    assert "haven't asked" in checkin.PROMPT or "haven't been" in checkin.PROMPT


def test_it_is_recorded_and_pushed(quiet, monkeypatch):
    sent = []
    monkeypatch.setattr(push, "send", lambda *a, **k: sent.append(a) or (True, "sent"))
    checkin.run_once(quiet, make_agent=FakeAgent)

    assert sent, "the whole point is that they aren't at the screen"
    assert FakeAgent.written in sent[0][0]
    assert "checkin" in [row["kind"] for row in quiet.notifications()]


def test_it_is_recorded_even_when_the_push_fails(quiet, monkeypatch):
    """Otherwise a phone that's off means a nudge every ten minutes.

    The 'checkin' notice is what tells `due` it has already fired, so it has to
    be written before the push and regardless of it.
    """
    monkeypatch.setattr(push, "send", lambda *a, **k: (False, "push failed"))
    checkin.run_once(quiet, make_agent=FakeAgent)
    assert "checkin" in [row["kind"] for row in quiet.notifications()]
    assert not checkin.due(quiet)


def test_it_is_still_recorded_when_the_push_raises(quiet, monkeypatch):
    """The real guarantee, as opposed to the order of two lines.

    If the notice were ever skipped because the phone was unreachable, she'd
    nudge again on the next poll, and every poll after that.
    """
    def explode(*args, **kwargs):
        raise RuntimeError("the notifier itself is broken")

    monkeypatch.setattr(push, "send", explode)
    checkin.run_once(quiet, make_agent=FakeAgent)
    assert "checkin" in [row["kind"] for row in quiet.notifications()]
    assert not checkin.due(quiet)


def test_a_model_failure_is_recorded_rather_than_killing_the_thread(quiet):
    """It runs on a background thread with nobody watching.

    An exception here is a thread that quietly stops, and a nudge that never
    comes again with no sign of why.
    """
    class Broken(FakeAgent):
        def send(self, text):
            raise RuntimeError("overloaded_error")

    assert checkin.run_once(quiet, make_agent=Broken) is None
    kinds = [row["kind"] for row in quiet.notifications()]
    assert "checkin-failed" in kinds
    assert "overloaded_error" in " ".join(r["message"] for r in quiet.notifications())


def test_a_failed_attempt_does_not_count_as_having_nudged(quiet):
    """It should try again, not go quiet because of one bad night."""
    class Broken(FakeAgent):
        def send(self, text):
            raise RuntimeError("nope")

    checkin.run_once(quiet, make_agent=Broken)
    assert checkin.due(quiet)


# --- routing ----------------------------------------------------------------


def test_the_check_in_can_run_on_a_cheaper_model(monkeypatch):
    """It runs unattended, once a day, and writes two sentences.

    That is not the work that needs your best model.
    """
    from sevanya import backends

    monkeypatch.setenv("SEVANYA_CHECKIN_BACKEND", "local")
    monkeypatch.setenv("SEVANYA_LOCAL_MODEL", "small-and-cheap")
    chosen = checkin.backend_for_checkin()
    assert isinstance(chosen, backends.LocalBackend)
    assert chosen.model == "small-and-cheap"


def test_without_that_variable_it_uses_whatever_she_uses(monkeypatch):
    """None means "don't override" — Agent then chooses for itself."""
    monkeypatch.delenv("SEVANYA_CHECKIN_BACKEND", raising=False)
    assert checkin.backend_for_checkin() is None


def test_the_backend_is_handed_to_the_agent(quiet, monkeypatch):
    """Choosing one and not passing it on would be a silent no-op."""
    seen = {}

    class Recording(FakeAgent):
        def __init__(self, store, conversation_id, system_extra="", backend=None):
            seen["backend"] = backend
            super().__init__(store, conversation_id, system_extra, backend)

    monkeypatch.setattr(checkin, "backend_for_checkin", lambda: "the cheap one")
    checkin.run_once(quiet, make_agent=Recording)
    assert seen["backend"] == "the cheap one"

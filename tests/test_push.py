"""Notifying the phone.

Two things this file is really about.

The topic name is the credential — anyone who knows it can read the
notifications and publish to them — so it must not turn up in a message, a log
line or an error.

And a push that fails must never break what the caller was doing. Most pushes
happen while something else is going wrong, and a failed notification becoming
the new exception is how "the reload failed" turns into "the reload crashed".
"""

import httpx
import pytest

from sevanya import push, tools


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("SEVANYA_NTFY_TOPIC", "a-long-unguessable-topic")
    monkeypatch.delenv("SEVANYA_NTFY_SERVER", raising=False)
    monkeypatch.delenv("SEVANYA_NTFY_TOKEN", raising=False)


@pytest.fixture
def unconfigured(monkeypatch):
    monkeypatch.delenv("SEVANYA_NTFY_TOPIC", raising=False)


def capture(status=200):
    """A client that records the request instead of sending it."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = request.content.decode()
        return httpx.Response(status, json={})

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


# --- configuration ---------------------------------------------------------


def test_a_bare_topic_goes_to_the_public_server(configured):
    assert push.configured()
    client, seen = capture()
    push.send("hello", client=client)
    assert seen["url"] == "https://ntfy.sh/a-long-unguessable-topic"


def test_the_server_can_be_pointed_somewhere_else(configured, monkeypatch):
    """Self-hosting should be a config change, not a code change."""
    monkeypatch.setenv("SEVANYA_NTFY_SERVER", "https://ntfy.example.com/")
    client, seen = capture()
    push.send("hello", client=client)
    assert seen["url"] == "https://ntfy.example.com/a-long-unguessable-topic"


def test_a_whole_url_can_be_given_as_the_topic(monkeypatch):
    monkeypatch.setenv("SEVANYA_NTFY_TOPIC", "https://ntfy.example.com/thing")
    client, seen = capture()
    push.send("hello", client=client)
    assert seen["url"] == "https://ntfy.example.com/thing"


def test_nothing_is_configured_by_default(unconfigured):
    assert not push.configured()
    sent, detail = push.send("hello")
    assert not sent
    assert "SEVANYA_NTFY_TOPIC" in detail, "it should say how to turn it on"


def test_the_topic_is_never_revealed(configured):
    """It's the credential. It must not leak into a log line or an error."""
    assert "a-long-unguessable-topic" not in push.where()
    assert push.where() == "https://ntfy.sh"

    client, _ = capture(status=500)
    _, detail = push.send("hello", client=client)
    assert "a-long-unguessable-topic" not in detail


def test_the_public_server_is_recognised_as_public(configured):
    """Worth surfacing: on ntfy.sh the topic name is the only thing between
    your notifications and anyone who guesses it, and the contents pass
    through a machine you don't run."""
    assert push.is_public()


def test_a_self_hosted_server_is_not_public(configured, monkeypatch):
    monkeypatch.setenv("SEVANYA_NTFY_SERVER", "http://nas:8080")
    assert not push.is_public()


def test_a_self_hosted_url_given_as_the_topic_is_not_public(monkeypatch):
    monkeypatch.setenv("SEVANYA_NTFY_TOPIC", "http://100.64.0.3:8080/games")
    assert not push.is_public()


def test_nothing_configured_is_not_reported_as_public(unconfigured):
    assert not push.is_public()


# --- sending ---------------------------------------------------------------


def test_the_request_is_what_ntfy_expects(configured):
    client, seen = capture()
    ok, _ = push.send("the parser tests pass now", title="Reload finished",
                      priority="high", tags="tada", client=client)
    assert ok
    assert seen["body"] == "the parser tests pass now"
    assert seen["headers"]["x-title"] == "Reload finished"
    assert seen["headers"]["x-priority"] == "high"
    assert seen["headers"]["x-tags"] == "tada"


def test_a_token_is_sent_when_the_topic_is_protected(configured, monkeypatch):
    monkeypatch.setenv("SEVANYA_NTFY_TOKEN", "tk_secret")
    client, seen = capture()
    push.send("hello", client=client)
    assert seen["headers"]["authorization"] == "Bearer tk_secret"


def test_no_authorization_header_without_a_token(configured):
    client, seen = capture()
    push.send("hello", client=client)
    assert "authorization" not in seen["headers"]


def test_a_server_error_is_returned_not_raised(configured):
    client, _ = capture(status=503)
    sent, detail = push.send("hello", client=client)
    assert not sent
    assert "push failed" in detail


def test_a_network_failure_is_returned_not_raised(configured):
    def handler(request):
        raise httpx.ConnectError("no route to host")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sent, detail = push.send("hello", client=client)
    assert not sent
    assert "ConnectError" in detail


# --- the log ---------------------------------------------------------------


def test_a_push_is_recorded_either_way(configured, store):
    client, _ = capture()
    push.send_and_log(store, "it worked", title="Done", client=client)

    client, _ = capture(status=500)
    push.send_and_log(store, "it didn't", title="Broken", client=client)

    kinds = [row["kind"] for row in store.notifications()]
    assert "push" in kinds
    assert "push-failed" in kinds, (
        "a push that silently didn't arrive is worse than one that never "
        "existed — you're left assuming you'd have been told"
    )


def test_nothing_is_logged_when_no_phone_is_set_up(unconfigured, store):
    """Otherwise the log fills with failures for a feature nobody turned on."""
    push.send_and_log(store, "hello")
    assert store.notifications() == []


# --- the tool --------------------------------------------------------------


@pytest.fixture
def call(store):
    conversation = store.new_conversation()

    def invoke(name, **arguments):
        return tools.dispatch(name, arguments, store=store, conversation_id=conversation)

    return invoke


def test_the_tool_sends(configured, call, monkeypatch):
    sent = []
    monkeypatch.setattr(push, "send", lambda *a, **k: sent.append((a, k)) or (True, "sent"))
    out, is_error = call("notify_phone", message="tests pass", title="Done")
    assert not is_error
    assert sent
    assert "tests pass" in out


def test_the_tool_explains_how_to_turn_it_on(unconfigured, call):
    """Otherwise she keeps trying and nobody finds out why nothing arrives."""
    out, is_error = call("notify_phone", message="hello")
    assert not is_error
    assert "SEVANYA_NTFY_TOPIC" in out


def test_an_empty_notification_is_refused(configured, call):
    out, _ = call("notify_phone", message="   ")
    assert "empty" in out


# --- the automatic ones ----------------------------------------------------


def test_completing_a_task_pushes(configured, call, monkeypatch, store):
    sent = []
    monkeypatch.setattr(push, "send", lambda *a, **k: sent.append((a, k)) or (True, "sent"))

    call("add_task", task="rewrite the parser loop")
    call("complete_task", task_id=1)
    assert sent, "ticking something off should reach the phone"
    assert "rewrite the parser loop" in sent[0][0][0]


def test_a_failed_push_does_not_stop_the_task_being_completed(configured, call, store, monkeypatch):
    """The push is the side effect. The completion is the point."""
    monkeypatch.setattr(push, "send", lambda *a, **k: (False, "push failed (ConnectError)"))

    call("add_task", task="something")
    out, is_error = call("complete_task", task_id=1)
    assert not is_error
    assert "completed task 1" in out
    assert store.get_task(1)["done"]


def test_a_push_that_raises_still_does_not_break_the_caller(configured, call, store, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("the notifier itself is broken")

    monkeypatch.setattr(push, "send", explode)
    call("add_task", task="something")
    out, is_error = call("complete_task", task_id=1)
    assert not is_error, f"a broken notifier took down complete_task: {out}"
    assert store.get_task(1)["done"]


def test_a_failed_reload_pushes(configured, call, monkeypatch):
    """You pressed reload and walked away. The alternative is assuming it worked."""
    from sevanya import lifecycle

    sent = []
    monkeypatch.setattr(push, "send", lambda *a, **k: sent.append((a, k)) or (True, "sent"))
    monkeypatch.setattr(lifecycle, "can_restart", lambda: True)
    monkeypatch.setattr(tools.deps, "ensure",
                        lambda root, install_missing=True: (False, "fastapi: not installed"))

    call("reload")
    assert sent, "a reload that couldn't fix the requirements should reach the phone"
    assert "fastapi" in sent[0][0][0]
    assert sent[0][1].get("priority") == "high"


def test_a_successful_reload_does_not_push(configured, call, monkeypatch):
    """The page comes back by itself; you're already looking at it."""
    from sevanya import lifecycle

    sent = []
    monkeypatch.setattr(push, "send", lambda *a, **k: sent.append(a) or (True, "sent"))
    monkeypatch.setattr(lifecycle, "can_restart", lambda: True)
    monkeypatch.setattr(lifecycle, "schedule_restart", lambda *a, **k: None)
    monkeypatch.setattr(tools.deps, "ensure", lambda root, install_missing=True: (True, "all good"))

    call("reload")
    assert not sent

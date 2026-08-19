"""Sending a notification to the phone, via ntfy.

Configure it with two environment variables:

    SEVANYA_NTFY_TOPIC=some-long-unguessable-string
    SEVANYA_NTFY_SERVER=https://ntfy.sh      # optional, or your own
    SEVANYA_NTFY_TOKEN=tk_...                # optional, for a protected topic

Then subscribe to the same topic in the ntfy app. A topic on the public server
is readable by anyone who knows its name, so make it long and don't send
anything through it you'd mind a stranger reading — or point SEVANYA_NTFY_SERVER
at your own instance, which is a config change and no code.

Unlike fetch_url, the address here is *yours*, not the model's, so it isn't put
through net.check_url — a self-hosted ntfy on your own network is exactly the
setup that check would refuse, and correctly so for a URL the model chose.

Nothing in here raises. A push failing is worth recording and worth telling
her about, but it is never worth losing the thing she was doing in order to
report it.
"""

import os

import httpx

TIMEOUT = 10


def _config() -> tuple[str, str | None] | None:
    """(url, token), or None when it isn't set up."""
    topic = (os.environ.get("SEVANYA_NTFY_TOPIC") or "").strip()
    if not topic:
        return None

    if topic.startswith(("http://", "https://")):
        url = topic  # a whole URL was given; use it as-is
    else:
        server = (os.environ.get("SEVANYA_NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
        url = f"{server}/{topic}"

    return url, (os.environ.get("SEVANYA_NTFY_TOKEN") or "").strip() or None


PUBLIC = "https://ntfy.sh"


def configured() -> bool:
    return _config() is not None


def is_public() -> bool:
    """Whether notifications are going through somebody else's server.

    Worth knowing rather than assuming: on the public instance the topic name
    is the only thing between your notifications and anyone who guesses it, and
    the contents of what you're building go through a machine you don't run.
    """
    config = _config()
    return config is not None and config[0].startswith(PUBLIC)


def where() -> str:
    """A description safe to show — the server, never the topic.

    The topic name is the credential: anyone holding it can read the
    notifications and publish to them.
    """
    config = _config()
    if config is None:
        return "not configured"
    url, _ = config
    return url.rsplit("/", 1)[0] or url


def _scrub(text: str, url: str) -> str:
    """Remove the topic from anything we're about to show or store."""
    topic = url.rsplit("/", 1)[-1]
    text = text.replace(url, where())
    return text.replace(topic, "…") if topic else text


def send(message: str, title: str = "Sevanya", priority: str | None = None,
         tags: str | None = None, client: httpx.Client | None = None) -> tuple[bool, str]:
    """Push one notification. Returns (sent, what happened)."""
    config = _config()
    if config is None:
        return False, "no phone configured (set SEVANYA_NTFY_TOPIC)"

    url, token = config
    headers = {"X-Title": title}
    if priority:
        headers["X-Priority"] = priority
    if tags:
        headers["X-Tags"] = tags
    if token:
        headers["Authorization"] = f"Bearer {token}"

    owned = client is None
    client = client or httpx.Client(timeout=TIMEOUT)
    try:
        response = client.post(url, content=message.encode("utf-8"), headers=headers)
        response.raise_for_status()
        return True, f"sent to {where()}"
    except Exception as exc:
        # Caught, not raised. The caller is usually already handling something
        # that went wrong, and a failed push must not become the new problem.
        #
        # Scrubbed, because httpx puts the failing URL in its message and the
        # topic is the credential — an error line ends up in the notices log,
        # which is exactly where it must not appear.
        return False, f"push failed ({type(exc).__name__}: {_scrub(str(exc), url)})"
    finally:
        if owned:
            client.close()


def send_and_log(store, message: str, title: str = "Sevanya", **kwargs) -> tuple[bool, str]:
    """Push, and record the attempt where it can be read later.

    Recorded either way. A push that silently didn't arrive is worse than one
    that never existed, because you're left assuming you'd have been told.
    """
    try:
        ok, detail = send(message, title=title, **kwargs)
        if configured():
            store.notify("push" if ok else "push-failed", f"{title}: {message} — {detail}")
        return ok, detail
    except Exception as exc:
        # send() already catches its own failures, so reaching here means the
        # notifier itself is broken. That still must not propagate: every
        # caller is doing something that matters more than the push, and a
        # notification failure turning into "complete_task crashed" is a much
        # worse bug than a missed buzz.
        return False, f"push failed ({type(exc).__name__})"

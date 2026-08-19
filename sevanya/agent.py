"""The agent loop.

This is the whole idea. Everything else in this project — the database, the
web server, the phone access — is plumbing around these forty lines.

The model cannot run anything. When it wants a tool, it emits a structured
block saying so and stops. Your code runs the function, appends the result to
the conversation, and calls the model again. Repeat until it stops asking.
That's what makes an "agent" rather than a chatbot.
"""

from . import backends, tools
from .prompt import SYSTEM
from .store import Store

# A runaway loop is a runaway bill. If it's made this many tool calls without
# arriving at an answer, something is wrong and you want to know about it.
MAX_STEPS = 12


class Agent:
    def __init__(
        self,
        store: Store,
        conversation_id: int,
        client=None,
        system_extra: str = "",
        backend=None,
    ):
        # `backend` decides which model answers — Anthropic, or something on
        # your own machine. `client` is still accepted because tests pass one;
        # it's wrapped rather than used directly, so there's one path through
        # here regardless.
        if backend is not None:
            self.backend = backend
        elif client is not None:
            self.backend = backends.AnthropicBackend(client=client)
        else:
            self.backend = backends.choose()
        self.store = store
        self.conversation_id = conversation_id
        # Extra instructions for this agent only — the voice endpoint uses it
        # to ask for short spoken answers without forking the whole prompt.
        self.system_extra = system_extra

        # The conversation. The API is stateless — the model remembers nothing
        # between calls — so this list *is* Sevanya's memory of the session,
        # and gets re-sent in full on every single turn.
        #
        # Loading it from SQLite is the entire trick behind resuming a session,
        # and later behind picking up on your phone where you left off at your
        # desk. There is no server-side session to reconnect to; you just
        # replay the transcript.
        self.messages: list[dict] = store.load_messages(conversation_id)

    def _system(self) -> str:
        """System prompt, plus a little continuity from the journal and the list.

        Injecting the few most recent notes costs almost nothing and means
        Sevanya opens knowing roughly where you left off, rather than needing
        to call `recall` before it can be useful. Anything older it has to go
        looking for — which is the right default, since old notes shouldn't
        crowd the context window forever.

        Open tasks go in for a second reason as well as continuity: completing
        one requires knowing its id, and a list she can already see is a list
        she can act on without a lookup first.
        """
        prompt = SYSTEM + self.system_extra

        recent = self.store.recent_journal(limit=5)
        if recent:
            lines = "\n".join(f"- [{r['topic']}] {r['note']}" for r in reversed(recent))
            prompt = f"{prompt}\n\nFrom your journal, most recent last:\n{lines}\n"

        tasks = self.store.open_tasks(limit=10)
        if tasks:
            lines = "\n".join(f"- {t['id']}. {t['task']}" for t in tasks)
            prompt = (
                f"{prompt}\n\nOn their task list, still open "
                f"(the number is the task_id):\n{lines}\n"
            )

        return prompt

    def send(self, user_text: str) -> str:
        """Send one user message, run any tools it triggers, return the reply."""
        self._record("user", user_text)

        for _ in range(MAX_STEPS):
            response = self.backend.send(self._system(), self.messages, tools.TOOLS)

            # Append the *whole* content list, not just the text. It contains
            # the tool_use blocks, and the next request will be rejected if
            # they're missing — every tool_result has to answer a tool_use the
            # model can still see. Same reason store.py keeps blocks as JSON.
            self._record("assistant", response.content, self.backend.describe())

            if response.stop_reason != "tool_use":
                return response.text()

            # It asked for one or more tools. Run them all, then send every
            # result back in a single user message — splitting them across
            # several messages trains the model out of asking in parallel.
            results = []
            for block in response.content:
                if block.get("type") != "tool_use":
                    continue
                output, is_error = tools.dispatch(
                    block["name"],
                    block.get("input", {}),
                    store=self.store,
                    conversation_id=self.conversation_id,
                )
                print(f"  [{block['name']} {block.get('input', {})}]")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],  # must match the request
                        "content": output,
                        "is_error": is_error,
                    }
                )

            self._record("user", results)

        return "(gave up after too many tool calls — something's looping)"

    def stream(self, user_text: str):
        """Same loop as send(), but yields as it goes instead of at the end.

        Yields (kind, payload) pairs:
            ("text", chunk)      a piece of the reply, as the model produces it
            ("tool", name)       a tool is about to run
            ("thinking", note)   a reasoning model is working before it answers —
                                  local backend only; Anthropic's thinking is
                                  never surfaced, by the same reasoning as below

        The web UI uses this so you can watch it work. Siri uses send()
        instead, because a Shortcut can't consume a stream — it makes one
        request and waits for one answer.
        """
        self._record("user", user_text)

        for _ in range(MAX_STEPS):
            # `yield from` on a generator hands back whatever it returns, which
            # is how the finished Reply comes out the other side of the text.
            response = yield from _tagged(
                self.backend.stream(self._system(), self.messages, tools.TOOLS)
            )

            self._record("assistant", response.content, self.backend.describe())

            if response.stop_reason != "tool_use":
                return

            results = []
            for block in response.content:
                if block.get("type") != "tool_use":
                    continue
                yield ("tool", block["name"])
                output, is_error = tools.dispatch(
                    block["name"],
                    block.get("input", {}),
                    store=self.store,
                    conversation_id=self.conversation_id,
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": output,
                        "is_error": is_error,
                    }
                )

            self._record("user", results)

        yield ("text", "\n(gave up after too many tool calls — something's looping)")

    def _record(self, role: str, content, model: str | None = None) -> None:
        """Append to the in-memory transcript and to disk, together.

        Kept as one call so the two can't drift. If you write to the list in
        one place and the database in another, you eventually find a path that
        does one and not the other, and reloading gives you a broken
        conversation the API refuses.
        """
        self.messages.append({"role": role, "content": content})
        self.store.append_message(self.conversation_id, role, content, model=model)


def _tagged(stream):
    """Relabel a backend's bare text chunks as ("text", chunk) pairs.

    The backend yields strings and *returns* the finished Reply; callers here
    want tagged tuples and the Reply. StopIteration.value is where a
    generator's return value arrives, and passing it back out is what lets
    Agent.stream say `response = yield from _tagged(...)`.

    A backend can also yield an already-tagged (kind, payload) tuple directly
    — LocalBackend does this for ("thinking", ...), since a reasoning model's
    thinking tokens aren't the answer and showing them as "text" would put
    her raw internal monologue in the transcript as if she'd said it.
    """
    while True:
        try:
            item = next(stream)
        except StopIteration as done:
            return done.value
        yield item if isinstance(item, tuple) else ("text", item)

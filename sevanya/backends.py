"""Which model answers, and how to talk to it.

Sevanya was written against Anthropic's API. Running her on a model on your own
machine means a different wire format — LM Studio and Ollama both speak the
OpenAI chat-completions shape — so this module is the translation layer, and
the only place in the project that knows either format exists.

**The stored transcript stays Anthropic-shaped.** Content blocks on disk keep
the shape they have always had, and every backend converts at the boundary.
That's deliberate: changing the stored shape would break every conversation
already in the database, which is the failure mode this project has been
careful about all along. It also means you can answer a thread with a local
model that Claude started, and vice versa.

Configure it with:

    SEVANYA_BACKEND=local            # or 'anthropic', the default
    SEVANYA_LOCAL_URL=http://localhost:1234/v1      # LM Studio's default
    SEVANYA_LOCAL_MODEL=some-model-you-loaded
    SEVANYA_LOCAL_KEY=...            # only if your server wants one

Ollama works too: point SEVANYA_LOCAL_URL at http://localhost:11434/v1.
"""

import json
import os

import httpx

# Anthropic's, when that's the backend. Local models are named by whatever you
# loaded, so there's no sensible default for those.
CLAUDE_MODEL = "claude-opus-5"

MAX_TOKENS = 16000
LOCAL_TIMEOUT = 600     # a local model on a busy GPU is not a fast API


class Reply:
    """One model turn, in the shape the store already holds.

    `content` is a list of Anthropic-style blocks — {"type": "text", ...} and
    {"type": "tool_use", ...}. `stop_reason` is "tool_use" when it wants a tool
    run, anything else when it's finished.
    """

    def __init__(self, content: list[dict], stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason

    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text")


# --- talking to Anthropic ---------------------------------------------------


def _as_dicts(blocks) -> list[dict]:
    """SDK objects to plain dicts, so everything downstream sees one shape."""
    out = []
    for block in blocks:
        out.append(block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block)
    return out


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, client=None, model: str = CLAUDE_MODEL):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model = model

    def describe(self) -> str:
        return f"anthropic · {self.model}"

    def send(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        return Reply(_as_dicts(response.content), response.stop_reason)

    def stream(self, system: str, messages: list[dict], tools: list[dict]):
        """Yields text as it arrives; returns the finished Reply.

        The return value comes back through StopIteration, which `yield from`
        hands you directly — see Agent.stream.
        """
        with self.client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools,
            thinking={"type": "adaptive"},
            messages=messages,
        ) as stream:
            # text_stream yields only visible text — thinking and tool_use
            # blocks are accumulated but not emitted here.
            for chunk in stream.text_stream:
                yield chunk
            final = stream.get_final_message()

        return Reply(_as_dicts(final.content), final.stop_reason)


# --- talking to something on your own machine -------------------------------


def tools_for_openai(tools: list[dict]) -> list[dict]:
    """Anthropic tool definitions to OpenAI's function shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


def messages_for_openai(system: str, messages: list[dict]) -> list[dict]:
    """Translate a stored transcript into OpenAI chat messages.

    The interesting cases are all about history written by a *different*
    backend. A conversation Claude started can contain thinking blocks, which
    mean nothing to a local model and are dropped rather than sent as
    something it will misread. Tool calls and their results have to be paired
    across two different conventions: Anthropic puts results in a user turn,
    OpenAI gives them a role of their own.
    """
    out: list[dict] = [{"role": "system", "content": system}]

    for message in messages:
        role = message["role"]
        content = message["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts, calls = [], []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    text_parts.append(block.get("text", ""))
                elif kind == "tool_use":
                    calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            # OpenAI wants the arguments as a JSON *string*.
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                # Anything else — thinking, redacted_thinking, whatever a
                # future model adds — is deliberately dropped. Passing a block
                # the other side doesn't understand is worse than saying less.
            turn = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if calls:
                turn["tool_calls"] = calls
            out.append(turn)
            continue

        # A user turn that's a list is Sevanya's tool results. OpenAI wants one
        # message per result, with its own role.
        results = [b for b in content if b.get("type") == "tool_result"]
        if results:
            for block in results:
                out.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": str(block.get("content", "")),
                })
            continue

        text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
        out.append({"role": role, "content": text})

    return out


def reply_from_openai(message: dict, finish_reason: str | None) -> Reply:
    """An OpenAI response message back into stored-transcript shape."""
    blocks: list[dict] = []

    text = message.get("content")
    if text:
        blocks.append({"type": "text", "text": text})

    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            # Small models produce malformed JSON often enough that this must
            # not be an exception. Hand it to the tool layer as an argument it
            # will reject, so the model sees an error it can correct rather
            # than the process dying.
            arguments = {"__malformed__": raw}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or f"call_{len(blocks)}",
            "name": function.get("name", ""),
            "input": arguments,
        })

    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else (finish_reason or "end_turn")
    return Reply(blocks, stop)


class LocalBackend:
    """Anything speaking OpenAI chat-completions: LM Studio, Ollama, vLLM."""

    name = "local"

    def __init__(self, url: str | None = None, model: str | None = None,
                 key: str | None = None, client: httpx.Client | None = None):
        self.url = (url or os.environ.get("SEVANYA_LOCAL_URL")
                    or "http://localhost:1234/v1").rstrip("/")
        # None means "ask the server what it has". Naming the model by hand is
        # a step you can't take until the thing is loaded, and getting it wrong
        # is a 404 from Ollama and a silent surprise from anything that quietly
        # serves whatever it has anyway.
        self._model = model or os.environ.get("SEVANYA_LOCAL_MODEL") or None
        self.key = key if key is not None else os.environ.get("SEVANYA_LOCAL_KEY", "not-needed")
        self._client = client

    @property
    def model(self) -> str:
        """The model to ask for — discovered once, if you didn't name one.

        Only cached on success. A failed lookup usually means the server isn't
        up yet, and remembering "unknown" from before you started it would be
        the wrong thing to hold on to.
        """
        if self._model is None:
            self._model = self.loaded_model()
        return self._model or "local-model"

    def loaded_model(self) -> str | None:
        """Whatever the server says it has, or None if it can't be asked."""
        try:
            client = self._client or httpx.Client(timeout=5)
            try:
                response = client.get(f"{self.url}/models")
                response.raise_for_status()
                listed = [m.get("id") for m in response.json().get("data", []) if m.get("id")]
            finally:
                if self._client is None:
                    client.close()
        except Exception:
            return None
        return listed[0] if listed else None

    def describe(self) -> str:
        return f"local · {self.model} · {self.url}"

    def _post(self, body: dict, stream: bool):
        client = self._client or httpx.Client(timeout=LOCAL_TIMEOUT)
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        return client, client.build_request(
            "POST", f"{self.url}/chat/completions", json=body, headers=headers
        )

    def _body(self, system, messages, tools, stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": messages_for_openai(system, messages),
            "max_tokens": MAX_TOKENS,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools_for_openai(tools)
        return body

    def send(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        client, request = self._post(self._body(system, messages, tools, False), False)
        try:
            response = client.send(request)
            response.raise_for_status()
            choice = response.json()["choices"][0]
        finally:
            if self._client is None:
                client.close()
        return reply_from_openai(choice["message"], choice.get("finish_reason"))

    def stream(self, system: str, messages: list[dict], tools: list[dict]):
        """Same contract as the Anthropic one: yield text, return the Reply."""
        client, request = self._post(self._body(system, messages, tools, True), True)
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        finish = None
        # Set once, the moment reasoning starts, so the client gets a single
        # "she's working on it" signal instead of one per thinking token —
        # a reasoning model's thinking can run to hundreds of tokens, and
        # nothing downstream wants that many events for something that isn't
        # the answer.
        announced_thinking = False

        try:
            response = client.send(request, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue  # keep-alive or a partial frame; not worth dying over

                choice = (chunk.get("choices") or [{}])[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}

                # Qwen3 (and other reasoning models) put thinking tokens in
                # their own field, separate from the answer. Not shown as
                # "text" — it's not what she's saying, it's what she's
                # working out before she says anything, and for a model like
                # this one it's most of the tokens: surfacing it as text
                # would put a wall of internal monologue in the transcript.
                if delta.get("reasoning_content") and not announced_thinking:
                    announced_thinking = True
                    yield ("thinking", "thinking…")

                piece = delta.get("content")
                if piece:
                    text_parts.append(piece)
                    yield piece

                # Tool calls arrive in fragments, indexed, with the arguments
                # split across chunks. They have to be reassembled by index —
                # the id and name usually come once, the arguments in pieces.
                for fragment in delta.get("tool_calls") or []:
                    slot = calls.setdefault(
                        fragment.get("index", 0),
                        {"id": None, "function": {"name": "", "arguments": ""}},
                    )
                    if fragment.get("id"):
                        slot["id"] = fragment["id"]
                    function = fragment.get("function") or {}
                    if function.get("name"):
                        slot["function"]["name"] = function["name"]
                    if function.get("arguments"):
                        slot["function"]["arguments"] += function["arguments"]
        finally:
            if self._client is None:
                client.close()

        message = {"content": "".join(text_parts) or None}
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return reply_from_openai(message, finish)


# --- choosing ---------------------------------------------------------------


def choose(name: str | None = None):
    """The backend named by SEVANYA_BACKEND, or Anthropic."""
    name = (name or os.environ.get("SEVANYA_BACKEND") or "anthropic").strip().lower()
    if name in ("local", "lmstudio", "lm-studio", "ollama", "openai"):
        return LocalBackend()
    if name == "anthropic":
        return AnthropicBackend()
    raise ValueError(f"unknown SEVANYA_BACKEND {name!r} — use 'anthropic' or 'local'")

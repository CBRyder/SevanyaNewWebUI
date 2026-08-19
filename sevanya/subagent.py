"""Sending a piece of work somewhere else, so it doesn't cost the conversation.

The reason this exists is not specialisation. It's context.

Her fixed overhead is about 3,500 tokens before you type — the teaching prompt
and thirteen tool definitions — and a single read_file can add ten thousand
more. On a local model with a 32k window, three file reads and the conversation
is full. Everything after that is her forgetting the beginning of the thing you
were working on.

A sub-agent reads those files in *its own* context and hands back a paragraph.
The main conversation pays for the paragraph. That's the whole idea, and it's
worth more on a small local model than on a large hosted one.

Two constraints hold it in place: it gets the reading tools only — no journal,
no task list, no notifying anyone, and crucially no delegating, so it can't
spawn its own children — and whatever it returns is capped, because a
sub-agent that hands back everything it read has achieved nothing.
"""

import os

from . import backends, tools

# What it may use. Reading, and nothing that writes or reaches you: this is a
# researcher, not a second Sevanya. Leaving `delegate` out is what stops a
# model with a taste for delegation from spawning agents until the step limit
# saves it.
TOOL_NAMES = ("read_file", "list_files", "grep", "sync_repo", "fetch_url")

# Its own limit. Higher than a conversational turn because searching is what it
# was sent to do, but bounded — a sub-agent that loops is a sub-agent burning
# your GPU with nobody watching the output.
MAX_STEPS = 16

# What comes back. A sub-agent that returns everything it read has moved the
# problem rather than solved it.
MAX_REPLY = 4000

SYSTEM = """\
You are answering one question for another assistant, using the tools you have.
You are not talking to a person and there is no conversation to maintain.

Work it out, then answer in a few sentences. Be specific: name files, quote the
line that matters, give the number. The assistant asking you cannot see what
you read — only what you write — so a vague answer is a wasted trip, and so is
pasting back a wall of code it now has to read anyway.

If the answer isn't in what you can reach, say so plainly and say where you
looked. A confident guess is worse than nothing here, because it will be
repeated to someone as fact.
"""


def sub_tools() -> list[dict]:
    return [t for t in tools.TOOLS if t["name"] in TOOL_NAMES]


def backend_for_subagents():
    """Which model does the legwork.

    SEVANYA_SUBAGENT_BACKEND lets the searching run somewhere cheaper than the
    conversation — a small local model is perfectly good at "find where this is
    defined", which is most of what gets delegated.
    """
    name = os.environ.get("SEVANYA_SUBAGENT_BACKEND")
    return backends.choose(name) if name else backends.choose()


def run(task: str, store, parent_conversation_id: int | None = None, backend=None) -> str:
    """Do the task in a fresh context. Return what it found.

    The transcript is kept — as its own conversation, marked so it doesn't
    clutter the list of yours — because when a delegated answer turns out to be
    wrong, the only way to find out why is to read what it actually did.
    """
    backend = backend or backend_for_subagents()
    conversation_id = store.new_conversation(title=f"sub: {task[:50]}", kind="subagent")

    messages: list[dict] = [{"role": "user", "content": task}]
    available = sub_tools()

    for _ in range(MAX_STEPS):
        reply = backend.send(SYSTEM, messages, available)
        messages.append({"role": "assistant", "content": reply.content})
        store.append_message(conversation_id, "assistant", reply.content,
                             model=backend.describe())

        if reply.stop_reason != "tool_use":
            answer = reply.text().strip()
            break

        results = []
        for block in reply.content:
            if block.get("type") != "tool_use":
                continue
            output, is_error = tools.dispatch(
                block["name"], block.get("input", {}),
                store=store, conversation_id=conversation_id,
            )
            results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": output,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})
        store.append_message(conversation_id, "user", results)
    else:
        answer = "(the sub-agent ran out of steps without reaching an answer)"

    if len(answer) > MAX_REPLY:
        # Truncating defeats the point less than passing it on would: the
        # caller asked for a finding, not a transcript.
        answer = answer[:MAX_REPLY] + f"\n\n[cut at {MAX_REPLY} characters — ask something narrower]"

    return answer or "(the sub-agent came back with nothing)"

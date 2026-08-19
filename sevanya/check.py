"""Is the local model set up, and can it actually use her tools?

    python -m sevanya.check

Everything she does well runs through tool calls — reading your code, grepping
it, recall, her list. A model that chats beautifully and calls tools badly will
feel like *she* got worse rather than like the model did, so this asks the
question directly instead of leaving you to infer it from a bad afternoon.
"""

import json
import sys

import httpx

from . import backends, tools
from .prompt import SYSTEM

# Every tool she has, not a convenient subset. Choosing the right one out of
# thirteen is a different job from calling the only one on offer, and it is the
# job small models actually fail — so testing with a single tool would flatter a
# model that disappoints you the moment it is real.
PROBE = tools.TOOLS

# Questions that can only be answered by calling a particular tool. Answering in
# words is one kind of failure; calling the wrong tool is another, and the
# difference is worth seeing.
ASKS = [
    ("What does README.md say? Read it.", "read_file"),
    ("What files are in this project? List them.", "list_files"),
    ("Where is MAX_STEPS defined? Search the code for it.", "grep"),
    ("What did we work on last time? Check your notes.", "recall"),
]

ASK = "What does README.md say? Read it."

# Roughly, and it doesn't need to be better than roughly: the point is whether
# the context window is in the right order of magnitude.
def _tokens(text: str) -> int:
    return round(len(text) / 4)


def report(backend: backends.LocalBackend, runs: int = 3) -> int:
    print(f"backend:  {backend.describe()}")
    print(f"url:      {backend.url}")

    # --- is anything there? -------------------------------------------------
    try:
        response = httpx.get(f"{backend.url}/models", timeout=10)
        response.raise_for_status()
        listed = [m.get("id") for m in response.json().get("data", [])]
    except Exception as exc:
        print(f"\ncannot reach it ({type(exc).__name__}: {exc})")
        print("\n  - is the server started?  LM Studio: Developer tab -> Start Server")
        print("  - is the port right?      SEVANYA_LOCAL_URL is what she'll use")
        return 2

    print(f"loaded:   {', '.join(listed) if listed else '(nothing loaded)'}")
    if backend.model not in listed and listed:
        print(f"\n  note: SEVANYA_LOCAL_MODEL is {backend.model!r}, which isn't in that list.")
        print(f"        Most servers serve whatever is loaded anyway, but setting it to")
        print(f"        {listed[0]!r} removes the doubt.")

    # --- how much room does she need? ---------------------------------------
    overhead = _tokens(SYSTEM) + _tokens(json.dumps(backends.tools_for_openai(tools.TOOLS)))
    print(f"\nher fixed overhead: ~{overhead} tokens before you type anything")
    print(f"  ({len(tools.TOOLS)} tool definitions plus the teaching prompt)")
    print("  one read_file can add ~10,000 more, so set the model's context to")
    print("  16k at the very least — 32k if the card will take it.")

    # --- the questions that matter -------------------------------------------
    #
    # Repeated, because with a small model the question is not "can it" but
    # "how often". One lucky call tells you nothing about the twentieth.
    print(f"\nasking it to use tools ({len(tools.TOOLS)} available, {runs} rounds):")

    called = right = malformed = 0
    attempts = 0

    for round_number in range(runs):
        for ask, expected in ASKS:
            attempts += 1
            try:
                reply = backend.send(SYSTEM, [{"role": "user", "content": ask}], PROBE)
            except Exception as exc:
                print(f"  the request failed ({type(exc).__name__}: {exc})")
                return 2

            calls = [b for b in reply.content if b.get("type") == "tool_use"]
            if not calls:
                if round_number == 0:
                    said = reply.text().strip().replace("\n", " ")
                    print(f"  {expected:<10} answered in words: {said[:60]!r}")
                continue

            called += 1
            chosen = calls[0]
            bad_args = "__malformed__" in chosen.get("input", {})
            if bad_args:
                malformed += 1
            if chosen["name"] == expected and not bad_args:
                right += 1
            elif round_number == 0:
                detail = "unparseable arguments" if bad_args else f"called {chosen['name']}"
                print(f"  {expected:<10} {detail}")

    print(f"\n  called some tool:     {called}/{attempts}")
    print(f"  called the right one: {right}/{attempts}")
    if malformed:
        print(f"  arguments that were not valid JSON: {malformed}")

    if attempts and right == attempts:
        print("\n  good. This model picks the right tool out of thirteen, every time.")
        return 0

    share = right / attempts if attempts else 0
    if share >= 0.75:
        print("\n  workable. It mostly picks the right tool; expect the occasional")
        print("  wrong turn, which she recovers from by seeing the tool's error.")
        return 0
    if called == 0:
        print("\n  it never called a tool. She will be badly worse on this model —")
        print("  she cannot read your code, grep it, or recall anything. Look for a")
        print("  model whose card mentions function calling or tool use, and check")
        print("  tool use is enabled in the server settings.")
        return 1

    print("\n  unreliable. It calls tools but often the wrong one, which costs a")
    print("  turn each time and sometimes ends in her giving up after MAX_STEPS.")
    print("  Whether a model was trained for tool use matters far more here than")
    print("  how many parameters it has.")
    return 1


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m sevanya.check", description=__doc__)
    parser.add_argument("--runs", type=int, default=3,
                        help="how many times to ask each question (default 3)")
    args = parser.parse_args(argv)

    return report(backends.choose("local"), runs=max(1, args.runs))


if __name__ == "__main__":
    raise SystemExit(main())

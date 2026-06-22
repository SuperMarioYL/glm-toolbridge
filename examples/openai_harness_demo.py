#!/usr/bin/env python3
"""Before / after: a stock OpenAI-format tool loop against GLM-5.2.

This runs fully offline. ``_FakeGLMCompletions`` stands in for a real GLM-5.2
endpoint and emits GLM's *actual* tool-call wire shape (native-object arguments,
an interleaved reasoning trace). That is exactly the shape a coding-agent harness
that hardcodes the OpenAI ``tool_calls`` schema cannot read.

LEFT  (raw):     the stock harness reads the GLM response directly and the tool
                 loop stalls — ``arguments`` is not the JSON *string* it expects,
                 and ``content`` is non-null when it should be null.
RIGHT (bridged): the *same* harness wraps the client with
                 ``glm_toolbridge.wrap`` (one import + one line) and the identical
                 tool call parses, runs, and the loop completes.

Run:  python examples/openai_harness_demo.py
"""

from __future__ import annotations

import json
from typing import Any

from glm_toolbridge import wrap


# --------------------------------------------------------------------------- #
# A stand-in GLM-5.2 endpoint that speaks GLM's real wire shape.               #
# --------------------------------------------------------------------------- #

class _FakeGLMCompletions:
    """Returns a GLM-shaped tool-call response (object args + reasoning trace)."""

    def create(self, **kwargs: Any) -> dict[str, Any]:
        # GLM-5.2 emits arguments as a native object, content non-null, plus a
        # reasoning_content trace alongside the call.
        return {
            "id": "chatcmpl-glm-demo",
            "object": "chat.completion",
            "model": kwargs.get("model", "glm-5.2"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check the weather for you.",
                        "reasoning_content": "User asked about Beijing weather; call get_weather.",
                        "tool_calls": [
                            {
                                "id": "call_demo_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    # <-- native object, NOT a JSON string
                                    "arguments": {"city": "Beijing", "unit": "celsius"},
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }


class _FakeGLMChat:
    def __init__(self) -> None:
        self.completions = _FakeGLMCompletions()


class FakeGLMClient:
    """Minimal stand-in with the OpenAI-SDK client surface used by harnesses."""

    def __init__(self) -> None:
        self.chat = _FakeGLMChat()


# --------------------------------------------------------------------------- #
# The "harness" — a stock OpenAI-format tool loop. It is NOT GLM-aware.        #
# --------------------------------------------------------------------------- #

def run_tool(name: str, arguments_json: str) -> str:
    """A trivial local tool the harness dispatches to."""
    args = json.loads(arguments_json)  # OpenAI guarantees arguments is a JSON string
    if name == "get_weather":
        return f"{args['city']}: 21 {args['unit']}, clear"
    return "unknown tool"


def harness_step(client: Any) -> str:
    """One step of a stock OpenAI tool loop. Raises if the response is off-shape.

    A real harness does exactly this: pull tool_calls, json.loads the arguments
    string, dispatch. If arguments is not a string (GLM's native object) the
    json.loads raises — the silent breakage devs hit today.
    """
    resp = client.chat.completions.create(
        model="glm-5.2",
        messages=[{"role": "user", "content": "weather in Beijing?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    # OpenAI-format access path — unchanged whether or not GLM is behind it.
    message = _message_of(resp)
    calls = _tool_calls_of(message)
    if not calls:
        raise RuntimeError("harness expected a tool call but found none")
    call = calls[0]
    name = _call_name(call)
    arguments = _call_arguments(call)  # MUST be a JSON string for json.loads
    return run_tool(name, arguments)


# Small accessors that work on both plain dicts (raw GLM) and pydantic models
# (the bridged path returns the same dict shape here since we pass a dict client).

def _message_of(resp: Any) -> dict[str, Any]:
    return resp["choices"][0]["message"]


def _tool_calls_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    return message.get("tool_calls") or []


def _call_name(call: dict[str, Any]) -> str:
    return call["function"]["name"]


def _call_arguments(call: dict[str, Any]) -> str:
    return call["function"]["arguments"]


# --------------------------------------------------------------------------- #
# main: run the same harness twice — raw (fails) then bridged (works).         #
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 64)
    print("  glm-toolbridge demo — same OpenAI harness, GLM-5.2 backend")
    print("=" * 64)

    raw = FakeGLMClient()
    print("\n[LEFT] stock harness against raw GLM-5.2 ...")
    try:
        result = harness_step(raw)
        print(f"  unexpected success: {result}")
    except Exception as exc:  # noqa: BLE001 - demoing the real failure
        print(f"  ✗ tool loop broke: {type(exc).__name__}: {exc}")
        print("    (GLM sent arguments as a native object; json.loads chokes — "
              "the silent breakage devs hit today.)")

    bridged = wrap(FakeGLMClient())
    print("\n[RIGHT] same harness, one-line wrap: client = wrap(client) ...")
    result = harness_step(bridged)
    print(f"  ✓ tool loop completed: {result}")
    print("    (arguments normalized to a JSON string, content forced null, "
          "reasoning relocated — the harness never knew GLM was behind it.)")

    print("\n" + "-" * 64)
    print("  Install:  uv add glm-toolbridge   (or: pip install glm-toolbridge)")
    print("  Repo:     https://github.com/SuperMarioYL/glm-toolbridge")
    print("-" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

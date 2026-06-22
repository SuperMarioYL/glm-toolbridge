"""Pure transforms: GLM-5.2 wire shape <-> canonical OpenAI tool_calls shape.

Everything here is a side-effect-free function over plain dicts. The typed
models live in :mod:`glm_toolbridge.adapter`; this module is the mechanical
layer the adapter composes, and the one the roundtrip tests hammer directly.

Two directions:

``normalize(glm_response)``
    GLM -> OpenAI. Fix every divergence enumerated in
    :mod:`glm_toolbridge.protocol` so a harness reading ``message.tool_calls``
    sees exactly what it expects.

``denormalize_tools(openai_tools)``
    OpenAI -> GLM. Lower OpenAI-shaped tool *definitions* into the request body
    GLM-5.2 accepts. (GLM accepts the OpenAI ``tools`` schema closely, but we
    normalize a few quirks so the round trip is total.)
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from .errors import (
    MalformedToolArguments,
    StreamAssemblyError,
    UnsupportedProtocolShape,
)

# --------------------------------------------------------------------------- #
# Argument encoding (delta: arg_encoding)                                      #
# --------------------------------------------------------------------------- #

def _coerce_arguments_to_json_string(args: Any, *, name: str | None) -> str:
    """Return ``args`` as the JSON *string* OpenAI harnesses expect.

    GLM may hand us a native object/list, an already-encoded JSON string, or
    ``None``. We normalize all of them; anything else is loud.
    """
    if args is None:
        return "{}"
    if isinstance(args, (dict, list)):
        return json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if isinstance(args, str):
        # Already a string. Validate it is JSON; if not, fail loudly rather
        # than forward a payload the harness will choke on.
        try:
            json.loads(args)
        except (json.JSONDecodeError, TypeError) as exc:
            raise MalformedToolArguments(
                "tool-call arguments string is not valid JSON",
                name=name,
                raw=args,
            ) from exc
        return args
    raise MalformedToolArguments(
        f"tool-call arguments have unsupported type {type(args).__name__}",
        name=name,
        raw=args,
    )


# --------------------------------------------------------------------------- #
# Reasoning interleave (delta: reasoning_interleave)                           #
# --------------------------------------------------------------------------- #

def _split_reasoning(message: dict[str, Any]) -> str | None:
    """Pop GLM's reasoning trace out of a message, returning it (or ``None``).

    OpenAI tool turns expect ``content`` to be ``null`` when ``tool_calls`` is
    present. We relocate the reasoning into a side channel rather than dropping
    it, so callers that *want* the trace can still read it from the normalized
    result's ``_glm_reasoning`` key.
    """
    reasoning = message.pop("reasoning_content", None)
    # When a call is present, GLM sometimes also leaves prose in `content`.
    if message.get("tool_calls") and message.get("content"):
        spilled = message.pop("content")
        message["content"] = None
        if reasoning:
            reasoning = f"{reasoning}\n{spilled}"
        else:
            reasoning = spilled
    return reasoning


# --------------------------------------------------------------------------- #
# Parallel calls (delta: parallel_calls)                                       #
# --------------------------------------------------------------------------- #

def _normalize_one_call(call: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Coerce a single raw GLM call into the OpenAI per-call object shape."""
    if not isinstance(call, dict):
        raise UnsupportedProtocolShape(
            "expected a tool-call object", fragment=call
        )
    fn = call.get("function")
    if not isinstance(fn, dict) or "name" not in fn:
        raise UnsupportedProtocolShape(
            "tool call is missing a function.name", fragment=call
        )
    name = fn["name"]
    arguments = _coerce_arguments_to_json_string(fn.get("arguments"), name=name)
    call_id = call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _collect_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten GLM's parallel-call framing into one OpenAI tool_calls array."""
    raw_calls: list[Any] = []
    envelope = message.get("parallel_tool_calls")
    if isinstance(envelope, list):
        raw_calls.extend(envelope)
    flat = message.get("tool_calls")
    if isinstance(flat, list):
        raw_calls.extend(flat)
    return [_normalize_one_call(c, index=i) for i, c in enumerate(raw_calls)]


# --------------------------------------------------------------------------- #
# Streaming assembly (delta: streaming_assembly)                              #
# --------------------------------------------------------------------------- #

def assemble_stream(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble a list of streamed GLM chunks into one non-streaming response.

    Each chunk looks like ``{"choices": [{"delta": {...}, ...}]}``. Tool-call
    fragments carry an ``index`` and incremental ``function.arguments`` strings
    that must be concatenated. The result is a normal response dict (using
    ``message`` rather than ``delta``) ready for :func:`normalize`.
    """
    if not chunks:
        raise StreamAssemblyError("cannot assemble an empty stream")

    calls_by_index: dict[int, dict[str, Any]] = {}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    role = "assistant"
    head = copy.deepcopy(chunks[0])

    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if delta.get("role"):
            role = delta["role"]
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        for frag in delta.get("tool_calls") or []:
            idx = frag.get("index")
            if idx is None:
                raise StreamAssemblyError(
                    "streamed tool-call fragment has no index"
                )
            slot = calls_by_index.setdefault(
                idx,
                {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
            )
            if frag.get("id"):
                slot["id"] = frag["id"]
            fn = frag.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]

    # Validate each assembled call has a name and balanced JSON arguments.
    assembled: list[dict[str, Any]] = []
    for idx in sorted(calls_by_index):
        slot = calls_by_index[idx]
        if not slot["function"]["name"]:
            raise StreamAssemblyError(
                f"streamed tool call at index {idx} never received a name"
            )
        args = slot["function"]["arguments"] or "{}"
        try:
            json.loads(args)
        except (json.JSONDecodeError, TypeError) as exc:
            raise StreamAssemblyError(
                f"streamed tool call at index {idx} ended with unbalanced JSON arguments"
            ) from exc
        slot["function"]["arguments"] = args
        assembled.append(slot)

    message: dict[str, Any] = {"role": role}
    message["content"] = "".join(content_parts) if content_parts else None
    if assembled:
        message["tool_calls"] = assembled
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)

    rebuilt_choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason or ("tool_calls" if assembled else "stop"),
    }
    head_choice = (head.get("choices") or [{}])[0]
    head_choice.pop("delta", None)
    head_choice.update(rebuilt_choice)
    head["choices"] = [head_choice]
    head["object"] = "chat.completion"
    return head


# --------------------------------------------------------------------------- #
# Top-level: GLM response -> OpenAI shape                                      #
# --------------------------------------------------------------------------- #

def normalize(glm_response: dict[str, Any]) -> dict[str, Any]:
    """Convert a (non-streaming) GLM-5.2 response into the OpenAI shape.

    For streamed responses, call :func:`assemble_stream` first.

    The returned dict is a deep copy; the input is never mutated. If a tool
    call is present, ``message.content`` is forced to ``null`` and any reasoning
    trace is relocated to ``message._glm_reasoning`` for callers that want it.
    """
    if not isinstance(glm_response, dict):
        raise UnsupportedProtocolShape("response is not an object", fragment=glm_response)

    out = copy.deepcopy(glm_response)
    choices = out.get("choices")
    if not isinstance(choices, list) or not choices:
        # No choices to touch (e.g. an error envelope) — pass through unchanged.
        return out

    for choice in choices:
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        has_calls = bool(message.get("tool_calls")) or bool(
            message.get("parallel_tool_calls")
        )
        reasoning = _split_reasoning(message)

        if has_calls:
            calls = _collect_calls(message)
            message.pop("parallel_tool_calls", None)
            message["tool_calls"] = calls
            message["content"] = None
            if "finish_reason" not in choice or choice.get("finish_reason") in (None, "stop"):
                choice["finish_reason"] = "tool_calls"

        if reasoning is not None:
            message["_glm_reasoning"] = reasoning

    out["object"] = out.get("object") or "chat.completion"
    return out


# --------------------------------------------------------------------------- #
# Top-level: OpenAI tool defs -> GLM request                                   #
# --------------------------------------------------------------------------- #

def denormalize_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lower OpenAI-shaped tool *definitions* into GLM-5.2's accepted form.

    GLM-5.2 accepts the OpenAI ``tools`` schema closely, but is strict about a
    few things harnesses are lax on:

    * every tool must have ``type == "function"`` and a ``function.name``;
    * ``function.parameters`` must be a JSON-schema object (default to an empty
      object schema if omitted).

    Returns a new list; the input is not mutated.
    """
    lowered: list[dict[str, Any]] = []
    for i, tool in enumerate(openai_tools or []):
        if not isinstance(tool, dict):
            raise UnsupportedProtocolShape(
                f"tool definition at index {i} is not an object", fragment=tool
            )
        fn = tool.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            raise UnsupportedProtocolShape(
                f"tool definition at index {i} is missing function.name",
                fragment=tool,
            )
        params = fn.get("parameters")
        if params is None:
            params = {"type": "object", "properties": {}}
        lowered_tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": fn["name"],
                "parameters": params,
            },
        }
        if fn.get("description"):
            lowered_tool["function"]["description"] = fn["description"]
        lowered.append(lowered_tool)
    return lowered

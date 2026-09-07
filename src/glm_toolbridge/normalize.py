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
    if args is None or args == "":
        # None and the empty string both mean "no arguments". Treat "" as
        # "{}" to match assemble_stream's ``or "{}"`` and to avoid rejecting
        # a no-arg call shape the OpenAI SDK accepts as a plain str field.
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


def _coerce_arguments_fragment(args: Any) -> str:
    """Coerce a streamed tool-call argument FRAGMENT to a string.

    Unlike :func:`_coerce_arguments_to_json_string`, this does NOT validate the
    fragment as whole JSON: OpenAI streaming contracts deliver ``function.arguments``
    as *partial* string fragments (e.g. ``'{"path": "out'``) that the harness
    concatenates and parses itself once complete. Validating each fragment as whole
    JSON would therefore raise on every valid partial payload. We only guarantee the
    OpenAI invariant — ``arguments`` is a STRING — without asserting it is parseable.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    if isinstance(args, (dict, list)):
        return json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    # An unexpected scalar type in a fragment — coerce to its repr rather than
    # crash an otherwise valid stream.
    return str(args)


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
    # Use the same has_calls condition normalize() uses (flat tool_calls OR
    # parallel_tool_calls) so a parallel_tool_calls-only envelope with spilled
    # content relocates the prose here rather than having normalize() null it.
    if (message.get("tool_calls") or message.get("parallel_tool_calls")) and message.get("content"):
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

    Multiple choices (``choices`` with differing ``index``) are assembled
    independently and rebuilt in index order — earlier versions silently dropped
    every choice beyond ``choices[0]``.
    """
    if not chunks:
        raise StreamAssemblyError("cannot assemble an empty stream")

    # Per-choice assembled state, keyed by the choice index (default 0 for
    # chunks that omit it, matching the OpenAI single-choice convention).
    by_choice: dict[int, dict[str, Any]] = {}
    head = copy.deepcopy(chunks[0])

    for chunk in chunks:
        choices = chunk.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            cidx = choice.get("index", 0)
            if not isinstance(cidx, int):
                cidx = 0
            st = by_choice.setdefault(cidx, {
                "content_parts": [],
                "reasoning_parts": [],
                "calls_by_index": {},
                "finish_reason": None,
                "role": "assistant",
            })
            if choice.get("finish_reason"):
                st["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("role"):
                st["role"] = delta["role"]
            if delta.get("content"):
                st["content_parts"].append(delta["content"])
            if delta.get("reasoning_content"):
                st["reasoning_parts"].append(delta["reasoning_content"])
            for frag in delta.get("tool_calls") or []:
                fidx = frag.get("index") if isinstance(frag, dict) else None
                if fidx is None:
                    raise StreamAssemblyError(
                        "streamed tool-call fragment has no index"
                    )
                slot = st["calls_by_index"].setdefault(
                    fidx,
                    {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
                )
                if frag.get("id"):
                    slot["id"] = frag["id"]
                fn = frag.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    # Coerce the fragment to a string before concatenating. GLM's
                    # arg_encoding delta may emit arguments as a native object even
                    # in a streamed fragment; ``+=`` on a dict raises TypeError. The
                    # per-chunk path (normalize_delta_chunk) already coerces via
                    # _coerce_arguments_fragment; assemble_stream must do the same.
                    slot["function"]["arguments"] += _coerce_arguments_fragment(
                        fn["arguments"]
                    )

    rebuilt_choices: list[dict[str, Any]] = []
    for cidx in sorted(by_choice):
        st = by_choice[cidx]
        assembled: list[dict[str, Any]] = []
        for fidx in sorted(st["calls_by_index"]):
            slot = st["calls_by_index"][fidx]
            if not slot["function"]["name"]:
                raise StreamAssemblyError(
                    f"streamed tool call at index {fidx} (choice {cidx}) "
                    "never received a name"
                )
            args = slot["function"]["arguments"] or "{}"
            try:
                json.loads(args)
            except (json.JSONDecodeError, TypeError) as exc:
                raise StreamAssemblyError(
                    f"streamed tool call at index {fidx} (choice {cidx}) "
                    "ended with unbalanced JSON arguments"
                ) from exc
            slot["function"]["arguments"] = args
            assembled.append(slot)

        message: dict[str, Any] = {"role": st["role"]}
        message["content"] = "".join(st["content_parts"]) if st["content_parts"] else None
        if assembled:
            message["tool_calls"] = assembled
        if st["reasoning_parts"]:
            message["reasoning_content"] = "".join(st["reasoning_parts"])
        finish = st["finish_reason"] or ("tool_calls" if assembled else "stop")
        rebuilt_choices.append(
            {"index": cidx, "message": message, "finish_reason": finish}
        )

    head["choices"] = rebuilt_choices

    # Merge late top-level fields that chunks[0] lacked. Per the OpenAI
    # streaming contract (and GLM's OpenAI-compatible endpoint under
    # stream_options.include_usage), metadata such as ``usage`` token stats
    # arrives in the FINAL chunk — typically with an empty ``choices`` list —
    # so a head seeded from ``chunks[0]`` would silently drop it. Carry any
    # non-choices top-level field from the chunk that carries it into head.
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for key, value in chunk.items():
            if key == "choices":
                continue
            if value is None:
                continue
            if key not in head or head[key] is None:
                head[key] = value

    # Reassembly by definition yields a complete non-streaming completion, not a
    # stream of deltas. The head was seeded from chunks[0], so it inherited the
    # chunk value "chat.completion.chunk"; relabel it so a harness switching on
    # ``object`` routes the reassembled response as a completion (and so
    # re-validation against the OpenAI SDK's ``Literal["chat.completion"]`` passes).
    head["object"] = "chat.completion"
    return head


def normalize_delta_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single streamed GLM delta chunk for an incremental stream.

    The drop-in streaming path (:func:`glm_toolbridge.client.wrap` with
    ``stream=True``) yields chunks one at a time so the harness assembles argument
    fragments itself, exactly as it would for an OpenAI stream. This function fixes
    the per-chunk divergences without full reassembly:

    * tool-call argument FRAGMENTS are coerced to strings *without* whole-JSON
      validation (OpenAI streaming fragments are partial by contract);
    * GLM's ``reasoning_content`` is relocated out of the delta into a
      ``_glm_reasoning`` side channel (deltas keep their incremental ``content``);
    * any ``parallel_tool_calls`` envelope on a delta is flattened into
      ``tool_calls`` so harnesses reading a flat array see the fragments.

    The input is never mutated; a deep copy is returned.
    """
    if not isinstance(chunk, dict):
        raise UnsupportedProtocolShape("stream chunk is not an object", fragment=chunk)

    out = copy.deepcopy(chunk)
    choices = out.get("choices")
    if not isinstance(choices, list):
        return out

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        reasoning = delta.pop("reasoning_content", None)

        # Flatten a parallel_tool_calls envelope on a delta into the flat array.
        parallel = delta.pop("parallel_tool_calls", None)
        if isinstance(parallel, list):
            merged = list(delta.get("tool_calls") or [])
            for frag in parallel:
                if isinstance(frag, dict):
                    if "type" not in frag:
                        frag["type"] = "function"
                    merged.append(frag)
            delta["tool_calls"] = merged

        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for frag in calls:
                if not isinstance(frag, dict):
                    continue
                fn = frag.get("function")
                if isinstance(fn, dict) and "arguments" in fn:
                    fn["arguments"] = _coerce_arguments_fragment(fn["arguments"])

        if reasoning is not None:
            delta["_glm_reasoning"] = reasoning

    out["object"] = out.get("object") or "chat.completion.chunk"
    return out


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
        if not isinstance(params, dict):
            # A malformed tool definition — parameters must be a JSON-schema
            # object. Forwarding a string/list/scalar to GLM would surface as an
            # opaque request-schema error far from the offending tool; raise
            # loudly here instead, mirroring the tool-call normalizer's pattern.
            raise UnsupportedProtocolShape(
                f"tool definition {fn['name']!r} has a non-object parameters schema",
                fragment=params,
            )
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

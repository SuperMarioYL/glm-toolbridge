"""The documented divergence between GLM-5.2 and OpenAI tool-call wire shapes.

This module is the m1 *audit* made executable. Each :class:`ProtocolDelta`
names one concrete way GLM-5.2's function-calling protocol differs from the
canonical OpenAI ``tool_calls`` shape, and ships a ``detect`` predicate that
returns ``True`` when a given (raw) response exhibits that divergence.

The deltas are the single source of truth shared by:

* ``docs/PROTOCOL_DELTAS.md`` — the human-readable audit,
* ``normalize.py`` — the transforms that *fix* each divergence,
* ``tests/test_protocol_deltas.py`` — which asserts every delta is detectable
  on a captured fixture (the falsification gate: if the deltas collapse to a
  single trivial rename, the product has no reason to exist).

A note on the wire shapes
-------------------------
*OpenAI* puts tool calls under ``choices[i].message.tool_calls``, each an object
``{"id", "type": "function", "function": {"name", "arguments"}}`` where
``arguments`` is a **JSON-encoded string**.

*GLM-5.2* diverges in four documented places, enumerated by :data:`DELTAS`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class DeltaKind(str, Enum):
    """The four axes on which GLM-5.2 diverges from OpenAI tool calls."""

    ARG_ENCODING = "arg_encoding"
    PARALLEL_CALLS = "parallel_calls"
    REASONING_INTERLEAVE = "reasoning_interleave"
    STREAMING_ASSEMBLY = "streaming_assembly"


@dataclass(frozen=True)
class ProtocolDelta:
    """One named, detectable difference between the two protocols.

    Attributes
    ----------
    kind:
        Which axis this delta lives on.
    summary:
        One-line human description (mirrored in ``PROTOCOL_DELTAS.md``).
    glm_shape:
        How GLM-5.2 encodes this aspect on the wire.
    openai_shape:
        How OpenAI encodes the same aspect.
    detect:
        Predicate over a *raw* GLM-style response dict that returns ``True``
        when the response exhibits this divergence. Detection is intentionally
        conservative: it returns ``False`` (rather than raising) on shapes it
        does not recognize, so a single odd field never masks the others.
    """

    kind: DeltaKind
    summary: str
    glm_shape: str
    openai_shape: str
    detect: Callable[[dict[str, Any]], bool] = field(repr=False)


# --------------------------------------------------------------------------- #
# Detection helpers — each isolates the divergence on one axis.               #
# --------------------------------------------------------------------------- #

def _iter_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the raw tool-call list out of a GLM-shaped response, if any."""
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls")
    return list(calls) if isinstance(calls, list) else []


def _detect_arg_encoding(response: dict[str, Any]) -> bool:
    """GLM may emit ``function.arguments`` as a native object, not a JSON string.

    OpenAI *always* sends a JSON-encoded string. So the divergence is present
    whenever any tool call carries arguments that are a ``dict`` (or any
    non-``str``), or a string that is not itself valid JSON.
    """
    for call in _iter_tool_calls(response):
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if args is None:
            continue
        if not isinstance(args, str):
            return True  # native object / list — divergent
        try:
            json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return True  # string but not parseable JSON — divergent
    return False


def _detect_parallel_calls(response: dict[str, Any]) -> bool:
    """GLM frames multiple simultaneous calls differently from OpenAI's flat array.

    OpenAI returns N parallel calls as N entries in one ``tool_calls`` array,
    each with a stable ``id``. GLM-5.2 surfaces them either nested under a
    ``parallel_tool_calls`` envelope or as array entries that lack the per-call
    ``id``/``type`` scaffolding OpenAI guarantees. Either framing is divergent.
    """
    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        if isinstance(message.get("parallel_tool_calls"), list):
            return True
    calls = _iter_tool_calls(response)
    if len(calls) > 1:
        # Multiple calls present: divergent if any lacks the OpenAI scaffolding.
        for call in calls:
            if "id" not in call or call.get("type") != "function":
                return True
    return False


def _detect_reasoning_interleave(response: dict[str, Any]) -> bool:
    """GLM-5.2 interleaves a chain-of-thought trace alongside tool calls.

    The reasoning text rides on ``message.reasoning_content`` (and sometimes is
    spliced into ``message.content`` ahead of the call). OpenAI tool-call turns
    carry no reasoning field, so its presence is the divergence — the harness
    expects ``content`` to be ``null`` when ``tool_calls`` is set.
    """
    choices = response.get("choices") or []
    if not choices:
        return False
    message = choices[0].get("message") or {}
    if message.get("reasoning_content"):
        return True
    # Reasoning spliced into content while tool_calls are also present.
    if message.get("tool_calls") and message.get("content"):
        return True
    return False


def _detect_streaming_assembly(response: dict[str, Any]) -> bool:
    """A streamed GLM response arrives as deltas that must be reassembled.

    A streaming chunk uses ``choices[i].delta`` (not ``message``) and its
    ``tool_calls`` entries carry incremental ``arguments`` fragments keyed by
    ``index``. The presence of a ``delta`` with tool-call fragments is the
    divergence the assembler must handle.
    """
    choices = response.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("tool_calls"), list):
        return True
    return False


# --------------------------------------------------------------------------- #
# The audit: the four deltas, in documentation order.                          #
# --------------------------------------------------------------------------- #

DELTAS: tuple[ProtocolDelta, ...] = (
    ProtocolDelta(
        kind=DeltaKind.ARG_ENCODING,
        summary="GLM may return tool arguments as a native JSON object; OpenAI "
        "always sends a JSON-encoded string.",
        glm_shape='function.arguments == {"path": "x"}  (object) or a non-JSON string',
        openai_shape='function.arguments == "{\\"path\\": \\"x\\"}"  (JSON string)',
        detect=_detect_arg_encoding,
    ),
    ProtocolDelta(
        kind=DeltaKind.PARALLEL_CALLS,
        summary="GLM frames multiple simultaneous calls under a "
        "`parallel_tool_calls` envelope or without per-call id/type scaffolding.",
        glm_shape="message.parallel_tool_calls: [...] or tool_calls without id/type",
        openai_shape="message.tool_calls: [{id, type:'function', function}, ...]",
        detect=_detect_parallel_calls,
    ),
    ProtocolDelta(
        kind=DeltaKind.REASONING_INTERLEAVE,
        summary="GLM-5.2 carries a chain-of-thought trace "
        "(`reasoning_content`) alongside the tool call; OpenAI tool turns do not.",
        glm_shape="message.reasoning_content: '...' (+ tool_calls)",
        openai_shape="message.content == null when tool_calls is set",
        detect=_detect_reasoning_interleave,
    ),
    ProtocolDelta(
        kind=DeltaKind.STREAMING_ASSEMBLY,
        summary="Streamed GLM tool calls arrive as `delta` fragments keyed by "
        "index that must be reassembled into one complete call.",
        glm_shape="choices[].delta.tool_calls[].function.arguments (fragments)",
        openai_shape="reassembled into choices[].message.tool_calls[]",
        detect=_detect_streaming_assembly,
    ),
)


def deltas_present(response: dict[str, Any]) -> list[DeltaKind]:
    """Return the kinds of every delta this raw response exhibits.

    Used by the audit tests and by :mod:`glm_toolbridge.adapter` to decide which
    transforms to run. Order follows :data:`DELTAS`.
    """
    return [d.kind for d in DELTAS if d.detect(response)]


def get_delta(kind: DeltaKind) -> ProtocolDelta:
    """Look up a single delta by kind."""
    for d in DELTAS:
        if d.kind == kind:
            return d
    raise KeyError(kind)  # pragma: no cover - defensive

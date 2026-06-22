"""Typed, composing layer over the pure transforms in :mod:`normalize`.

``adapter.py`` is where the four protocol deltas come together into the two
public verbs of the bridge:

* :func:`normalize_response` — GLM-5.2 response (or stream) -> validated
  OpenAI ``tool_calls`` shape, returned both as a typed
  :class:`OpenAIChatCompletion` and as the plain dict a harness consumes.
* :func:`denormalize_request` — an OpenAI-shaped request body (``tools`` etc.)
  -> the request body GLM-5.2 accepts.

The Pydantic models give the roundtrip tests a strict contract to assert
against: if ``normalize`` ever produces a shape that does not validate as
:class:`OpenAIChatCompletion`, the test fails — that *is* the m2 falsification
gate.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from . import normalize as _norm
from .protocol import DeltaKind, deltas_present


# --------------------------------------------------------------------------- #
# Typed OpenAI tool-call models (the canonical shape harnesses parse).         #
# --------------------------------------------------------------------------- #

class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: str  # JSON-encoded string, per the OpenAI contract

    def parsed_arguments(self) -> Any:
        """Decode the JSON-string arguments back into a Python object."""
        return json.loads(self.arguments)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    # `extra=allow` so the relocated `_glm_reasoning` side-channel survives.
    model_config = ConfigDict(extra="allow")
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @property
    def glm_reasoning(self) -> str | None:
        """The GLM chain-of-thought trace relocated out of the call block."""
        return self.model_extra.get("_glm_reasoning") if self.model_extra else None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class OpenAIChatCompletion(BaseModel):
    """Validated OpenAI-shaped chat completion — what the harness reads."""

    model_config = ConfigDict(extra="allow")
    object: str = "chat.completion"
    choices: list[Choice]

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Convenience: the tool calls on the first choice (empty if none)."""
        if not self.choices:
            return []
        return self.choices[0].message.tool_calls or []


# --------------------------------------------------------------------------- #
# Public verbs.                                                                #
# --------------------------------------------------------------------------- #

class NormalizeResult(BaseModel):
    """The output of :func:`normalize_response`: typed view + raw dict + audit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    completion: OpenAIChatCompletion
    raw: dict[str, Any] = Field(repr=False)
    deltas_applied: list[DeltaKind]

    def as_openai_dict(self) -> dict[str, Any]:
        """The plain dict a stock OpenAI-format harness expects to receive."""
        return self.raw


def normalize_response(
    glm_response: dict[str, Any] | list[dict[str, Any]],
) -> NormalizeResult:
    """GLM-5.2 response (or streamed chunk list) -> validated OpenAI shape.

    Pass a list of chunks to reassemble a streamed response first.
    """
    applied: list[DeltaKind] = []

    if isinstance(glm_response, list):
        applied.append(DeltaKind.STREAMING_ASSEMBLY)
        source = _norm.assemble_stream(glm_response)
    else:
        source = glm_response

    # Record which divergences the (pre-normalization) source exhibited.
    for kind in deltas_present(source):
        if kind not in applied:
            applied.append(kind)

    normalized = _norm.normalize(source)
    completion = OpenAIChatCompletion.model_validate(normalized)
    return NormalizeResult(completion=completion, raw=normalized, deltas_applied=applied)


def denormalize_request(openai_request: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI-shaped request body -> the body GLM-5.2 accepts.

    Only the ``tools`` field needs lowering today; everything else passes
    through untouched. Returns a new dict.
    """
    out = dict(openai_request)
    if isinstance(out.get("tools"), list):
        out["tools"] = _norm.denormalize_tools(out["tools"])
    return out

"""Named, loud errors for protocol shapes glm-toolbridge cannot reconcile.

The whole point of this library is that GLM-5.2 tool-call breakage stops being
*silent*. When GLM emits a wire shape that no documented :class:`ProtocolDelta`
covers, we raise a clear, named exception instead of returning a half-parsed
structure that the harness will choke on three frames later.
"""

from __future__ import annotations

from typing import Any


class ToolBridgeError(Exception):
    """Base class for every error raised by glm-toolbridge."""


class UnsupportedProtocolShape(ToolBridgeError):
    """GLM returned a tool-call shape that no delta knows how to normalize.

    Raised by the normalizer when a response field is present but structured in
    a way the adapter does not recognize. Carries the offending fragment so the
    caller can file a precise bug report (and so we can add a delta for it).
    """

    def __init__(self, message: str, *, fragment: Any | None = None) -> None:
        super().__init__(message)
        self.fragment = fragment

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = super().__str__()
        if self.fragment is not None:
            return f"{base} (offending fragment: {self.fragment!r})"
        return base


class MalformedToolArguments(ToolBridgeError):
    """A tool call's arguments could not be coerced to the OpenAI JSON-string form.

    GLM-5.2 sometimes emits arguments as a native JSON object and sometimes as a
    string; harnesses expect a JSON *string*. If neither path yields valid JSON
    we fail loudly rather than forwarding garbage.
    """

    def __init__(self, message: str, *, name: str | None = None, raw: Any | None = None) -> None:
        super().__init__(message)
        self.name = name
        self.raw = raw


class StreamAssemblyError(ToolBridgeError):
    """Streaming tool-call deltas could not be reassembled into a complete call.

    Raised when GLM's streamed fragments reference an index that was never
    opened, or when a stream ends mid-arguments with unbalanced JSON.
    """


class DenormalizeError(ToolBridgeError):
    """OpenAI-shaped tool definitions could not be lowered into a GLM request."""

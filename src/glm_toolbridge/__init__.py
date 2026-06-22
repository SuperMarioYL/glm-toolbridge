"""glm-toolbridge — reconcile GLM-5.2 tool calls with the OpenAI tool_calls schema.

A thin protocol adapter. Wrap your existing OpenAI-SDK client in one line and
the tool-call loop that was silently mis-parsing against GLM-5.2 now parses
correctly — your harness's OpenAI code path is untouched.

    from openai import OpenAI
    from glm_toolbridge import wrap

    client = wrap(OpenAI(base_url=GLM_URL, api_key="..."))
    resp = client.chat.completions.create(model="glm-5.2", messages=..., tools=...)
    resp.choices[0].message.tool_calls  # valid OpenAI shape

Lower-level building blocks are also exported for callers who want the pure
transforms without the client wrapper.
"""

from __future__ import annotations

from .adapter import (
    NormalizeResult,
    OpenAIChatCompletion,
    ToolCall,
    denormalize_request,
    normalize_response,
)
from .client import GLM_DEFAULT_BASE_URL, BridgedClient, wrap
from .errors import (
    DenormalizeError,
    MalformedToolArguments,
    StreamAssemblyError,
    ToolBridgeError,
    UnsupportedProtocolShape,
)
from .normalize import assemble_stream, denormalize_tools, normalize
from .protocol import DELTAS, DeltaKind, ProtocolDelta, deltas_present, get_delta

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # client
    "wrap",
    "BridgedClient",
    "GLM_DEFAULT_BASE_URL",
    # adapter (typed)
    "normalize_response",
    "denormalize_request",
    "NormalizeResult",
    "OpenAIChatCompletion",
    "ToolCall",
    # normalize (pure)
    "normalize",
    "denormalize_tools",
    "assemble_stream",
    # protocol (audit)
    "DELTAS",
    "DeltaKind",
    "ProtocolDelta",
    "deltas_present",
    "get_delta",
    # errors
    "ToolBridgeError",
    "UnsupportedProtocolShape",
    "MalformedToolArguments",
    "StreamAssemblyError",
    "DenormalizeError",
]

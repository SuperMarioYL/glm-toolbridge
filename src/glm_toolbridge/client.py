"""The drop-in wrapper: ``wrap(client)`` and the loop never knows GLM is behind it.

A coding-agent harness does exactly one thing with its client:

    resp = client.chat.completions.create(model=..., messages=..., tools=...)

and then reads ``resp.choices[0].message.tool_calls``. :func:`wrap` returns a
transparent proxy with the *same* interface, so the harness's OpenAI code path
is unchanged — but on the way out it lowers ``tools`` to GLM's accepted shape
(:func:`~glm_toolbridge.adapter.denormalize_request`) and on the way back it
normalizes GLM's response to the OpenAI ``tool_calls`` shape
(:func:`~glm_toolbridge.adapter.normalize_response`).

The wrapper is intentionally tiny and uses ``__getattr__`` delegation so any
attribute the harness touches that we don't override (``.models``, ``.embeddings``,
custom headers, …) reaches the real client untouched.
"""

from __future__ import annotations

from typing import Any

from .adapter import denormalize_request, normalize_response

# Default 智谱 GLM OpenAI-compatible base URL. Override via wrap(..., base_url=...)
# or by constructing the underlying client with your own base_url.
GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class _BridgedCompletions:
    """Wraps ``client.chat.completions`` to bridge the protocol on each call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        # 1. Lower the OpenAI-shaped request (tool defs) into GLM's form.
        glm_kwargs = denormalize_request(kwargs)

        # GLM streaming would need delta accumulation by the caller; for the
        # drop-in path we request a complete response and normalize it whole.
        glm_kwargs.pop("stream", None)

        raw_response = self._inner.create(**glm_kwargs)

        # 2. The OpenAI SDK returns a pydantic model; reduce to a plain dict so
        #    our transforms operate on the wire shape, then rebuild a model the
        #    harness can read identically.
        as_dict = _to_dict(raw_response)
        result = normalize_response(as_dict)
        return _rebuild_like(raw_response, result.as_openai_dict())

    def __getattr__(self, name: str) -> Any:  # delegate everything else
        return getattr(self._inner, name)


class _BridgedChat:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.completions = _BridgedCompletions(inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class BridgedClient:
    """Transparent proxy over an OpenAI-SDK client that speaks GLM-5.2 underneath.

    Only ``.chat.completions.create`` is intercepted; every other attribute is
    delegated to the wrapped client, so this is a true drop-in.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.chat = _BridgedChat(inner.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap(client: Any) -> BridgedClient:
    """Wrap an ``openai.OpenAI`` (or compatible) client for GLM-5.2.

    Usage::

        from openai import OpenAI
        from glm_toolbridge import wrap

        client = wrap(OpenAI(base_url=GLM_URL, api_key="..."))
        resp = client.chat.completions.create(model="glm-5.2", messages=..., tools=...)
        # resp.choices[0].message.tool_calls is now valid OpenAI shape

    The wrapped client's interface is identical to the original; the only change
    is that GLM's tool-call divergences are reconciled on the way through.
    """
    return BridgedClient(client)


# --------------------------------------------------------------------------- #
# SDK-shape helpers (kept dependency-light so unit tests can pass plain dicts).#
# --------------------------------------------------------------------------- #

def _to_dict(response: Any) -> dict[str, Any]:
    """Reduce an OpenAI-SDK response object (or a dict) to a plain dict."""
    if isinstance(response, dict):
        return response
    # openai>=1.x models expose model_dump(); fall back to to_dict / __dict__.
    for attr in ("model_dump", "to_dict"):
        fn = getattr(response, attr, None)
        if callable(fn):
            return fn()
    raise TypeError(
        f"cannot reduce response of type {type(response).__name__} to a dict"
    )


def _rebuild_like(template: Any, normalized: dict[str, Any]) -> Any:
    """Rebuild a response of the same class as ``template`` from a normalized dict.

    If the template is an OpenAI-SDK pydantic model we re-validate the dict back
    into that class so the harness keeps getting ``.choices[0].message`` access.
    If the template was a plain dict (unit-test path), return the dict.
    """
    if isinstance(template, dict):
        return normalized
    cls = type(template)
    for ctor in ("model_validate", "construct", "parse_obj"):
        fn = getattr(cls, ctor, None)
        if callable(fn):
            try:
                return fn(normalized)
            except Exception:  # pragma: no cover - fall through to next ctor
                continue
    return normalized  # pragma: no cover - last-resort passthrough

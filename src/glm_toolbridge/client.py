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

from typing import Any, Iterator

from .adapter import denormalize_request, normalize_response
from .errors import UnsupportedProtocolShape
from .normalize import normalize_delta_chunk

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

        # 2. Streaming: forward stream and yield normalized delta chunks so the
        #    harness assembles argument fragments itself, exactly as it would for
        #    an OpenAI stream. (v0.1 silently popped stream and forced a complete
        #    response — the silent-misbehavior this library exists to prevent.)
        if kwargs.get("stream"):
            return self._stream(**glm_kwargs)

        raw_response = self._inner.create(**glm_kwargs)

        # 3. The OpenAI SDK returns a pydantic model; reduce to a plain dict so
        #    our transforms operate on the wire shape, then rebuild a model the
        #    harness can read identically.
        as_dict = _to_dict(raw_response)
        result = normalize_response(as_dict)
        return _rebuild_like(raw_response, result.as_openai_dict())

    def _stream(self, **kwargs: Any) -> Iterator[Any]:
        # The inner client's create(stream=True) returns an iterable of chunk
        # objects (or dicts). Normalize each delta and rebuild it in the SDK shape
        # so the harness reads a stream of OpenAI-shaped chunks.
        inner_stream = self._inner.create(**kwargs)
        for chunk in inner_stream:
            as_dict = _to_dict(chunk)
            normalized = normalize_delta_chunk(as_dict)
            yield _rebuild_like(chunk, normalized)

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

    ``stream=True`` is honored: the bridged ``create`` returns an iterator of
    normalized delta chunks (argument fragments coerced to strings) so a
    streaming harness assembles tool calls exactly as it would for an OpenAI
    stream. For async clients, use :func:`awrap`.
    """
    return BridgedClient(client)


# --------------------------------------------------------------------------- #
# Async wrapper — mirrors the sync path for AsyncOpenAI clients.              #
# --------------------------------------------------------------------------- #

class _AsyncBridgedCompletions:
    """Wraps ``client.chat.completions`` (async) to bridge the protocol on each call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def create(self, **kwargs: Any) -> Any:
        glm_kwargs = denormalize_request(kwargs)

        if kwargs.get("stream"):
            # The inner async create(stream=True) returns an async stream once
            # awaited; wrap it so each yielded chunk is a normalized delta.
            inner_stream = await self._inner.create(**glm_kwargs)
            return self._bridged_stream(inner_stream)

        raw_response = await self._inner.create(**glm_kwargs)
        as_dict = _to_dict(raw_response)
        result = normalize_response(as_dict)
        return _rebuild_like(raw_response, result.as_openai_dict())

    def _bridged_stream(self, inner_stream: Any) -> Any:
        async def _gen() -> Any:
            async for chunk in inner_stream:
                as_dict = _to_dict(chunk)
                normalized = normalize_delta_chunk(as_dict)
                yield _rebuild_like(chunk, normalized)

        return _gen()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AsyncBridgedChat:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.completions = _AsyncBridgedCompletions(inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AsyncBridgedClient:
    """Transparent async proxy over an ``AsyncOpenAI`` (or compatible) client.

    Only ``.chat.completions.create`` is intercepted; every other attribute is
    delegated to the wrapped async client. Mirrors :class:`BridgedClient`.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.chat = _AsyncBridgedChat(inner.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def awrap(client: Any) -> AsyncBridgedClient:
    """Wrap an ``openai.AsyncOpenAI`` (or compatible) async client for GLM-5.2.

    Mirrors :func:`wrap` for async harnesses: ``await client.chat.completions.create(...)``
    returns a normalized completion, and ``stream=True`` returns an async iterator
    of normalized delta chunks. No new third-party deps (``asyncio`` is stdlib).
    """
    return AsyncBridgedClient(client)


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

    Failures are loud (per errors.py / plan §4: "failures are loud, never
    silent"). When a validating constructor (``model_validate`` / ``parse_obj``)
    rejects the normalized shape, raise :class:`UnsupportedProtocolShape` carrying
    the underlying error instead of silently degrading through an unvalidated
    ``construct`` or returning a bare dict masquerading as a structured response.
    """
    if isinstance(template, dict):
        return normalized
    cls = type(template)
    # Validating constructors first. A failure here is a real shape divergence
    # the caller must see — never swallowed.
    for ctor in ("model_validate", "parse_obj"):
        fn = getattr(cls, ctor, None)
        if not callable(fn):
            continue
        try:
            return fn(normalized)
        except Exception as exc:
            raise UnsupportedProtocolShape(
                f"normalized response could not be re-validated as "
                f"{cls.__name__}: {exc}",
                fragment=normalized,
            ) from exc
    # No validating constructor exists for this response type. Refuse to return
    # an untyped dict as a silent substitute — the caller must see the mismatch.
    raise UnsupportedProtocolShape(
        f"cannot rebuild response of type {cls.__name__}: "
        f"no validating constructor (model_validate/parse_obj) available",
        fragment=normalized,
    )

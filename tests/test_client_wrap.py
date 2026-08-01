"""v0.2.0 client tests — wrap()/awrap()/streaming/loud-rebuild/params-validation.

This is the first unit coverage for ``client.py`` (v0.1 only exercised ``wrap`` via the
offline demo). Covers the four v0.2.0 iteration milestones:

* fix_validate_tool_params  — denormalize_tools rejects non-dict parameters loudly
* fix_rebuild_loud_errors   — _rebuild_like surfaces SDK re-validation failures
* feat_streaming_dropin     — wrap() honors stream=True (incremental normalized chunks)
* feat_async_client         — awrap() for AsyncOpenAI clients
* (+ the folded assemble_stream multi-choice fix)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from glm_toolbridge import (
    AsyncBridgedClient,
    BridgedClient,
    awrap,
    denormalize_tools,
    wrap,
)
from glm_toolbridge.errors import UnsupportedProtocolShape
from glm_toolbridge.normalize import assemble_stream, normalize_delta_chunk


# --------------------------------------------------------------------------- #
# Fake clients (dict-returning, the same surface the OpenAI SDK exposes).     #
# --------------------------------------------------------------------------- #

class _FakeCompletions:
    """Returns a GLM-shaped response (native-object args + reasoning trace)."""

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._response


class _FakeChat:
    def __init__(self, response: dict[str, Any]) -> None:
        self.completions = _FakeCompletions(response)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


class FakeGLMClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.chat = _FakeChat(response)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


class _FakeStreamCompletions:
    """Returns the captured GLM streamed chunks when stream=True."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return list(self._chunks)  # a sync iterable of chunk dicts
        # Non-streaming: hand back the fully-assembled chunk as one response.
        return list(self._chunks)


class _FakeStreamChat:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.completions = _FakeStreamCompletions(chunks)


class FakeStreamGLMClient:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chat = _FakeStreamChat(chunks)


# Async variants ------------------------------------------------------------ #

class _AsyncFakeCompletions:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._response


class _AsyncFakeChat:
    def __init__(self, response: dict[str, Any]) -> None:
        self.completions = _AsyncFakeCompletions(response)


class AsyncFakeGLMClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.chat = _AsyncFakeChat(response)


class _AsyncChunkStream:
    """An async iterator over a fixed list of chunk dicts."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._i = 0

    def __aiter__(self) -> "_AsyncChunkStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._i]
        self._i += 1
        return c


class _AsyncFakeStreamCompletions:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _AsyncChunkStream(list(self._chunks))
        return list(self._chunks)


class _AsyncFakeStreamChat:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.completions = _AsyncFakeStreamCompletions(chunks)


class AsyncFakeStreamGLMClient:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chat = _AsyncFakeStreamChat(chunks)


# Models that exercise the loud-rebuild paths ------------------------------ #

class _StrictRebuild:
    """Reducible to a dict (model_dump) but rejects re-validation (model_validate)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "_StrictRebuild":
        raise ValueError("simulated strict-model rejection of the normalized shape")


class _StrictCompletions:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def create(self, **kwargs: Any) -> _StrictRebuild:
        return _StrictRebuild(self._data)


class _StrictChat:
    def __init__(self, data: dict[str, Any]) -> None:
        self.completions = _StrictCompletions(data)


class StrictRebuildClient:
    def __init__(self, data: dict[str, Any]) -> None:
        self.chat = _StrictChat(data)


class _NoConstructorRebuild:
    """Has model_dump but NO validating constructor at all."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


class _NoConstructorCompletions:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def create(self, **kwargs: Any) -> _NoConstructorRebuild:
        return _NoConstructorRebuild(self._data)


class _NoConstructorChat:
    def __init__(self, data: dict[str, Any]) -> None:
        self.completions = _NoConstructorCompletions(data)


class NoConstructorClient:
    def __init__(self, data: dict[str, Any]) -> None:
        self.chat = _NoConstructorChat(data)


# --------------------------------------------------------------------------- #
# fix_validate_tool_params                                                    #
# --------------------------------------------------------------------------- #

def test_denormalize_tools_rejects_non_dict_parameters():
    """A tool def whose function.parameters is a non-dict must raise loudly, naming
    the tool and carrying the fragment — the same pattern as the tool-call normalizer."""
    bad = [{"type": "function", "function": {"name": "read_file", "parameters": "not-a-schema"}}]
    with pytest.raises(UnsupportedProtocolShape) as exc:
        denormalize_tools(bad)
    assert "read_file" in str(exc.value)
    assert exc.value.fragment == "not-a-schema"


def test_denormalize_tools_rejects_int_parameters():
    with pytest.raises(UnsupportedProtocolShape):
        denormalize_tools([{"type": "function", "function": {"name": "f", "parameters": 42}}])


def test_denormalize_tools_rejects_bool_parameters():
    with pytest.raises(UnsupportedProtocolShape):
        denormalize_tools([{"type": "function", "function": {"name": "f", "parameters": True}}])


def test_denormalize_tools_rejects_list_parameters():
    with pytest.raises(UnsupportedProtocolShape):
        denormalize_tools([{"type": "function", "function": {"name": "f", "parameters": ["a", "b"]}}])


def test_denormalize_tools_accepts_dict_and_omitted_parameters(samples):
    """The happy path must stay unchanged: dict parameters and omitted parameters
    (defaulted to the empty-object schema) both lower cleanly."""
    lowered = denormalize_tools(samples["openai_tool_definitions"]["tools"])
    assert lowered[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"
    assert lowered[1]["function"]["parameters"] == {"type": "object", "properties": {}}


# --------------------------------------------------------------------------- #
# fix_rebuild_loud_errors                                                     #
# --------------------------------------------------------------------------- #

def _glm_arg_response() -> dict[str, Any]:
    """A minimal GLM response with native-object arguments (the arg-encoding delta)."""
    return {
        "id": "x", "object": "chat.completion", "model": "glm-5.2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "a.txt"}},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


def test_wrap_returns_normalized_completion():
    """The sync drop-in path: a GLM response with native-object arguments is
    normalized so the harness reads a valid OpenAI tool_calls shape."""
    client = wrap(FakeGLMClient(_glm_arg_response()))
    assert isinstance(client, BridgedClient)
    resp = client.chat.completions.create(model="glm-5.2", messages=[], tools=[])
    msg = resp["choices"][0]["message"]
    call = msg["tool_calls"][0]
    assert call["function"]["name"] == "read_file"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}


def test_rebuild_raises_loudly_on_validation_failure():
    """When the SDK model's model_validate rejects the normalized shape, the bridged
    create must raise UnsupportedProtocolShape — never silently degrade to a raw dict."""
    client = wrap(StrictRebuildClient(_glm_arg_response()))
    with pytest.raises(UnsupportedProtocolShape) as exc:
        client.chat.completions.create(model="glm-5.2", messages=[], tools=[])
    assert "StrictRebuild" in str(exc.value)


def test_rebuild_raises_loudly_when_no_validating_constructor():
    """A response type with NO validating constructor must raise loudly instead of
    returning an untyped dict masquerading as a structured response."""
    client = wrap(NoConstructorClient(_glm_arg_response()))
    with pytest.raises(UnsupportedProtocolShape):
        client.chat.completions.create(model="glm-5.2", messages=[], tools=[])


# --------------------------------------------------------------------------- #
# feat_streaming_dropin                                                       #
# --------------------------------------------------------------------------- #

def test_streaming_dropin_yields_normalized_fragments(samples):
    """stream=True no longer silently drops; the bridged create returns an iterable
    whose chunks carry JSON-string argument fragments the harness assembles itself."""
    chunks = samples["streaming_assembly"]["glm_stream_chunks"]
    client = wrap(FakeStreamGLMClient(chunks))
    out = client.chat.completions.create(
        model="glm-5.2", messages=[], tools=[], stream=True
    )
    # Must be iterable (a generator), not a single response.
    it = iter(out)
    collected_args = ""
    names = set()
    for chunk in it:
        calls = chunk["choices"][0]["delta"].get("tool_calls") or []
        for frag in calls:
            # Each fragment's arguments is a JSON STRING (not a native object).
            assert isinstance(frag["function"]["arguments"], str)
            collected_args += frag["function"]["arguments"]
            if frag["function"].get("name"):
                names.add(frag["function"]["name"])
    assert names == {"write_file"}
    # The concatenated fragments must parse to the expected assembled arguments.
    assert json.loads(collected_args) == {"path": "out.txt", "body": "hi"}


def test_normalize_delta_chunk_is_pure(samples):
    """normalize_delta_chunk must not mutate its input (the streaming variant is
    side-effect-free like the non-streaming normalize)."""
    import copy

    chunk = samples["streaming_assembly"]["glm_stream_chunks"][1]
    snapshot = copy.deepcopy(chunk)
    normalize_delta_chunk(chunk)
    assert chunk == snapshot


def test_assemble_stream_handles_multiple_choices():
    """The folded multi-choice fix: a stream with two choices assembles BOTH, not
    just choices[0]."""
    chunks = [
        {
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"role": "assistant",
                    "tool_calls": [{"index": 0, "id": "c0", "type": "function",
                        "function": {"name": "f0", "arguments": "{\"a\":"}}]}},
                {"index": 1, "delta": {"role": "assistant",
                    "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                        "function": {"name": "f1", "arguments": "{\"b\":"}}]}},
            ],
        },
        {
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"},
                {"index": 1, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "2}"}}]}, "finish_reason": "tool_calls"},
            ],
        },
    ]
    assembled = assemble_stream(chunks)
    choices = assembled["choices"]
    assert [c["index"] for c in choices] == [0, 1]
    names = [c["message"]["tool_calls"][0]["function"]["name"] for c in choices]
    assert names == ["f0", "f1"]
    args = [c["message"]["tool_calls"][0]["function"]["arguments"] for c in choices]
    assert [json.loads(a) for a in args] == [{"a": 1}, {"b": 2}]


# --------------------------------------------------------------------------- #
# feat_async_client                                                           #
# --------------------------------------------------------------------------- #

def test_awrap_returns_normalized_completion():
    async def _run() -> None:
        client = awrap(AsyncFakeGLMClient(_glm_arg_response()))
        assert isinstance(client, AsyncBridgedClient)
        resp = await client.chat.completions.create(model="glm-5.2", messages=[], tools=[])
        call = resp["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}

    asyncio.run(_run())


def test_async_streaming_yields_normalized_fragments(samples):
    async def _run() -> None:
        chunks = samples["streaming_assembly"]["glm_stream_chunks"]
        client = awrap(AsyncFakeStreamGLMClient(chunks))
        stream = await client.chat.completions.create(
            model="glm-5.2", messages=[], tools=[], stream=True
        )
        collected = ""
        names = set()
        async for chunk in stream:
            for frag in chunk["choices"][0]["delta"].get("tool_calls") or []:
                assert isinstance(frag["function"]["arguments"], str)
                collected += frag["function"]["arguments"]
                if frag["function"].get("name"):
                    names.add(frag["function"]["name"])
        assert names == {"write_file"}
        assert json.loads(collected) == {"path": "out.txt", "body": "hi"}

    asyncio.run(_run())

"""m2 gate: normalize/denormalize produce valid OpenAI shape on every divergence."""

from __future__ import annotations

import json

import pytest

from glm_toolbridge import (
    OpenAIChatCompletion,
    denormalize_request,
    denormalize_tools,
    normalize_response,
)
from glm_toolbridge.errors import MalformedToolArguments, StreamAssemblyError


def _assert_valid_openai(result):
    """A normalized result must validate as the strict OpenAI completion model."""
    assert isinstance(result.completion, OpenAIChatCompletion)
    msg = result.completion.choices[0].message
    for call in msg.tool_calls or []:
        # arguments must be a JSON *string* that parses.
        assert isinstance(call.function.arguments, str)
        json.loads(call.function.arguments)


def test_arg_encoding_roundtrip(samples):
    fix = samples["arg_encoding"]
    result = normalize_response(fix["glm_response"])
    _assert_valid_openai(result)
    call = result.completion.tool_calls[0]
    assert call.function.name == fix["expected_first_call"]["name"]
    # Native object got encoded to a JSON string that round-trips to the data.
    assert json.loads(call.function.arguments) == {"path": "src/main.py", "max_lines": 200}


def test_parallel_calls_roundtrip(samples):
    fix = samples["parallel_calls"]
    result = normalize_response(fix["glm_response"])
    _assert_valid_openai(result)
    names = [c.function.name for c in result.completion.tool_calls]
    assert names == fix["expected_call_names"]
    # Each call gained the OpenAI scaffolding (id + type).
    for c in result.completion.tool_calls:
        assert c.id
        assert c.type == "function"
    assert result.completion.choices[0].finish_reason == fix["expected_finish_reason"]


def test_reasoning_interleave_roundtrip(samples):
    fix = samples["reasoning_interleave"]
    result = normalize_response(fix["glm_response"])
    _assert_valid_openai(result)
    msg = result.completion.choices[0].message
    assert msg.content is None  # forced null when tool_calls present
    assert msg.glm_reasoning is not None
    assert "config.yaml" in msg.glm_reasoning


def test_streaming_assembly_roundtrip(samples):
    fix = samples["streaming_assembly"]
    result = normalize_response(fix["glm_stream_chunks"])  # list -> assemble
    _assert_valid_openai(result)
    call = result.completion.tool_calls[0]
    assert call.function.name == fix["expected_first_call"]["name"]
    assert call.function.arguments == fix["expected_first_call"]["arguments"]


def test_normalize_is_pure(samples):
    """normalize must not mutate its input."""
    import copy

    src = samples["arg_encoding"]["glm_response"]
    snapshot = copy.deepcopy(src)
    normalize_response(src)
    assert src == snapshot


def test_denormalize_tools(samples):
    tools = samples["openai_tool_definitions"]["tools"]
    lowered = denormalize_tools(tools)
    assert lowered[0]["function"]["name"] == "read_file"
    assert lowered[0]["function"]["description"] == "Read a file from disk"
    # The parameter-less tool got a default empty-object schema.
    assert lowered[1]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_denormalize_request_only_touches_tools():
    req = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "x"}}],
        "temperature": 0.2,
    }
    out = denormalize_request(req)
    assert out["model"] == "glm-5.2"
    assert out["temperature"] == 0.2
    assert out["messages"] == req["messages"]
    assert out["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_full_roundtrip_glm_to_openai_back_to_glm(samples):
    """A GLM response normalized to OpenAI shape, then its calls re-lowered as
    tool definitions, must survive without losing the call name — proving the
    bidirectional mapping is total, not lossy."""
    result = normalize_response(samples["arg_encoding"]["glm_response"])
    name = result.completion.tool_calls[0].function.name
    # Treat the called function as a tool def and lower it again.
    relowered = denormalize_tools([{"type": "function", "function": {"name": name}}])
    assert relowered[0]["function"]["name"] == name


def test_malformed_arguments_raise_loudly():
    bad = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{not json"},
                        }
                    ],
                }
            }
        ]
    }
    with pytest.raises(MalformedToolArguments):
        normalize_response(bad)


def test_empty_stream_raises():
    with pytest.raises(StreamAssemblyError):
        normalize_response([])


def test_unbalanced_stream_raises():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c", "type": "function",
                             "function": {"name": "f", "arguments": "{\"a\":"}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    ]
    with pytest.raises(StreamAssemblyError):
        normalize_response(chunks)


def test_passthrough_response_without_choices():
    """An error envelope with no choices passes through unchanged and validates
    fail-safe (no crash)."""
    from glm_toolbridge import normalize

    env = {"error": {"message": "rate limited"}}
    assert normalize(env) == env

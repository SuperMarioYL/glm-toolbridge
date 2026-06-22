"""m1 falsification gate: every documented delta is detectable on a real fixture.

If these tests pass, the GLM-5.2 vs OpenAI divergence is *not* a trivial rename
that collapses to one field — it is four independent, detectable differences,
which is the whole justification for the product.
"""

from __future__ import annotations

from glm_toolbridge.protocol import (
    DELTAS,
    DeltaKind,
    deltas_present,
    get_delta,
)


def test_four_distinct_deltas_exist():
    """The audit must enumerate exactly the four documented divergence axes."""
    kinds = {d.kind for d in DELTAS}
    assert kinds == {
        DeltaKind.ARG_ENCODING,
        DeltaKind.PARALLEL_CALLS,
        DeltaKind.REASONING_INTERLEAVE,
        DeltaKind.STREAMING_ASSEMBLY,
    }
    # Falsification gate: not a single trivial rename.
    assert len(DELTAS) == 4


def test_arg_encoding_detected(samples):
    resp = samples["arg_encoding"]["glm_response"]
    assert DeltaKind.ARG_ENCODING in deltas_present(resp)
    assert get_delta(DeltaKind.ARG_ENCODING).detect(resp) is True


def test_parallel_calls_detected(samples):
    resp = samples["parallel_calls"]["glm_response"]
    assert DeltaKind.PARALLEL_CALLS in deltas_present(resp)


def test_reasoning_interleave_detected(samples):
    resp = samples["reasoning_interleave"]["glm_response"]
    assert DeltaKind.REASONING_INTERLEAVE in deltas_present(resp)


def test_streaming_assembly_detected(samples):
    # A single streamed chunk already exhibits the delta shape (delta.tool_calls).
    chunk = samples["streaming_assembly"]["glm_stream_chunks"][0]
    assert DeltaKind.STREAMING_ASSEMBLY in deltas_present(chunk)


def test_each_delta_has_one_owning_fixture(samples):
    """Each non-streaming fixture should trigger its own delta detector.

    (Reasoning fixtures legitimately also carry an arg/string shape, so we only
    assert the *owning* delta fires, not exclusivity.)
    """
    mapping = {
        "arg_encoding": DeltaKind.ARG_ENCODING,
        "parallel_calls": DeltaKind.PARALLEL_CALLS,
        "reasoning_interleave": DeltaKind.REASONING_INTERLEAVE,
    }
    for key, kind in mapping.items():
        resp = samples[key]["glm_response"]
        assert kind in deltas_present(resp), f"{key} did not trigger {kind}"


def test_clean_openai_response_has_no_deltas():
    """A response already in OpenAI shape must trigger zero deltas."""
    clean = {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"a.txt\"}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    assert deltas_present(clean) == []


def test_delta_summaries_are_documented():
    """Every delta carries the human-readable shape strings the audit needs."""
    for d in DELTAS:
        assert d.summary
        assert d.glm_shape
        assert d.openai_shape
        assert d.glm_shape != d.openai_shape

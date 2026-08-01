# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-01

### Fixed
- **Loud rebuild on SDK re-validation failure.** `_rebuild_like` no longer
  swallows `model_validate` / `parse_obj` exceptions through an unvalidated
  `construct` or a bare-dict passthrough. A response that normalized but cannot
  re-validate into the SDK model now raises `UnsupportedProtocolShape` with the
  underlying error — honoring the library's "failures are loud, never silent"
  contract. The dict-template test path is unchanged.
- **`denormalize_tools` validates the parameters schema.** A tool definition
  whose `function.parameters` is present but not a dict now raises
  `UnsupportedProtocolShape` (naming the tool, carrying the fragment) instead of
  forwarding a malformed schema to GLM. `None` still defaults to the empty-object
  schema.

### Added
- **Streaming drop-in (`stream=True`).** `wrap()` no longer silently drops
  `stream` and forces a complete response. When `stream=True`, the bridged
  `create` returns an iterator of normalized delta chunks — argument *fragments*
  coerced to JSON strings without per-fragment validation (OpenAI streaming
  fragments are partial by contract), reasoning relocated out of the delta, and
  any parallel framing flattened — so a streaming harness assembles tool calls
  exactly as it would for an OpenAI stream. `assemble_stream` also now assembles
  every choice (n>1), not just `choices[0]`.
- **Async client (`awrap`).** `awrap(client)` mirrors `wrap` for `AsyncOpenAI`
  clients: `await client.chat.completions.create(...)` returns a normalized
  completion, and `stream=True` returns an async iterator of normalized chunks.
  Exposed as `awrap` / `AsyncBridgedClient`. No new third-party deps (`asyncio`
  is stdlib).
- **`normalize_delta_chunk`** public pure transform (the per-chunk streaming
  normalizer) and the first unit tests for `client.py` (`tests/test_client_wrap.py`).

### Changed
- Version bumped to 0.2.0 in `pyproject.toml` and `glm_toolbridge.__version__`.

## [0.1.0] - 2026-06-22

### Added
- **Protocol audit (m1).** `docs/PROTOCOL_DELTAS.md` documents the four concrete
  GLM-5.2 vs OpenAI tool-call divergences — argument encoding, parallel-call
  framing, reasoning interleave, and streaming assembly — each with a captured
  fixture and an executable detector in `protocol.py`.
- **Bidirectional adapter (m2).** `normalize_response()` converts a GLM-5.2
  response (or streamed chunk list) into a validated OpenAI `tool_calls` shape;
  `denormalize_request()` lowers OpenAI-shaped tool definitions into the request
  GLM accepts. Covered by roundtrip tests on every divergence case.
- **Drop-in client wrapper (m3).** `wrap(client)` returns a transparent proxy
  over an OpenAI-SDK client that reconciles the protocol on each
  `chat.completions.create` call — the harness's OpenAI code path is unchanged.
- **Runnable demo.** `examples/openai_harness_demo.py` shows a stock OpenAI tool
  loop failing against raw GLM-5.2, then succeeding through one-line `wrap`.
- Typed Pydantic models (`OpenAIChatCompletion`, `ToolCall`) and named errors
  (`UnsupportedProtocolShape`, `MalformedToolArguments`, `StreamAssemblyError`)
  so failures are loud, never silent.

[0.2.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.2.0
[0.1.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.1.0

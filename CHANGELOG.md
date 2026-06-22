# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.1.0

# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-09-07

### Fixed
- **Relabel the reassembled stream response as `chat.completion`, not
  `chat.completion.chunk`.** `assemble_stream` seeded the reassembled head
  from `copy.deepcopy(chunks[0])`, so `head["object"]` inherited the streamed
  chunk value `"chat.completion.chunk"`. The relabel
  `head["object"] = head.get("object") or "chat.completion"` could not fire
  (the chunk value is truthy), so the reassembled response — structurally a
  non-streaming `chat.completion` (it carries `message`, not `delta`) — still
  advertised itself as a streaming chunk. A harness switching on `object` to
  route chunk-vs-completion handling would misroute it, and re-validation
  against the OpenAI SDK's `Literal["chat.completion"]` would reject it. The
  object field is now set to `"chat.completion"` unconditionally on reassembly.
- **Detect reasoning interleave for `parallel_tool_calls`-only spilled
  content.** `_detect_reasoning_interleave` flagged spilled content only when
  a flat `tool_calls` array was present, but the v0.3.0 `_split_reasoning` fix
  expanded the normalizer to also relocate spilled content for a
  `parallel_tool_calls`-only envelope. Such an envelope therefore had its
  prose relocated into `_glm_reasoning` (the reasoning_interleave transform
  ran) yet `deltas_applied` omitted `REASONING_INTERLEAVE` — the audit
  under-reported a divergence the library actually fixed. The detector's
  spilled-content check now uses the same `has_calls` condition the normalizer
  uses (`tool_calls` OR `parallel_tool_calls`), so the audit stays honest.

### Changed
- Version bumped to 0.5.0 in `pyproject.toml`, `VERSION`, and
  `glm_toolbridge.__version__`. Added `content_version` to `web/site.json`.

## [0.4.0] - 2026-08-28

### Fixed
- **Coerce streamed argument fragments before concatenating in `assemble_stream`.**
  `assemble_stream` concatenated streamed tool-call argument fragments with
  `slot["function"]["arguments"] += fn["arguments"]`, assuming every fragment's
  arguments was a string. GLM's documented `arg_encoding` delta may return
  `function.arguments` as a native JSON object; such a fragment raised an
  unhandled `TypeError` (str += dict), crashing stream reassembly. The
  per-chunk path `normalize_delta_chunk` already coerced fragments via
  `_coerce_arguments_fragment`; `assemble_stream` now does the same before
  concatenating.
- **Treat empty-string tool-call arguments as no-args, not malformed.**
  `_coerce_arguments_to_json_string` rejected an empty-string `arguments`
  (`""`) with `MalformedToolArguments`, while the sibling `assemble_stream`
  normalized empty assembled arguments to `"{}"`. A non-streaming tool call
  whose `function.arguments` was `""` (a no-arg convention the OpenAI SDK
  accepts as a plain `str`) therefore crashed the bridge. Empty-string
  arguments now normalize to `"{}"`, matching `assemble_stream`; non-empty
  non-JSON strings still raise.

### Changed
- Version bumped to 0.4.0 in `pyproject.toml`, `VERSION`, and
  `glm_toolbridge.__version__`.

## [0.3.0] - 2026-08-22

### Fixed
- **Relocate spilled content for `parallel_tool_calls`-only messages.**
  `_split_reasoning` gated content relocation on `message.tool_calls` only,
  while `normalize()` computes `has_calls` from both `tool_calls` AND
  `parallel_tool_calls` and then nulls `content`. A `parallel_tool_calls`-only
  envelope with non-null content therefore had its spilled prose silently
  dropped to `None` instead of relocated into `_glm_reasoning`. The gate now
  uses the same `has_calls` condition `normalize()` uses, honoring the
  "relocate rather than drop" contract.
- **Preserve `usage` and late top-level fields when reassembling a stream.**
  `assemble_stream` seeded the reassembled head from `copy.deepcopy(chunks[0])`
  and only replaced `choices`, so any top-level field arriving exclusively in a
  later chunk was lost — most commonly `usage` token stats, which per the
  OpenAI streaming contract (and `stream_options.include_usage`) arrive in the
  FINAL chunk with an empty `choices` list. The reassembled response now merges
  late non-choices top-level fields that `chunks[0]` lacked, so token/cost
  accounting survives stream reassembly.

### Changed
- Version bumped to 0.3.0 in `pyproject.toml`, `VERSION`, and
  `glm_toolbridge.__version__`.

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

[0.5.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.5.0
[0.4.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.4.0
[0.3.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.3.0
[0.2.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.2.0
[0.1.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.1.0

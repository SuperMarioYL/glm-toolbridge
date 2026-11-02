# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-11-03

### Fixed
- **Raise named errors for malformed nested response envelopes.**
  `normalize()` assumed every `choices` entry was an object, and the four
  detectors in `protocol.py` assumed `choices[0]` and its `message` were
  objects. A response carrying a non-object choice (or a non-object
  message/call) therefore leaked a raw `AttributeError: 'str' object has no
  attribute 'get'` out of `deltas_present`/`normalize` instead of a named
  `ToolBridgeError` — the same crash-on-bad-input class v0.5.0 and v0.6.0
  removed one level up. Detectors now return `False` for shapes they cannot
  recognize (their documented contract), and `normalize()` raises
  `UnsupportedProtocolShape` with the index and offending fragment for a
  present non-list `choices` or a non-object choice. Absent/null `choices`
  and an empty list keep their documented pass-through.
- **Reject a non-object streamed `delta` with `StreamAssemblyError`.**
  `assemble_stream` used `choice.get("delta") or {}` and then called
  `delta.get(...)`, so a truthy non-object delta (a string or list from a
  malformed provider frame) leaked a raw `AttributeError`. It now treats an
  absent/null delta as the empty default and raises `StreamAssemblyError`
  naming the choice index for a present non-object delta, matching the
  function's existing named-error treatment of non-object chunks and choices.
- **Stop dropping the streamed `parallel_tool_calls` envelope.**
  `assemble_stream` read only `delta["tool_calls"]`, so a streamed chunk that
  framed its calls under GLM's documented `parallel_tool_calls` envelope lost
  every call silently (the reassembled turn came back with no `tool_calls` and
  `finish_reason: "stop"`). It now merges a list-valued envelope into the
  fragment stream, and both streaming paths raise a named error for a
  present-but-non-list envelope instead of discarding it. The streaming
  detector also recognizes the envelope, so `deltas_applied` records the
  divergence.
- **Remove documentation for interfaces that never shipped.** Both README
  locales and the landing-page copy told readers that "the CLI exposes the
  following operations", but the package declares no `[project.scripts]` entry
  point and installs no console script. The docs now state that the library
  ships no CLI and present the bundled example as the reference usage. The
  `client.py` comment promising `wrap(..., base_url=...)` is gone too:
  `wrap()` takes only the client, and `base_url` belongs on the underlying
  client.

### Changed
- Version bumped to 0.7.0 in `pyproject.toml`, `VERSION`,
  `glm_toolbridge.__version__`, `web/site.json` `content_version`, `uv.lock`,
  and this changelog.
- Added the missing `[0.6.0]` link reference and a `[0.7.0]` reference.
  `tests/test_version_consistency.py` now also asserts that the editable
  `uv.lock` project version matches `VERSION` and that every version heading
  has a link reference; both assertions fail on the unmodified v0.6.0 tree.

## [0.6.0] - 2026-09-22

### Fixed
- **Raise a named error when the tool-call envelope is present but not a
  list.** `_collect_calls` only read envelopes that passed
  `isinstance(..., list)`, while `normalize()` treated any truthy
  `tool_calls`/`parallel_tool_calls` value as "has calls": a
  present-but-non-list envelope (e.g. a bare call object) was silently
  replaced with an empty `tool_calls` array and `finish_reason` forced to
  `"tool_calls"` — the call data was discarded without a sound. Such an
  envelope now raises `UnsupportedProtocolShape` naming the field and
  carrying the offending fragment; list envelopes and explicit nulls behave
  exactly as before.
- **Reject non-dict stream chunks with `StreamAssemblyError`.**
  `assemble_stream` called `chunk.get(...)` unguarded, so a chunk list
  containing a non-dict (a `None` from a provider hiccup, a stray string)
  leaked a raw `AttributeError` that escapes the `ToolBridgeError` hierarchy.
  Chunks are now validated up front and a malformed one raises
  `StreamAssemblyError` naming the offending index — the same named-error
  treatment `normalize` and `normalize_delta_chunk` already apply to their
  inputs.
- **Surface typed re-validation failures as `UnsupportedProtocolShape`.**
  `normalize_response` ended in an unguarded
  `OpenAIChatCompletion.model_validate(...)`, so a shape the pure transform
  passes through but the typed model rejects (e.g. `message.content` as a
  list of content parts) leaked a raw pydantic `ValidationError` — invisible
  to a caller following the documented `except ToolBridgeError` contract.
  The adapter's gate now wraps the failure in `UnsupportedProtocolShape`,
  mirroring `client._rebuild_like`.

### Changed
- Version bumped to 0.6.0 in `pyproject.toml`, `VERSION`,
  `glm_toolbridge.__version__`, `web/site.json` `content_version`, and this
  changelog.

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

[0.7.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.7.0
[0.6.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.6.0
[0.5.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.5.0
[0.4.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.4.0
[0.3.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.3.0
[0.2.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.2.0
[0.1.0]: https://github.com/SuperMarioYL/glm-toolbridge/releases/tag/v0.1.0

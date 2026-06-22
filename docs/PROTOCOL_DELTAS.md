# GLM-5.2 vs OpenAI tool-call protocol deltas

This is the m1 audit: the exact, captured differences between GLM-5.2 (智谱)
function-calling output and the canonical OpenAI `tool_calls` shape that
coding-agent harnesses hardcode. Each delta below is paired with a fixture in
[`tests/fixtures/glm_tool_call_samples.json`](../tests/fixtures/glm_tool_call_samples.json)
and an executable detector in [`src/glm_toolbridge/protocol.py`](../src/glm_toolbridge/protocol.py).

**Falsification gate.** If these collapsed to a single trivial field rename,
this would be a gist, not a library. They do not — there are four independent,
detectable divergences across argument encoding, parallel-call framing,
reasoning interleave, and streaming assembly. `tests/test_protocol_deltas.py`
asserts each one fires on its fixture.

For reference, the OpenAI shape a harness expects is:

```jsonc
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,                          // null when tool_calls is set
      "tool_calls": [{
        "id": "call_abc",
        "type": "function",
        "function": {
          "name": "read_file",
          "arguments": "{\"path\": \"a.txt\"}"  // a JSON-encoded STRING
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

---

## Delta 1 — argument encoding (`arg_encoding`)

**GLM-5.2:** `function.arguments` may be a **native JSON object** (or, less
often, a non-JSON string).
**OpenAI:** `function.arguments` is **always a JSON-encoded string**.

```jsonc
// GLM-5.2
"function": { "name": "read_file", "arguments": { "path": "src/main.py", "max_lines": 200 } }

// OpenAI (what the harness calls json.loads() on)
"function": { "name": "read_file", "arguments": "{\"path\":\"src/main.py\",\"max_lines\":200}" }
```

**Why it breaks harnesses:** the harness does `json.loads(call.function.arguments)`.
When `arguments` is already a dict, `json.loads` raises `TypeError` — the tool
loop stalls with no useful message.

**Fix:** `normalize._coerce_arguments_to_json_string` re-encodes native objects
to a compact JSON string and validates pre-existing strings, raising
`MalformedToolArguments` if neither path yields valid JSON.

---

## Delta 2 — parallel-call framing (`parallel_calls`)

**GLM-5.2:** multiple simultaneous calls are framed either under a
`message.parallel_tool_calls` envelope **or** as `tool_calls` entries that omit
the per-call `id` / `type` scaffolding.
**OpenAI:** N parallel calls are N entries in one flat `message.tool_calls`
array, each with a stable `id` and `"type": "function"`.

```jsonc
// GLM-5.2
"message": {
  "parallel_tool_calls": [
    { "function": { "name": "list_dir", "arguments": { "path": "." } } },
    { "function": { "name": "grep",     "arguments": { "pattern": "TODO" } } }
  ]
}

// OpenAI
"message": {
  "tool_calls": [
    { "id": "call_1", "type": "function", "function": { "name": "list_dir", "arguments": "{\"path\":\".\"}" } },
    { "id": "call_2", "type": "function", "function": { "name": "grep",     "arguments": "{\"pattern\":\"TODO\"}" } }
  ]
}
```

**Why it breaks harnesses:** the harness iterates `message.tool_calls`, which is
absent (calls hid under the envelope) or whose entries lack the `id` it needs to
correlate the eventual `tool` result message.

**Fix:** `normalize._collect_calls` flattens both framings into one OpenAI
`tool_calls` array and synthesizes a stable `id` (`call_<uuid>`) where GLM
omitted one. `finish_reason` is corrected to `"tool_calls"`.

---

## Delta 3 — reasoning interleave (`reasoning_interleave`)

**GLM-5.2:** carries a chain-of-thought trace on `message.reasoning_content`
alongside the tool call, and sometimes leaves prose in `message.content` even
when a call is present.
**OpenAI:** a tool-call turn has `content == null` and no reasoning field.

```jsonc
// GLM-5.2
"message": {
  "content": "Let me check the config first.",
  "reasoning_content": "DB url lives in config.yaml, so read it.",
  "tool_calls": [ ... ]
}

// OpenAI
"message": { "content": null, "tool_calls": [ ... ] }
```

**Why it breaks harnesses:** some harnesses treat a non-null `content` on a
tool-call turn as a final assistant message and short-circuit the loop, never
dispatching the tool. The unexpected `reasoning_content` key also trips strict
schema validators.

**Fix:** `normalize._split_reasoning` forces `content` to `null` on tool-call
turns and relocates the reasoning to `message._glm_reasoning` (a side channel)
so it is preserved for callers that want it but invisible to the standard path.

---

## Delta 4 — streaming assembly (`streaming_assembly`)

**GLM-5.2:** streamed tool calls arrive as `choices[].delta` fragments keyed by
`index`, with `function.arguments` delivered as incremental string fragments
that must be concatenated.
**OpenAI:** the SDK's helpers reassemble streamed deltas, but harnesses that
read GLM's stream directly must reassemble the fragments themselves — and GLM's
fragment boundaries can split a JSON token mid-string.

```jsonc
// GLM-5.2 streamed chunks (abridged)
{ "delta": { "tool_calls": [ { "index": 0, "id": "c1", "function": { "name": "write_file", "arguments": "" } } ] } }
{ "delta": { "tool_calls": [ { "index": 0, "function": { "arguments": "{\"path\": \"out" } } ] } }
{ "delta": { "tool_calls": [ { "index": 0, "function": { "arguments": ".txt\"}" } } ] }, "finish_reason": "tool_calls" }

// Reassembled (OpenAI message shape)
"tool_calls": [ { "id": "c1", "type": "function", "function": { "name": "write_file", "arguments": "{\"path\": \"out.txt\"}" } } ]
```

**Why it breaks harnesses:** naive concatenation per chunk without index
bookkeeping interleaves arguments from parallel calls, and a stream that ends
mid-arguments yields invalid JSON the harness silently swallows.

**Fix:** `normalize.assemble_stream` accumulates fragments by `index`,
concatenates `arguments`, and validates the final JSON — raising
`StreamAssemblyError` on an unindexed fragment, a never-named call, or unbalanced
JSON at stream end.

---

## Summary

| Delta | GLM-5.2 | OpenAI | Detector | Transform |
|---|---|---|---|---|
| arg_encoding | native object args | JSON string args | `_detect_arg_encoding` | `_coerce_arguments_to_json_string` |
| parallel_calls | envelope / no id+type | flat array w/ id+type | `_detect_parallel_calls` | `_collect_calls` |
| reasoning_interleave | `reasoning_content` + content | `content: null` | `_detect_reasoning_interleave` | `_split_reasoning` |
| streaming_assembly | indexed delta fragments | reassembled message | `_detect_streaming_assembly` | `assemble_stream` |

All four are exercised end-to-end in `tests/test_adapter_roundtrip.py`.

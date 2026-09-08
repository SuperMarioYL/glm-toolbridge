[简体中文](./README.md) · [Website](https://glm-toolbridge.lei6393.com/) · [GitHub](https://github.com/SuperMarioYL/glm-toolbridge)

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/hero-dark.svg">
  <img src="./assets/presentation/hero-light.svg" width="960" alt="Hero diagram">
</picture>

# glm-toolbridge

**Make tool-call arguments match your harness.**

glm-toolbridge converts supported GLM-style tool-call shapes into the OpenAI-compatible structures expected by a Python harness.

## Why use it

A tool dispatcher may expect arguments as a JSON string while a response contains an object. Keep that protocol conversion in one adapter so the tool dispatcher can continue using its established input contract.

- **One protocol boundary** — Conversion stays outside your tool dispatch code.
- **Multiple calling modes** — Sync, async and stream adapters share normalization.
- **Explicit failures** — Unsupported shapes produce named errors.

## Architecture

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-dark.svg">
  <img src="./assets/presentation/architecture-light.svg" width="960" alt="Architecture diagram">
</picture>

Request denormalization prepares tool definitions. Response normalization handles supported argument encoding, parallel-call and reasoning differences. Client wrappers apply the transformations to sync, async and streamed results, while explicit errors identify unsupported shapes.

| Component | Responsibility |
| --- | --- |
| `Client wrapper` | src/glm_toolbridge/client.py |
| `Request adapter` | adapter.py |
| `Normalize / assemble` | normalize.py |
| `Typed result` | OpenAI-compatible response |

## Install and quickstart

Build with the version declared in the repository manifest. Run the example from the repository root.

```bash
git clone https://github.com/SuperMarioYL/glm-toolbridge.git
cd glm-toolbridge
uv venv .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

The included Python script defines a complete response with object arguments and prints the normalized message.

```bash
.venv/bin/python examples/presentation-demo.py
```

## Recorded demo

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/process-dark.svg">
  <img src="./assets/presentation/process-light.svg" width="960" alt="Process diagram">
</picture>

The weather call arguments become a JSON string while the call ID and function name remain available.

```text
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call-1",
      "type": "function",
      "function": {
        "name": "weather",
        "arguments": "{\"city\":\"Beijing\"}"
      }
    }
  ],
  "content": null,
  "_glm_reasoning": "Checking the city"
}
```

The complete command and output are recorded in [docs/demo-results.json](./docs/demo-results.json). Inputs and reproduction code are included in the repository.

![Existing terminal recording](./assets/demo.gif)

The existing recording is retained for context; the text example above documents the reproducible scenario.

## Usage

The CLI exposes the following operations. Commands after the example use your own paths or identifiers.

```bash
# Offline before/after harness example:
.venv/bin/python examples/openai_harness_demo.py
```

## Configuration

Use wrap(client) for a synchronous OpenAI client and awrap(client) for an async client. Configure the endpoint, model and credentials on the underlying client. Pure normalize_response and denormalize_request functions need no key.

```python
from glm_toolbridge import normalize_response
raw = {"choices": [{"index": 0, "message": {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": {"city": "Beijing"}}}]}, "finish_reason": "tool_calls"}]}
normalized = normalize_response(raw).as_openai_dict()
```

## Integrations and responsibilities

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-dark.svg">
  <img src="./assets/presentation/integrations-light.svg" width="960" alt="Integrations diagram">
</picture>

The following routes are implemented in the source. Choose the input that matches your task and keep the resulting artifact with your project.

| Route | Implemented role |
| --- | --- |
| wrap / awrap | Synchronous and asynchronous clients |
| Response dictionaries | Pure normalization functions |
| Stream chunks | Delta normalization and assembly |
| Protocol errors | Named unsupported-shape failures |

## Limits and next steps

- The demo uses a synthetic response. It does not establish the behavior of any current GLM deployment or compatibility with every agent harness.
- Only the supported OpenAI-style contract is adapted; Anthropic Messages is outside the current adapter.

Expand the supported delta set only from reproducible response examples. Additional model protocols and Anthropic Messages remain future work.

## License and contributions

See [LICENSE](./LICENSE). When reporting an issue, include a minimal input, the command, and the observed output.

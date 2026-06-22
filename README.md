<div align="right"><sub><a href="./README.en.md">English</a>&nbsp;&nbsp;⇄&nbsp;&nbsp;<b>简体中文</b></sub></div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
  <img src="./assets/hero-light.svg" width="880" alt="glm-toolbridge — 让 GLM-5.2 的 tool-call 在 OpenAI 格式编码 agent 里正确解析的协议适配层">
</picture>

<p><sub>把 GLM-5.2 接进任意 OpenAI 格式编码 agent，tool-call 不再静默错配——一层薄协议适配，harness 的 OpenAI 代码路径一行不改。</sub></p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/SuperMarioYL/glm-toolbridge/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/glm-toolbridge" alt="Release"></a>
  <a href="https://github.com/SuperMarioYL/glm-toolbridge/actions/workflows/ci.yml"><img src="https://github.com/SuperMarioYL/glm-toolbridge/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-3776AB.svg" alt="Python">
  <img src="https://img.shields.io/badge/GLM--5.2-ready-10A37F.svg" alt="GLM-5.2 ready">
  <img src="https://img.shields.io/badge/Coding%20Agent-drop--in-5E5CE6.svg" alt="Coding Agent drop-in">
</p>

**把 GLM-5.2 接到硬编码 OpenAI `tool_calls` 解析的编码 agent 上，tool-call 会静默错配、循环卡死；`glm-toolbridge` 用一行 `wrap()` 在中间补一层协议适配，让那条循环直接跑通。**

国内开发者越来越多地把 GLM-5.2（智谱）当作 **Claude Code** 式 harness 或 **Cursor** 这类编码 agent 的后端模型——可这些 harness 默认对端说的是 OpenAI/Anthropic 的 function-call 协议。GLM-5.2 的 tool-call 在四个地方和 OpenAI 形状不一样（参数编码、并行调用、推理交织、流式拼装），于是 `json.loads(arguments)` 抛异常、`content` 非空被当成终态、并行调用对不上 id——而且大多是**静默**失败。像 [farion1231/cc-switch](https://github.com/farion1231/cc-switch) 这样"在同一个 harness 后面换模型后端"的工具正把这种用法变成日常，错位也就发生在换上 GLM 的那一刻。`glm-toolbridge` 把这层差异收敛进一个透明代理：你的 OpenAI 代码路径一行不改，GLM 的响应回到 harness 时已经是合法的 OpenAI `tool_calls`。

---

<h2><img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 架构</h2>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
  <img src="./assets/atlas-light.svg" width="880" alt="数据流：OpenAI 格式 harness → glm-toolbridge 适配层 → GLM-5.2，去程 denormalize，回程 normalize">
</picture>

单进程 Python 库，没有服务、没有守护进程。`wrap(client)` 返回一个和原 client 接口完全一致的透明代理：

- **去程** `denormalize_request(tools)`——把 OpenAI 形状的 tool 定义降格成 GLM-5.2 接受的请求体；
- **回程** `normalize_response()`——把 GLM 形状的响应还原成 harness 期望的 OpenAI `tool_calls`；
- 遇到任何已知 delta 都覆盖不了的 shape，抛出**有名字的显式错误**（`UnsupportedProtocolShape` / `MalformedToolArguments` / `StreamAssemblyError`），而不是返回半成品让 harness 三帧之后才崩——和今天的"静默错配"正好相反。

<h2><img src="https://api.iconify.design/tabler:download.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 安装</h2>

```bash
uv add glm-toolbridge        # 或者：pip install glm-toolbridge
```

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 快速开始</h2>

冷启动到第一个可见结果，三条命令：

```bash
git clone https://github.com/SuperMarioYL/glm-toolbridge && cd glm-toolbridge
uv sync
uv run python examples/openai_harness_demo.py
```

<details>
<summary>示例输出</summary>

```text
================================================================
  glm-toolbridge demo — same OpenAI harness, GLM-5.2 backend
================================================================

[LEFT] stock harness against raw GLM-5.2 ...
  ✗ tool loop broke: TypeError: the JSON object must be str, bytes or bytearray, not dict
    (GLM sent arguments as a native object; json.loads chokes — the silent breakage devs hit today.)

[RIGHT] same harness, one-line wrap: client = wrap(client) ...
  ✓ tool loop completed: Beijing: 21 celsius, clear
    (arguments normalized to a JSON string, content forced null, reasoning relocated — the harness never knew GLM was behind it.)
```

</details>

<h2><img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 用法</h2>

`glm-toolbridge` 给了三个抽象层级，按需取用。完整可运行示例见 [`examples/openai_harness_demo.py`](examples/openai_harness_demo.py)。

### 1. 一行接入（推荐）

把现有的 OpenAI-SDK client 包一层，之后的代码一行不动：

```python
from openai import OpenAI
from glm_toolbridge import wrap, GLM_DEFAULT_BASE_URL

client = wrap(OpenAI(base_url=GLM_DEFAULT_BASE_URL, api_key="你的智谱-key"))

resp = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=[...],   # 你原本的 OpenAI 形状 tool 定义
)
resp.choices[0].message.tool_calls   # 已是合法的 OpenAI 形状
```

### 2. 纯函数转换（不想包 client 时）

直接对 wire-shape 字典做转换，适合自己掌控请求/响应循环的 harness：

```python
from glm_toolbridge import normalize_response, denormalize_request

glm_kwargs = denormalize_request(openai_request)   # OpenAI tool 定义 → GLM 请求体
result = normalize_response(glm_raw_response)       # GLM 响应 → OpenAI 形状
result.completion.tool_calls          # 经 pydantic 校验的类型化视图
result.as_openai_dict()               # harness 直接消费的普通 dict
result.deltas_applied                 # 这次实际命中了哪几条 delta
```

流式响应先把分片列表交给 `assemble_stream()`（或直接把 chunk 列表传给 `normalize_response()`）拼装成完整调用，再走上面的转换。

### 3. 协议审计（看清到底差在哪）

四条 delta 都是可执行的检测器，可单独查询：

```python
from glm_toolbridge import DELTAS, deltas_present

for d in DELTAS:
    print(d.kind.value, "—", d.summary)

deltas_present(glm_raw_response)   # → [DeltaKind.ARG_ENCODING, DeltaKind.REASONING_INTERLEAVE, ...]
```

完整的差异对照表见 [`docs/PROTOCOL_DELTAS.md`](docs/PROTOCOL_DELTAS.md)。

<h2><img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 演示</h2>

![demo](assets/demo.gif)

同一个 OpenAI 格式 harness：左边直连 GLM-5.2，tool 循环静默卡死；右边只多了一行 `wrap()`，同一个 tool-call 解析成功、循环跑完。

<h2><img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 路线图</h2>

- [x] **m1 协议审计**——把 GLM-5.2 vs OpenAI 的四条 tool-call 差异连同捕获的 fixture 落进 `docs/PROTOCOL_DELTAS.md`，每条都有可执行检测器
- [x] **m2 双向适配器**——`normalize()` / `denormalize()` 在四条差异的 fixture 上通过 roundtrip 测试
- [x] **m3 drop-in 包装器**——`wrap()` 透明适配 OpenAI-SDK client，`examples/` 跑通"未接入失败 / 接入成功"
- [ ] 覆盖更多 GLM-5.2 tool-call 边界场景（按真实 issue 反馈补 delta）
- [ ] Anthropic Messages 格式适配（当前仅 OpenAI `tool_calls`）
- [ ] 视需求决定是否扩展到其他国产模型协议——深度优先于广度

> 不在 v0.1 范围内：Web UI / dashboard、其他模型（Qwen / Kimi / DeepSeek / 豆包 / MiniMax）的适配、自带的编码 agent、托管服务 / 计费、微调。

<h2><img src="https://api.iconify.design/tabler:license.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 许可与贡献</h2>

MIT 许可证，详见 [LICENSE](./LICENSE)。欢迎提交 issue 或 PR——尤其是你撞见了某个当前 delta 没覆盖的 GLM-5.2 tool-call 形状，请把那段响应贴进 issue，我们会补一条 delta。

---

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>

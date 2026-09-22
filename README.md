# jev-decide

[![CI](https://github.com/OwenZhao9/jev-decide/actions/workflows/ci.yml/badge.svg)](https://github.com/OwenZhao9/jev-decide/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## 1. 一句话定位

把「让大模型写一段话再解析」换成「问一个带类型的问题，直接拿到**选择 + 概率 + 置信度**」，
并且在置信度不够的时候**什么都不做**。

## 2. 为什么存在

大模型天生是写给人看的（文本生成，text generation）。可一旦是程序要用这个判断，
中间就多了一道又脆又慢的手续：先让它写一段话，再用正则和 `json.loads` 去猜它的意思。
写法变了、语气变了、多加了一句「当然，具体还要看……」，解析就崩了。

更糟的是**没有拒答的余地**。模型不确定的时候，它照样会给你一个答案，而你的代码分不出
「它很确定选 A」和「A 和 B 它各摇摆一半」。于是不确定的猜测被当成确定的指令执行下去——
在一个会动的机器上，这就是事故。

这个库做两件事：

1. **问一个有类型的问题，拿一个有类型的答案。** `choice()` 返回你给的选项之一，
   外带一张归一化的概率表；`score()` 返回一个落在 `[lo, hi]` 里的数。没有解析，没有正则。
2. **给不确定留一个出口。** 每个答案都带 `confidence`（0..1），`Decider.gate()`
   在置信度不足时**保持现状**而不是照做。学名叫**选择性预测**
   （selective prediction，用覆盖率换风险，见第 9 节）。

还有第三件事，是工程上的现实：**它永远会返回**。没有 API key、网断了、对面 529 了、
超时了——都降级到你自己写的那个纯函数，并且在 `degraded=True` 和 `note` 里说清楚发生了什么。
不会抛异常，不会挂在那里。

> [!WARNING]
> **`Decider` 的所有方法都会阻塞**，最长 `2 × timeout_s`（默认 4 秒）。
> **禁止在实时控制环里调用。** 放到任务边界上，或者丢给一个工作线程，
> 让控制环只读最近一次的结果。

> [!WARNING]
> **实例不是线程安全的**，而且内部**故意不加锁**（加锁会拖慢热路径）。
> 一个实例只在一个线程里用；要并发就一个线程一个实例。

## 3. 安装

```bash
uv add "jev-decide @ git+https://github.com/OwenZhao9/jev-decide@v0.1.0"
```

或者：

```bash
pip install "jev-decide @ git+https://github.com/OwenZhao9/jev-decide@v0.1.0"
```

需要 Python >= 3.11。**没有任何运行期依赖**——HTTP 走标准库 `urllib`。

## 4. 60 秒上手

不需要 API key 也能跑通，直接复制运行：

```python
from jev_decide import Decider

# 你自己的兜底逻辑：一个纯函数，零依赖、永远可用
def house_rules(state: dict) -> str:
    if state["battery_pct"] < 20 or state["tilt_deg"] > 40:
        return "off"
    return "low" if state["tilt_deg"] > 20 else "high"

decider = Decider("auto", timeout_s=1.5, on_decision=lambda d: print(d.to_dict()))
state = {"battery_pct": 62, "tilt_deg": 31.0, "gait": "stairs_up"}

c = decider.choice(state, "这个状态下助力档位给多少？", ["off", "low", "high"], rules=house_rules)
print(c.value, c.confidence, c.backend, c.degraded)
# low 1.0 rules True     <- 没配 key，降级到 rules，degraded 如实标出来

# 关键的一步：不够确定就别动
action = Decider.gate(c, min_confidence=0.7, on_low="keep", current="off")
print(action)   # 置信度够 -> "low"；不够 -> 保持 "off"
```

配上 key，同一段代码不改一个字就开始用模型：

```bash
export TYPESAFE_API_KEY=sk-...     # 走 jev 后端（TypeSafe System One）
# 或者
export OPENAI_API_KEY=sk-...       # 走 llm 后端（任意 OpenAI 兼容端点）
```

完整可运行版本见 [`examples/quickstart.py`](./examples/quickstart.py)。

## 5. API 参考

导出的全部公开符号：`Choice`、`Score`、`Decider`、`JevDecideError`、`ConfigError`。

单位约定：`*_ms` 是毫秒，`*_s` 是秒，`confidence` 和 `probs` 的值都在 0..1，
`probs` 的所有值加起来等于 1。

### `Choice`

```python
@dataclass(frozen=True)
class Choice:
    value: str                      # 选中的那个选项，必定是你传进来的 options 之一
    probs: dict[str, float]         # 每个选项的概率，按 options 顺序排列，和为 1
    confidence: float               # 0..1，分布有多尖；不是 value 的概率
    latency_ms: float               # 整次调用的墙上时间（含降级重试），毫秒
    backend: str                    # 实际给出答案的后端："jev" | "llm" | "rules"
    degraded: bool = False          # 是不是降级来的
    note: str = ""                  # 降级原因，人能读懂
    raw: dict | None = None         # 后端原始返回，留给审计；rules 后端为 None

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Choice": ...
```

`to_dict()` 出来的就是纯 JSON，不用写 encoder，`from_dict()` 能原样还原。
`probs` 在构造时会被复制一份，改外面那个 dict 影响不到已经生成的结果。

### `Score`

```python
@dataclass(frozen=True)
class Score:
    value: float                    # 分数，必定落在 [lo, hi] 内
    lo: float                       # 你要的量程下界
    hi: float                       # 上界
    confidence: float               # 0..1
    latency_ms: float               # 毫秒
    backend: str                    # "jev" | "llm" | "rules"
    degraded: bool = False
    note: str = ""

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Score": ...
```

### `Decider`

```python
class Decider:
    def __init__(self, backend: Literal["auto","jev","llm","rules"] = "auto", *,
                 api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout_s: float = 2.0,
                 fallback_chain: Sequence[str] = ("jev", "llm", "rules"),
                 on_decision: Callable[[Choice | Score], None] | None = None) -> None
```

- `timeout_s`：**单个后端**的超时，秒。整次调用（含所有降级）的上限是它的 **2 倍**。
- `fallback_chain`：`backend="auto"` 时的尝试顺序。缺了 `"rules"` 会自动补在最后
  （必须有一个不会失败的兜底，否则没法保证「永不抛异常」）。
- `on_decision`：每次决策都会被调用一次，拿到的就是返回给你的那个对象。
  回调里抛异常不会影响决策本身（会记一条 warning 日志）。
- 参数不合法一律抛 `ConfigError`（它同时是 `ValueError`）。

```python
    def choice(self, state: Mapping[str, Any], question: str, options: Sequence[str], *,
               rules: Callable[[Mapping[str, Any]], str] | None = None) -> Choice
```

在 `options` 里挑一个。`state` 必须是 **JSON 可序列化**的 mapping；
不是的话抛 `ValueError`，**绝不会被悄悄 `str()` 掉**（见第 6 节）。
`rules` 是给 `rules` 后端用的纯函数 `state -> option`。

```python
    def score(self, state: Mapping[str, Any], rubric: str,
              lo: float = 0.0, hi: float = 10.0, *,
              rules: Callable[[Mapping[str, Any]], float] | None = None) -> Score
```

按 `rubric` 给 `state` 打分。`lo` 必须严格小于 `hi`。
`rules` 返回的数超出量程会被**夹到边界**并在 `note` 里写明。

```python
    @staticmethod
    def gate(c: Choice, *, min_confidence: float,
             on_low: Literal["keep","first","raise"] = "keep",
             current: str | None = None) -> str
```

`c.confidence >= min_confidence` 时返回 `c.value`；否则按 `on_low`：

| `on_low` | 行为 |
|---|---|
| `"keep"` | 返回 `current`，也就是**什么都不改**。必须传 `current` |
| `"first"` | 返回第一个选项（`probs` 保持 `options` 的顺序），约定上是最安全的那个 |
| `"raise"` | 抛 `ValueError`，让上层显式处理 |

```python
    def health(self) -> dict
```

各后端可用性 + 最近延迟分位数（p50/p90/p99，毫秒）+ 成功失败计数 + 最后一条错误。
**不发网络请求、不阻塞**，可以直接挂在仪表盘的刷新 tick 上。返回值是纯 JSON。

```python
    # 只读属性
    decider.backend    -> str              # 构造时选的后端
    decider.chain      -> tuple[str, ...]  # 实际的尝试顺序
    decider.timeout_s  -> float
```

### 异常

```python
class JevDecideError(Exception): ...                 # 本库所有异常的基类
class ConfigError(JevDecideError, ValueError): ...   # 参数错误，同时是 ValueError
```

**只有构造期 / 参数校验会抛异常。** 运行期（超时、网络失败、后端不可用）一律降级，
用 `degraded=True` 和 `note` 说明。唯一的例外是你自己要求的 `gate(on_low="raise")`。

## 6. 后端与配置

### `rules`：你自己的纯函数

零依赖、无 IO、永远可用。这是整条链的地板，也是**断网、没 key 时仍然能跑**的原因。

没传 `rules=` 函数时它不会瞎编：返回第一个选项 + 均匀分布 + `confidence=0.0` +
`degraded=True`，于是 `gate()` 默认就会拒绝行动。这是有意的——
**没有依据的答案，置信度就应该是 0。**

### `jev`：TypeSafe System One（已实现）

按 [docs.typesafe.ai](https://docs.typesafe.ai/introduction) 的公开
[API reference](https://docs.typesafe.ai/api) 实现，端点、鉴权和请求体都来自官方文档，
没有一处是猜的：

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json

{"state": {...}, "model": "jev-latest",
 "questions": {"decision": {"type": "choice", "instructions": "...",
                            "criteria": {"off": null, "low": null, "high": null}}}}
```

- `choice()` 直接映射到 TypeSafe 的 Choice 原语，用它返回的 `probabilities` 和
  `confidence`。
- `score()` 映射到 TypeSafe 的 Score 原语。TypeSafe 的 Score 要的是一组**有序的档位描述**，
  返回的是跨档位的概率加权索引。本库固定用 5 档
  （Not at all / Slightly / Moderately / Largely / Completely），
  再把 0..4 的索引**线性映射**到你要的 `[lo, hi]`。
- **没有 key 就直接标记不可用**，`auto` 跳过它，一个包都不会发出去。

### `llm`：任意 OpenAI 兼容端点

`POST {base_url}/chat/completions`，用 `response_format` 把回答**钉死在 JSON schema 上**
（`strict: true`，选项用 `enum` 限死），所以模型没法用散文回答。
解析失败会带着「你刚才那条被拒了，原因是……」**修复重试一次**（在同一个超时预算内）——
这就是 [instructor](https://github.com/567-labs/instructor) 立下的 schema → 校验 → 重试范式。

这里的 `confidence` **由返回的概率分布算出来**，不采信模型自称的「我有 95% 把握」：
自称的数字没有锚，而分布的形状至少和 `jev` 后端报的是同一个量。

### `auto`：按 `fallback_chain` 依次降级

默认 `("jev", "llm", "rules")`。**超时立刻降级，不阻塞下一个。**

时间预算：整次调用的上限是 `2 × timeout_s`。每个网络后端拿到的额度是
`min(timeout_s, 剩余预算 × 0.9)`，所以三段加起来最多 `1.9 × timeout_s`，
留了一成余量。剩余预算不足 5 ms 时直接跳过网络后端。
`rules` 是纯本地计算，不占预算，**哪怕预算耗尽也一定会被执行**——
这就是「永远有结果返回」的保证。

只要不是链条上的第一个后端给出的答案，`degraded` 就是 `True`，
`note` 里会按顺序列出前面每一个后端失败的原因。

### 环境变量

| 变量 | 作用 | 默认值 |
|---|---|---|
| `TYPESAFE_API_KEY` | `jev` 后端的 key | 无（无 key = 不可用） |
| `TYPESAFE_BASE_URL` | `jev` 后端的 API 根地址 | `https://api.typesafe.ai` |
| `TYPESAFE_DEFAULT_MODEL` | `jev` 后端的模型名 | `jev-latest` |
| `JEV_DECIDE_LLM_API_KEY` / `OPENAI_API_KEY` | `llm` 后端的 key | 无 |
| `JEV_DECIDE_LLM_BASE_URL` / `OPENAI_BASE_URL` | `llm` 后端的 API 根地址 | `https://api.openai.com/v1` |
| `JEV_DECIDE_LLM_MODEL` | `llm` 后端的模型名 | `gpt-4o-mini` |

构造函数里显式传的 `api_key` / `base_url` / `model` **优先于环境变量**，并且会同时交给
`jev` 和 `llm` 两个后端。所以 `backend="auto"` 时如果你只有一边的 key，
建议顺手把 `fallback_chain` 收窄，别让 key 被送去敲错门：

```python
Decider("auto", api_key=openai_key, fallback_chain=("llm", "rules"))
```

### `state` 的序列化规则

`state` 必须是 JSON 可序列化的（数字、字符串、布尔、`None`、列表、字典）。
以下一律抛 `ValueError`，并在消息里指出**具体是哪一条路径**上的哪个类型：

- 自定义对象、`set`、`bytes` 等非 JSON 类型
- `NaN` / `Infinity`（`json.dumps` 默认会吐出非法 JSON）
- 非字符串的字典 key（`json.dumps` 会把 `1` 悄悄变成 `"1"`，这是静默的信息损失）

**绝不会悄悄 `str()` 掉**——那等于偷偷改掉了这次决策依据的输入。

## 7. 边界：不做什么

- **不在实时控制环里用。** 会阻塞到 `2 × timeout_s`。它是任务边界上的工具，不是热路径上的。
- **不加锁、不做线程安全。** 一个实例一个线程。
- **不做重试队列、不做熔断、不做限流退避。** 一次调用走一遍链条就结束。
  429/529 的指数退避请在调用方或网关做（TypeSafe 官方 SDK 自带退避，本库刻意不重复造）。
- **不缓存。** 同样的 `state` 问两次就是两次请求。要缓存请在外面包一层。
- **不维护滑动窗口、不做任何跨帧状态。** 需要「最近 1 秒平均功率」这类量，
  调用方自己算好，作为普通 key 放进 `state`。
- **不做流式输出、不做多问题批量提问。** TypeSafe 的一次调用可以并行问多个问题
  （speculative fan-out），本库的公开 API 一次只问一个；批量属于未实现项，见下面的附录。
- **不落盘、不写配置文件、不自动注册任何账号。** key 只从参数或环境变量读。
- **不包含任何领域专有代码**（外骨骼、机器人、业务规则都不在这里）。
  领域知识在你传进来的 `rules` 函数和 `state` 里。
- **不依赖 `httpx` 或 `requests`。** 标准库 `urllib` 够用，所以连 optional extra 都没做。
- **不碰硬件、不打开串口。**

## 8. 验证与实测数据

`uv run pytest` → **171 个测试全绿**，`ruff check` 无告警。
CI 在 Python 3.11 / 3.12 / 3.13 上跑 pytest + ruff。

**测试里没有一个真实网络请求**：`urllib.request.urlopen` 被整体替换成一个脚本化的假实现，
既验证了我们真实的 `urllib` 代码路径（请求头、编码、错误映射），又不会开任何 socket。

覆盖到的点：

| 分组 | 数量 | 覆盖内容 |
|---|---:|---|
| `test_types.py` | 29 | `Choice`/`Score` 的 JSON 往返、frozen、构造期校验、`probs` 防外部篡改 |
| `test_state_validation.py` | 16 | 非 JSON `state` 抛 `ValueError`、错误路径定位、**确认没有被 `str()` 掉** |
| `test_rules_backend.py` | 10 | 纯函数成功路径、抛异常/返回非法值降级、夹断、无函数时 confidence=0、确定性 |
| `test_jev_backend.py` | 29 | 按官方文档校验请求体与端点、概率归一化、Score 档位线性映射、401/422/429/529/500 降级、畸形响应降级 |
| `test_llm_backend.py` | 14 | JSON schema 请求体、enum 限定、散文回答→修复重试一次、分布加权打分、网关/自建端点 |
| `test_auto_fallback.py` | 35 | **无 key 时 `auto` 跑通且 `degraded=True`**、逐级降级、链条顺序、构造期与参数校验 |
| `test_timeout.py` | 6 | 超时立刻降级、**总耗时 ≤ 2 × `timeout_s`**、单次 ≤ `timeout_s`、预算耗尽仍执行 `rules` |
| `test_gate.py` | 16 | `keep`/`first`/`raise` 三种策略、阈值边界（闭区间）、参数校验 |
| `test_callbacks_and_health.py` | 16 | **`on_decision` 每次都被调用**、回调抛异常不影响决策、`health()` 不发请求、延迟分位数 |

实测延迟（macOS 26.2 / Apple Silicon arm64 / Python 3.11.15，3000 次采样）：

| 路径 | p50 | p90 | p99 |
|---|---:|---:|---:|
| `backend="rules"` 的 `choice()` | 4.8 µs | 5.0 µs | 15.4 µs |
| `backend="auto"` 无 key（走完整条链再落到 rules） | 12.1 µs | 15.2 µs | 78.1 µs |

这两个数字是**没有网络时**的开销。一旦真的调用 `jev` 或 `llm`，
延迟就是那一跳的网络延迟，量级在几十到几百毫秒——所以第 2 节那条警告依然成立：
**不要放进实时控制环。**

## 9. 出处与致谢

- 论文：[arXiv:2607.03528](https://arxiv.org/abs/2607.03528) *Aligning Language Models with
  Selective Prediction* —— `gate()` 的学术名字就是**选择性预测**，
  核心是风险-覆盖率权衡（risk-coverage tradeoff）：主动放弃一部分样本不答，
  换取答了的那部分错误率更低。
- 论文：[arXiv:2603.24704](https://arxiv.org/abs/2603.24704) 保形选择性预测与风险控制
  （conformal selective prediction）——把「阈值定多少」从拍脑袋变成有统计保证的事。
  本库目前只提供阈值这个旋钮，校准留给调用方，见附录。
- 实现参考：[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
  —— jEV 的旗舰用法：动态索引动作空间 + 推测式并行提问。
- 实现参考：[567-labs/instructor](https://github.com/567-labs/instructor)
  —— `llm` 后端的工程标杆：schema → 校验 → 重试。
- 协议文档：[TypeSafe 文档](https://docs.typesafe.ai/introduction) /
  [API reference](https://docs.typesafe.ai/api) —— `jev` 后端的端点、请求体和答案结构全部出自这里。

灵感来自 TypeSafe jEV 的 System One 思路：模型不该只会写字，它应该能给出一个程序可以直接分支的、
带概率的判断。本库是那个思路的一个**极简、无依赖、可降级**的实现。

## 10. 许可

MIT，见 [LICENSE](./LICENSE)。

---

## 附录：待确认

契约里没有写死、本库按**最保守**的解释先做了的地方。接入方如果有不同意见，
可以在不改公开 API 形状的前提下调整实现。

1. **`gate(on_low="keep")` 但没传 `current`**：契约只说「keep = 保持 current」。
   本库的选择是**每次调用都立刻抛 `ConfigError`**（哪怕这次置信度够），
   而不是在低置信度时悄悄返回 `c.value`。理由是宁可在第一次跑测试时就炸，
   也不要凌晨三点静默地按一个不确定的答案行动。
2. **公开异常类**：契约第 3 节的 API 清单里只有 `Choice` / `Score` / `Decider`，
   但第 0.4 节要求每个库定义自己的 `<Lib>Error(Exception)` 基类。
   本库导出了 `JevDecideError` 和 `ConfigError`（后者同时继承 `ValueError`，
   这样「参数错误抛 ValueError」和「有自己的异常基类」两条要求同时满足）。
3. **`from_dict()`**：契约第 3 节只列了 `to_dict`，但第 0.2 节要求所有公开数据结构
   能 JSON 往返。本库两个都实现了。
4. **指定后端时仍会降级到 `rules`**：`Decider("jev")` 的链条实际是 `("jev", "rules")`。
   因为第 0.4 节明确「运行期永不抛异常」，必须有一个不会失败的兜底。
   `backend` 字段会如实写 `"rules"`，不会假装是 `jev` 给的。
5. **`state` 里的 tuple**：按 JSON 数组处理（往返回来会变成 list）。
   `set` / `bytes` / 自定义对象一律拒绝。
6. **`llm` 后端的 `confidence`**：由概率分布算（归一化负熵），不采信模型自报的数字。
   自报值原样保留在 `Choice.raw` 里，需要的话自己取。
7. **`score()` 的档位数**：`jev` 和 `llm` 后端都固定用 5 档再线性映射到 `[lo, hi]`。
   TypeSafe 文档允许 2..10 档，暴露成参数会改动公开 API 的形状，所以先写死。
8. **`health()` 不主动探活**：只报「配置上可不可用」+ 本实例已经观测到的统计，
   不发探测请求。理由是探活会阻塞，而 `health()` 大概率被挂在界面刷新上。
9. **未实现**：一次调用问多个问题（TypeSafe 支持的 speculative fan-out）、
   Noul（是非题）原语、流式、缓存、429/529 的自动退避重试、置信度的保形校准。
   这些都超出了契约第 3 节的公开 API，没有假装实现。

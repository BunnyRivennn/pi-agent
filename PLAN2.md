# PLAN2：流式工具参数 JSON 解析对齐（施工图纸）

> 目标：让 Python 版的流式工具参数解析与官方 TS 版 Pi 的 `utils/json-parse.ts` + `providers/openai-responses-shared.ts` 行为对齐。
> 本文是**施工图纸**：每个改动点写明「文件 / 函数 / 当前行号 / 现状代码 / 改成什么 / 对齐 TS 哪里」。
> 行号基于 2026-09 代码（含 DeepSeek 双 ID 映射修复），函数名是稳定锚点，行号可能随编辑漂移。

---

## 0. 一句话结论

- 校验（jsonschema）：**已完成**，不在本次范围。
- 流式参数解析：**未对齐，但只对齐"最终解析"，不对齐"每片解析"**。真正的缺陷：
  1. **Responses provider**：done 时 `json.loads` 失败即整体归零 `{}`——坏 JSON（路径转义/截断）丢失。
  2. **Completions provider**：同上，收口函数零容错。
  3. completions 的 delta 分支每片调用 `_parse_streaming_json`（:389）——**这个"对齐"反而是要讨论的点，见 §6，不作为要补齐的能力。**

修法：新增 1 个共享模块（容错解析器），改 2 个 provider 的最终收口点，加 1 个测试文件。**不动 agent_core、不改事件协议、不引第三方依赖、不做 delta 每片解析。**

> **关于"TS 每片都解析，我们为什么不跟"**：每片解析是 O(n²) 的全量重扫（200 字符/50 片 ≈ 5000 次字符扫描，vs 最后一次 200 次），且 90% 结果立刻被覆盖、无人读取；补全出的半成品还可能误导（截断数字被猜成错误值）。TS 付这个代价是因为 grammar/custom tool 需要在流中途程序化消费结构化参数；本项目没有此类消费者，UI 看原始字符串即可。详见 §6。

---

## 1. TS 参照行为（对齐目标）

### 1.1 `utils/json-parse.ts` 的三级瀑布

```
parseStreamingJson(s):
  ① JSON.parse(s)                         严格
  ② JSON.parse(repairJson(s))             修复字符串内脏字符后再严格
  ③ partialJson(s)                        补全截断结构（npm partial-json）
  ④ partialJson(repairJson(s))            补全 + 修复组合
  ⑤ 全失败 → {}                           永不抛异常
```

两个修复器职责分离：

| 修复器 | 修什么 | 典型触发 |
|---|---|---|
| `repairJson`（自研字符状态机） | 字符串内裸控制字符（换行/Tab）转义；非法反斜杠（`\U`、结尾孤立 `\`）翻倍 | 参数值含 Windows 路径 `C:\new`、多行文本，**不截断也会炸** |
| `partial-json`（第三方库） | 补全不完整结构：未闭合引号、括号栈、悬空逗号 | 超长被截断、流中断、无 done 事件 |

### 1.2 `openai-responses-shared.ts` 的调用点

```ts
// delta：每片累积后立即解析（约文件中部 function_call_arguments.delta 分支）
slot.block.partialJson += event.delta;
slot.block.arguments = parseStreamingJson(slot.block.partialJson);

// done：服务端完整串覆盖，解析；并只推"新增部分"的 toolcall_delta
slot.block.partialJson = event.arguments;
slot.block.arguments = parseStreamingJson(slot.block.partialJson);
if (event.arguments.startsWith(previousPartialJson)) {
    const delta = event.arguments.slice(previousPartialJson.length);
    if (delta.length > 0) pushToolCallDelta(slot, delta);
}

// output_item.done：定稿，删草稿字段
slot.block.arguments = parseStreamingJson(item.arguments || slot.block.partialJson || "{}");
delete slot.block.partialJson;
```

Python 侧草稿存在 `state.tool_arg_buffers`（独立字典），不挂在消息 block 上，因此**不需要"删草稿字段"那一步**——这是现有设计的优点，保留。

---

## 2. 改动清单（总览）

| # | 文件 | 动作 | 对齐 TS |
|---|---|---|---|
| 1 | `src/pi_agent/pi_ai/providers/_json_repair.py` | **新建** | `json-parse.ts` 全文 |
| 2 | `.../providers/openai.py` → `_extract_tool_call_arguments`（当前 ~:1150） | 委托共享解析器 | parseStreamingJson |
| 3 | `.../providers/openai_completions.py` → `_extract_tool_call_arguments`（:803）与 `_parse_streaming_json`（:818） | 委托共享解析器（去掉假实现） | parseStreamingJson |
| 4 | `tests/test_json_repair.py` | **新建** | 覆盖 §4 全部用例 |

不改：delta 分支（两 provider 均保持"只累积字符串"，见 §6）、`_apply_output_item` 中对 Mapping 型 arguments 的处理（已正确）、agent_loop、事件类型。

---

## 3. 逐项施工

### 改动 1：新建 `src/pi_agent/pi_ai/providers/_json_repair.py`

下划线前缀 = 内部模块。以下为可直接落地的完整代码（`repair_json_string` 逐行移植 TS `repairJson`；`_complete_json` 是 `partial-json` 在「流式 object 参数」场景的简化子集，够用且零依赖）：

```python
"""流式工具参数的容错 JSON 解析。

对齐 TS 版 pi 的 utils/json-parse.ts：
严格解析 → 转义修复 → 截断补全 → 修复+补全 → {}。
任何输入都不抛异常；救不回完整 object 时返回 {}。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_VALID_ESCAPES = set('"\\/bfnrtu')
_CONTROL_ESCAPES = {
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def repair_json_string(raw: str) -> str:
    """修复字符串字面量内部的脏字符（移植 TS repairJson）。

    - 字符串内裸控制字符 → \\n / \\t / \\uXXXX
    - 非法反斜杠转义（\\U、结尾孤立 \\）→ 反斜杠翻倍
    不补全任何缺失的括号/引号（那是 _complete_json 的职责）。
    """
    out: list[str] = []
    in_string = False
    i, n = 0, len(raw)

    while i < n:
        ch = raw[i]

        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            i += 1
            continue

        if ch == "\\":
            next_ch = raw[i + 1] if i + 1 < n else None
            if next_ch is None:                       # 结尾孤立的反斜杠
                out.append("\\\\")
                i += 1
                continue
            if next_ch == "u":
                hexd = raw[i + 2 : i + 6]
                if len(hexd) == 4 and all(
                    c in "0123456789abcdefABCDEF" for c in hexd
                ):
                    out.append(f"\\u{hexd}")
                    i += 6
                    continue
            if next_ch in _VALID_ESCAPES:
                out.append("\\" + next_ch)
                i += 2
                continue
            out.append("\\\\")                        # 非法转义 → 翻倍
            i += 1
            continue

        if ord(ch) <= 0x1F:
            out.append(_CONTROL_ESCAPES.get(ch, f"\\u{ord(ch):04x}"))
        else:
            out.append(ch)
        i += 1

    return "".join(out)


def _complete_json(raw: str) -> str:
    """尽力补全被截断的 JSON（partial-json 的 object/array 子集）。

    补：未闭合的字符串引号、未闭合的 {[ 括号栈、字符串外的悬空逗号。
    在 repair_json_string 之后调用（裸字符已转义，扫描状态才可靠）。
    """
    s = raw.rstrip()

    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    # 字符串外的悬空逗号（如 {"a": 1,）→ 去掉；字符串内的逗号保留（靠补引号收口）
    if not in_string and s.endswith(","):
        s = s[:-1]

    return s + ('"' if in_string else "") + "".join(reversed(stack))


def parse_streaming_json(raw: Any) -> dict[str, Any]:
    """对齐 TS parseStreamingJson。返回 dict；任何失败都安全降级为 {}。"""
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    if not isinstance(raw, str) or not raw.strip():
        return {}

    repaired = repair_json_string(raw)

    for candidate in (raw, repaired, _complete_json(raw), _complete_json(repaired)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return {str(key): value for key, value in parsed.items()}

    return {}
```

> 瀑布顺序与 TS 完全一致：① 原始严格 → ② 修复后严格 → ③ 补全原始 → ④ 补全修复后。

### 改动 2：`openai.py` 的 `_extract_tool_call_arguments`（当前 ~:1150）

现状：

```python
def _extract_tool_call_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        if isinstance(parsed, Mapping):
            return {str(key): value for key, value in parsed.items()}
    return {}
```

改为整体委托（函数名保留，因为 done 分支和 `_close_open_tool_calls` 兜底路径都在调它）：

```python
def _extract_tool_call_arguments(raw: Any) -> dict[str, Any]:
    return parse_streaming_json(raw)
```

这样两个调用点（done 时的最终解析、流异常结束时的兜底）同时获得容错。

### 改动 3：`openai_completions.py` 的两个函数（:803 / :818）

`_extract_tool_call_arguments` 整体委托（`_finish_current_block` :432 的调用点自动获益）：

```python
def _extract_tool_call_arguments(raw: Any) -> dict[str, Any]:
    return parse_streaming_json(raw)
```

`_parse_streaming_json`（:818，当前是零容错假实现）有两个选择：

- **方案 A（推荐，最小改动）**：同样委托共享解析器，**但保留 :389 的每片调用不动**——函数变真后每片解析自动有了容错，零额外成本，行为只在"坏 JSON"时从 `{}` 变为尽力救回。
- **方案 B（更彻底）**：删掉 :389 的每片调用与 `_parse_streaming_json`，delta 分支只保留 buffer 累积（与 responses provider 统一），解析只发生在 `_finish_current_block`。

本图纸默认 A；若认同 §6"每片解析无消费者"的判断，B 更干净但需确认没有外部代码读 partial.arguments。两者最终结果等价（终态都从完整串解析）。

---

## 4. 测试：新建 `tests/test_json_repair.py`

参数化覆盖下列输入 → 期望（全部通过即对齐）：

| # | 输入 | 期望输出 | 对应级别 |
|---|---|---|---|
| 1 | `{"city": "Shanghai"}` | `{"city": "Shanghai"}` | ① 严格 |
| 2 | `{"city": "Shanghai"}` 的 dict 直传 | 原样 dict | Mapping 直通 |
| 3 | `""` / 空白 / None | `{}` | 空值守卫 |
| 4 | `{"path": "C:\\new\\test"}`（源码里即 `\n` `\t` 非法转义形态） | `{"path": "C:\\new\\test"}`（值保留） | ② repair |
| 5 | `{"q": "a`<裸换行>`b"}` | 值中的换行被正确转义后解析 | ② repair |
| 6 | `{"city": "Shang` | `{"city": "Shang"}` | ③ 截断补全 |
| 7 | `{"a": 1,` | `{"a": 1}` | ③ 悬空逗号 |
| 8 | `{"a": [1, 2` | `{"a": [1, 2]}` | ③ 嵌套括号栈 |
| 9 | `{"a": 1, "b": {"c":` | `{"a": 1, "b": {"c": None}}` 或安全 `{}`（断言"不抛异常 + 1 已救回"二选一，按实现定） | ④ |
| 10 | `完全不是 json` | `{}` | ⑤ 兜底，不抛 |
| 11 | `{"a": 1}垃圾尾随` | `{}`（严格模式拒绝尾随垃圾，不误解析） | 安全性 |

另加两个**集成级**测试（仿现有 tests 风格，喂伪造事件给 provider）：

- Responses：发「带非法转义的完整 arguments.done」→ 终态 AssistantMessage 的 ToolCall.arguments 正确
- Completions：发「被截断的最后一个 arguments chunk（无完整 JSON）」→ 走 `_finish_current_block` 后救回已闭合字段

## 5. 验收命令

```bash
uv run pytest tests/ -q          # 现有 40 个 + 新增用例全过
uv run python examples/openai_agent_streaming.py   # DeepSeek 真实链路回归（工具只打印一次）
uv run python examples/debug_provider_events.py    # 伪造事件回归
```

## 6. 明确不做

- **不做 delta 每片解析（responses provider 不加；completions 见改动 3 方案 B 选项）**。TS `slot.block.arguments = parseStreamingJson(partialJson)` 每片执行，我们不跟，理由：
  1. **无消费者**：项目没有流式结构化 UI、没有 grammar tool、没有流中途程序化读取 arguments 的逻辑；给人看直接渲染原始 buffer 字符串即可。
  2. **O(n²) 浪费**：解析是全量重扫非增量，n 字符 k 片累计 O(n·k) 扫描，90%+ 结果下一片即覆盖。
  3. **半成品会撒谎**：补全器对截断值的猜测无正确性保证（`20` vs `202`），被中途读取反成 bug 源。
  4. **最终结果不受影响**：终态消息从 done/`output_item.done` 的完整串解析，delta 解析只影响过程中的 partial 快照。
  - 未来若出现"流未结束就需结构化参数"的真实需求（流式表单 UI、参数预取/预校验、grammar 类协议），再按 TS 形状启用，插入点就在 delta 分支累积 buffer 之后。
- **不引第三方库**：object 参数场景下自研 ~50 行已覆盖；若未来要支持深层嵌套/数组流式半成品，再评估 `json-repair`（对标 npm partial-json）。
- **不做"为截断而抢救"的产品化承诺**：截断救回的值本身可能残缺（`"Shang"`），是否重试由上层 agent 决定；解析器只负责"不丢已有信息"。
- **不移植 grammar/custom tool**（`custom_tool_call_input.*`、`constrained-sampling.ts`、每片结构化 input）：OpenAI grammar 专属协议。
- **不改 output_index 槽位方案**：DeepSeek 双 ID 已用 `tool_item_id_to_call_id` 映射修好；output_index 重构属 Phase 4 兼容性议题，另案。
- **不补 done 增量去重**：Python 协议事件无重复，TS 那段是针对其 SDK 事件形状的防御。

## 7. 来源

- Python：`src/pi_agent/pi_ai/providers/openai.py`、`openai_completions.py`
- TS 官方 Pi：`providers/openai-responses-shared.ts`（processResponsesStream）、`utils/json-parse.ts`（repairJson / parseStreamingJson）、`providers/openai-codex-responses.ts`（草稿字段清理）
- 实测：DeepSeek Responses deepseek-v4-flash，2026-09

# PLAN2：流式工具参数 JSON 解析对齐（施工图纸）

> 目标：让 Python 版的流式工具参数解析与官方 TS 版 Pi 的 `utils/json-parse.ts` + `providers/openai-responses-shared.ts` 行为对齐。
> 本文是**施工图纸**：每个改动点写明「文件 / 函数 / 当前行号 / 现状代码 / 改成什么 / 对齐 TS 哪里」。
> 行号基于 2026-09 代码（含 DeepSeek 双 ID 映射修复），函数名是稳定锚点，行号可能随编辑漂移。

---

## 0. 一句话结论

- 校验（jsonschema）：**已完成**，不在本次范围。
- 流式参数解析：**未对齐**。两处缺陷：
  1. **Responses provider**：delta 时只拼字符串、不解析（TS 每片都解析）；done 时 `json.loads` 失败即整体归零 `{}`。
  2. **Completions provider**：delta 时**已经每片调用** `_parse_streaming_json`（结构已对齐），但该函数是假的——只是 `json.loads` 包一层，零容错，坏 JSON 照样归零。

修法：新增 1 个共享模块（容错解析器），改 2 个 provider 共 4 个收口点，加 1 个测试文件。**不动 agent_core、不改事件协议、不引第三方依赖。**

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
| 2 | `.../providers/openai.py` → delta 分支（当前 :370-382） | 累积后加一行每片解析 | shared.ts delta 分支 |
| 3 | `.../providers/openai.py` → `_extract_tool_call_arguments`（当前 ~:1150） | 委托共享解析器 | parseStreamingJson |
| 4 | `.../providers/openai_completions.py` → `_parse_streaming_json`（:818） | 委托共享解析器（去掉假实现） | parseStreamingJson |
| 5 | `.../providers/openai_completions.py` → `_extract_tool_call_arguments`（:803） | 委托共享解析器 | parseStreamingJson |
| 6 | `tests/test_json_repair.py` | **新建** | 覆盖 §4 全部用例 |

不改：`_apply_output_item` 中对 Mapping 型 arguments 的处理（已正确）、agent_loop、事件类型。

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

### 改动 2：`openai.py` delta 分支（当前 :370-382）加每片解析

现状（只累积、不解析，partial 里 arguments 一直是 `{}` 直到 done）：

```python
        delta = _as_str(event.get("delta")) or ""
        if delta:
            state.tool_arg_buffers[call_id] = (
                    state.tool_arg_buffers.get(call_id, "") + delta
            )
            stream.push(
                {
                    "type": "toolcall_delta",
                    "content_index": content_index,
                    "delta": delta,
                    "partial": state.partial,
                }
            )
        return False
```

改为（累积后立刻解析回写 ToolCall.arguments，对齐 shared.ts delta 分支）：

```python
        delta = _as_str(event.get("delta")) or ""
        if delta:
            buffer = state.tool_arg_buffers.get(call_id, "") + delta
            state.tool_arg_buffers[call_id] = buffer
            tool_call = cast(ToolCall, state.partial.content[content_index])
            tool_call.arguments = parse_streaming_json(buffer)
            stream.push(
                {
                    "type": "toolcall_delta",
                    "content_index": content_index,
                    "delta": delta,
                    "partial": state.partial,
                }
            )
        return False
```

文件顶部加导入：

```python
from ._json_repair import parse_streaming_json
```

> 说明：TS 的 done 增量去重（startsWith 切片）**不做**——Python 的 `function_call_arguments.done` 事件本身不重复推 delta，无重复显示问题。

### 改动 3：`openai.py` 的 `_extract_tool_call_arguments`（当前 ~:1150）

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

### 改动 4：`openai_completions.py` 的 `_parse_streaming_json`（:818）

现状（名字叫 streaming，实际零容错）：

```python
def _parse_streaming_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    return _extract_tool_call_arguments(raw)
```

改为：删除此函数，文件顶部加 `from ._json_repair import parse_streaming_json`，把 :389 的调用改为共享函数：

```python
        existing_tool_call.arguments = parse_streaming_json(buffer)
```

### 改动 5：`openai_completions.py` 的 `_extract_tool_call_arguments`（:803）

与改动 3 相同，整体委托：

```python
def _extract_tool_call_arguments(raw: Any) -> dict[str, Any]:
    return parse_streaming_json(raw)
```

（`_finish_current_block` :432 的调用点自动获益。）

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

- **不引第三方库**：object 参数场景下自研 ~50 行已覆盖；若未来要支持深层嵌套/数组流式半成品，再评估 `json-repair`（对标 npm partial-json）。
- **不做"为截断而抢救"的产品化承诺**：截断救回的值本身可能残缺（`"Shang"`），是否重试由上层 agent 决定；解析器只负责"不丢已有信息"。
- **不移植 grammar/custom tool**（`custom_tool_call_input.*`、`constrained-sampling.ts`、每片结构化 input）：OpenAI grammar 专属协议。
- **不改 output_index 槽位方案**：DeepSeek 双 ID 已用 `tool_item_id_to_call_id` 映射修好；output_index 重构属 Phase 4 兼容性议题，另案。
- **不补 done 增量去重**：Python 协议事件无重复，TS 那段是针对其 SDK 事件形状的防御。

## 7. 来源

- Python：`src/pi_agent/pi_ai/providers/openai.py`、`openai_completions.py`
- TS 官方 Pi：`providers/openai-responses-shared.ts`（processResponsesStream）、`utils/json-parse.ts`（repairJson / parseStreamingJson）、`providers/openai-codex-responses.ts`（草稿字段清理）
- 实测：DeepSeek Responses deepseek-v4-flash，2026-09

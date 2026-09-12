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
            if next_ch is None:  # 结尾孤立的反斜杠
                out.append("\\\\")
                i += 1
                continue
            if next_ch == "u":
                hexd = raw[i + 2: i + 6]
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
            out.append("\\\\")  # 非法转义 → 翻倍
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
    """容错解析工具参数。返回 dict；任何失败都安全降级为 {}。

    瀑布顺序：严格 → 转义修复后严格 → 结构补全（补引号/括号/去尾逗号）。

    重要原则：**只修表达形式，绝不删除模型已生成的语义内容**。
    不做"砍尾抢救"（如 {"a": 1, "cit → {"a": 1}）——那会静默丢弃字段，
    让下游拿着残缺参数继续执行；返回 {} 让 jsonschema required 显式拦下，
    由 agent loop 把错误喂回模型重试，才是正确的失败方式。
    """
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

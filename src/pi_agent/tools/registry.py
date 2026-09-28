"""内置工具的统一注册表。

设计取舍：**显式注册表 + 装饰器登记**，但不依赖「装饰器副作用」来保证可用性。

- 每个工具模块导出一个工厂函数 ``create_xxx_tool(cwd, options) -> AgentTool``；
  工厂函数（而不是模块级单例）意味着 cwd 是参数，同一进程可以并存多个绑定不同
  目录的实例。官方也是这么改的（删掉了预绑定 cwd 的单例导出）。
- 工厂函数用 :func:`register_tool` 登记到全局表；``tools/__init__.py`` 里显式
  import 全部内置模块，保证表一定是满的——装饰器最大的坑就是「模块没被 import
  所以没注册」，靠显式 import 一次性堵掉。
- 第三方/自定义工具在自己的模块里挂同一个装饰器即可，无需改动本文件。

加一个自己的工具：

    from pi_agent.tools import register_tool, ToolFactoryOptions

    @register_tool("my_tool")
    def create_my_tool(cwd, options=None):
        return AgentTool(name="my_tool", ...)

然后 ``create_tools(["my_tool"], cwd)`` 就能拿到它。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..agent_core.types import AgentTool

__all__ = [
    "ToolFactory",
    "register_tool",
    "tool_names",
    "create_tool",
    "create_tools",
    "create_coding_tools",
    "create_read_only_tools",
    "create_all_tools",
    "build_tools_prompt",
    "CODING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "ALL_TOOL_NAMES",
]

#: 工具工厂：接收 cwd 与可选配置，返回一个可执行的 AgentTool。
ToolFactory = Callable[..., AgentTool]

_REGISTRY: dict[str, ToolFactory] = {}

#: 最小编码集：读、跑命令、改、写。对齐官方 createCodingTools。
CODING_TOOL_NAMES: tuple[str, ...] = ("read", "bash", "edit", "write")

#: 只读集，适合审查/探索型 agent。对齐官方 createReadOnlyTools。
READ_ONLY_TOOL_NAMES: tuple[str, ...] = ("read", "grep", "find", "ls")

#: 全集。powershell 仅在 Windows 上真正可用，其他平台上调用时才报错，
#: 这样装配工具集本身不会因平台差异失败。
ALL_TOOL_NAMES: tuple[str, ...] = (
    "read",
    "bash",
    "powershell",
    "edit",
    "write",
    "grep",
    "find",
    "ls",
)


def register_tool(name: str) -> Callable[[ToolFactory], ToolFactory]:
    """把工具工厂登记到全局注册表。

    Raises:
        ValueError: 同名工具被重复注册（通常是复制粘贴时忘了改名）。
    """

    def decorator(factory: ToolFactory) -> ToolFactory:
        if name in _REGISTRY:
            raise ValueError(f"Tool already registered: {name}")
        _REGISTRY[name] = factory
        return factory

    return decorator


def tool_names() -> tuple[str, ...]:
    """当前已注册的全部工具名（字母序，不依赖 import 顺序）。"""
    return tuple(sorted(_REGISTRY))


def create_tool(
    name: str,
    cwd: Path,
    options: Mapping[str, Any] | None = None,
) -> AgentTool:
    """按名字造一个工具实例。

    Args:
        name: 已注册的工具名。
        cwd: 该实例绑定的工作目录。
        options: 传给工厂的关键字参数。

    Raises:
        KeyError: 工具名未注册。
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown tool: {name}. Registered tools: {available}")
    return factory(cwd, **(options or {}))


def create_tools(
    names: Iterable[str],
    cwd: Path,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[AgentTool]:
    """批量造工具。

    Args:
        names: 工具名序列，返回顺序与之一致。
        cwd: 绑定的工作目录。
        options: ``{工具名: 该工具的 kwargs}``。
    """
    per_tool = options or {}
    return [create_tool(name, cwd, per_tool.get(name)) for name in names]


def create_coding_tools(
    cwd: Path,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[AgentTool]:
    """最小编码套餐：read / bash / edit / write。"""
    return create_tools(CODING_TOOL_NAMES, cwd, options)


def create_read_only_tools(
    cwd: Path,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[AgentTool]:
    """只读套餐：read / grep / find / ls。适合代码审查、仓库探索等不该落盘的场景。"""
    return create_tools(READ_ONLY_TOOL_NAMES, cwd, options)


def create_all_tools(
    cwd: Path,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[AgentTool]:
    """全套内置工具。

    用 :data:`ALL_TOOL_NAMES` 而不是 :func:`tool_names`，故顺序固定且不会
    把第三方通过 :func:`register_tool` 注册的工具意外带进来。
    """
    return create_tools(ALL_TOOL_NAMES, cwd, options)


def build_tools_prompt(tools: Iterable[AgentTool]) -> str:
    """把工具自带的提示词元数据拼成系统提示词片段。

    这样「启用了哪些工具」和「系统提示词怎么描述它们」永远一致，
    加工具不需要回去手改提示词。
    """
    tool_list = list(tools)
    lines: list[str] = []

    snippets = [(t.name, t.prompt_snippet) for t in tool_list if t.prompt_snippet]
    if snippets:
        lines.append("Available tools:")
        lines.extend(f"- {name}: {snippet}" for name, snippet in snippets)

    guidelines = [g for t in tool_list for g in t.prompt_guidelines]
    if guidelines:
        if lines:
            lines.append("")
        lines.append("Guidelines:")
        lines.extend(f"- {g}" for g in guidelines)

    return "\n".join(lines)

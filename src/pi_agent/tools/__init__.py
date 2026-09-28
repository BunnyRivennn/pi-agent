"""内置工具集。

每个工具一个模块，通过 :func:`register_tool` 登记到统一注册表；本文件**显式
import 全部内置模块**，保证 ``create_tools()`` 拿到的表一定是满的（装饰器注册
最常见的坑就是模块没被 import 过，这里一次性堵掉）。工具的供给顺序由
:data:`CODING_TOOL_NAMES` 等显式常量决定，不依赖 import 顺序。

典型用法::

    from pathlib import Path
    from pi_agent.tools import create_coding_tools, build_tools_prompt

    tools = create_coding_tools(Path.cwd())
    agent.set_tools(tools)
    agent.set_system_prompt(f"You are a coding agent.\\n\\n{build_tools_prompt(tools)}")

想限制工具只能在项目目录内读写::

    tools = create_coding_tools(root, {"read": {"root": root}, "edit": {"root": root}})

加自己的工具：在任意模块里挂 ``@register_tool("名字")``，import 该模块后即可用。
"""

from __future__ import annotations

from .bash import create_bash_tool
from .edit import create_edit_tool
from .find import create_find_tool
from .grep import create_grep_tool
from .ls import create_ls_tool
from .paths import PathOutsideRootError
from .powershell import create_powershell_tool
from .read import create_read_tool
from .registry import (
    ALL_TOOL_NAMES,
    CODING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    ToolFactory,
    build_tools_prompt,
    create_all_tools,
    create_coding_tools,
    create_read_only_tools,
    create_tool,
    create_tools,
    register_tool,
    tool_names,
)
from .write import create_write_tool

__all__ = [
    # 注册表
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
    # 工厂
    "create_read_tool",
    "create_bash_tool",
    "create_powershell_tool",
    "create_edit_tool",
    "create_write_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
    # 错误
    "PathOutsideRootError",
]

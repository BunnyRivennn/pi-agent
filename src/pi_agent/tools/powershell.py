"""``powershell`` 工具：执行 PowerShell 命令。

与 ``bash`` 共用 :mod:`.shell` 的执行逻辑，只换解释器和命令前缀。
在 Windows 上做 COM/注册表/WMI 这类操作时，PowerShell 比 bash 顺手得多。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..agent_core.types import AgentTool
from .registry import register_tool
from .shell import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    ShellConfig,
    create_shell_tool,
    resolve_powershell_shell,
)

__all__ = [
    "create_powershell_tool",
    "POWERSHELL_PROMPT_SNIPPET",
    "POWERSHELL_PROMPT_GUIDELINES",
]

POWERSHELL_PROMPT_SNIPPET = "Execute PowerShell commands"
POWERSHELL_PROMPT_GUIDELINES = (
    "Use powershell for Windows-specific tasks (registry, WMI, COM, Windows services); "
    "prefer bash for ordinary file and text work.",
)

#: PowerShell 默认按控制台代码页输出，中文会变成乱码；强制 UTF-8。
#: try/catch 包起来是因为某些宿主（如 ISE）不允许改 OutputEncoding。
_UTF8_PREFIX = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n"

_CONFIG = ShellConfig(
    name="powershell",
    label="powershell",
    shell_name="PowerShell",
    description=(
        "Run a PowerShell command and return its combined stdout/stderr plus exit code. "
        "Every call is subject to a timeout; output is truncated in the middle if large."
    ),
    prompt_snippet=POWERSHELL_PROMPT_SNIPPET,
    prompt_guidelines=POWERSHELL_PROMPT_GUIDELINES,
    resolve_shell=resolve_powershell_shell,
    command_prefix=_UTF8_PREFIX,
)


@register_tool("powershell")
def create_powershell_tool(
    cwd: Path,
    *,
    shell: Sequence[str] | None = None,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    env: Mapping[str, str] | None = None,
) -> AgentTool:
    """构造 ``powershell`` 工具。

    系统上没有 PowerShell 时，**创建不会失败**，而是在调用时报错——
    这样工具集的装配不会因为平台差异整个崩掉。
    """
    return create_shell_tool(
        cwd,
        _CONFIG,
        shell=shell,
        default_timeout=default_timeout,
        max_output_bytes=max_output_bytes,
        env=env,
    )

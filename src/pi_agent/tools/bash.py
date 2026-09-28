"""``bash`` 工具：执行 shell 命令。

实现在 :mod:`.shell`，本模块只提供 bash 的差异化配置。
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
    resolve_bash_shell,
)

__all__ = ["create_bash_tool", "BASH_PROMPT_SNIPPET", "BASH_PROMPT_GUIDELINES"]

BASH_PROMPT_SNIPPET = "Run shell commands"
BASH_PROMPT_GUIDELINES = (
    "Every bash call has a timeout; long-running servers or watchers must be started in the "
    "background with output redirected to a log file, then polled.",
    "Prefer the dedicated read/edit/write/grep/find/ls tools over their shell equivalents "
    "(cat/sed/tee/grep/find/ls) — they give structured, truncation-aware output.",
)

_CONFIG = ShellConfig(
    name="bash",
    label="bash",
    shell_name="bash",
    description=(
        "Run a shell command and return its combined stdout/stderr plus exit code. "
        "Every call is subject to a timeout; output is truncated in the middle if large."
    ),
    prompt_snippet=BASH_PROMPT_SNIPPET,
    prompt_guidelines=BASH_PROMPT_GUIDELINES,
    resolve_shell=resolve_bash_shell,
)


@register_tool("bash")
def create_bash_tool(
    cwd: Path,
    *,
    shell: Sequence[str] | None = None,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    env: Mapping[str, str] | None = None,
) -> AgentTool:
    """构造 ``bash`` 工具。

    Args:
        cwd: 命令的工作目录。
        shell: 覆盖默认 shell，如 ``("/bin/sh", "-c")``。
        default_timeout: 未指定 timeout 时的默认秒数。
        max_output_bytes: 输出截断阈值。
        env: 覆盖/追加的环境变量。
    """
    return create_shell_tool(
        cwd,
        _CONFIG,
        shell=shell,
        default_timeout=default_timeout,
        max_output_bytes=max_output_bytes,
        env=env,
    )

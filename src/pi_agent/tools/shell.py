"""shell 类工具的公共实现。

``bash`` 和 ``powershell`` 的差别只有「用哪个解释器、怎么传命令字符串」，
执行/超时/中断/截断逻辑完全一致——官方也是这么组织的（``powershell.ts``
复用 ``bash.ts`` 的 ``createShellToolDefinition``）。

几个关键取舍：

- **必须有超时**。没有超时的命令（服务、watcher、死循环）会永久挂住 agent。
- **输出双端截断**。报错通常在尾部，只留头部会丢掉最关键的信息，所以保留
  头尾、省略中间。
- **stdout/stderr 合并**。模型需要同时看到两者才能判断成败。
- **进程组级终止**。只 kill 直接子进程会留下孤儿（``sh -c`` 起的子进程），
  POSIX 上按进程组发信号。
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any

from ..agent_core.types import AgentTool, AgentToolResult, TextContent

__all__ = [
    "ShellConfig",
    "create_shell_tool",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OUTPUT_BYTES",
]

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_BYTES = 30_000


@dataclass(frozen=True, slots=True)
class ShellConfig:
    """一种 shell 工具的差异化配置。"""

    name: str
    label: str
    shell_name: str
    description: str
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...]
    #: 返回 ``(可执行文件, 执行参数)``，如 ``("/bin/sh", "-c")``。
    resolve_shell: Callable[[], Sequence[str]]
    #: 在用户命令前拼接的前缀（PowerShell 用它强制 UTF-8 输出）。
    command_prefix: str = ""


def _truncate_middle(text: str, max_bytes: int) -> str:
    """超长输出保留头尾，中间标注省略了多少。"""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    head_size = max_bytes // 2
    tail_size = max_bytes - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore")
    omitted = len(encoded) - head_size - tail_size
    return f"{head}\n\n[... {omitted} bytes omitted ...]\n\n{tail}"


def resolve_bash_shell() -> Sequence[str]:
    """挑一个 bash。

    Windows 上优先 Git Bash（项目约定用它，而不是 PowerShell）；找不到就退回
    ``cmd /c``，保证工具在裸 Windows 上也不是废的。
    """
    if sys.platform != "win32":
        return ("/bin/sh", "-c")

    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ):
        if Path(candidate).exists():
            return (candidate, "-c")

    found = which("bash")
    return (found, "-c") if found else ("cmd.exe", "/c")


def resolve_powershell_shell() -> Sequence[str]:
    """挑一个 PowerShell，优先跨平台的 pwsh。"""
    for exe in ("pwsh", "powershell"):
        found = which(exe)
        if found:
            # -NoProfile 避免用户配置文件拖慢启动或污染输出。
            return (found, "-NoProfile", "-NonInteractive", "-Command")
    raise RuntimeError(
        "PowerShell is not available on this system (looked for 'pwsh' and 'powershell')."
    )


def build_shell_schema(default_timeout: float) -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout": {
                "type": "number",
                "description": f"Timeout in seconds (default {default_timeout:g})",
            },
        },
        "required": ["command"],
    }


def create_shell_tool(
    cwd: Path,
    config: ShellConfig,
    *,
    shell: Sequence[str] | None = None,
    default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    env: Mapping[str, str] | None = None,
) -> AgentTool:
    """按 ``config`` 造一个 shell 工具。"""

    async def execute(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, on_update
        command = str(params["command"])
        timeout = float(params.get("timeout") or default_timeout)

        # shell 解析放到执行时：PowerShell 缺失应该报成工具错误，
        # 而不是在创建工具集时就把整个 agent 炸掉。
        shell_cmd = tuple(shell) if shell else tuple(config.resolve_shell())

        child_env = {**os.environ, **(env or {})}
        # 让子进程的 Python 输出用 UTF-8，否则 Windows 控制台默认 cp1252 会炸中文。
        child_env.setdefault("PYTHONIOENCODING", "utf-8")

        popen_kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            # 自成进程组，便于连带终止整棵子进程树。
            popen_kwargs["start_new_session"] = True

        process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            config.command_prefix + command,
            cwd=str(cwd),
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **popen_kwargs,
        )

        async def _terminate() -> None:
            if process.returncode is not None:
                return
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass

        # 命令执行与中断信号赛跑，谁先到算谁。
        communicate_task = asyncio.ensure_future(process.communicate())
        waiters: list[asyncio.Future[Any]] = [communicate_task]
        abort_task: asyncio.Future[Any] | None = None
        if abort_event is not None:
            abort_task = asyncio.ensure_future(abort_event.wait())
            waiters.append(abort_task)

        timed_out = False
        aborted = False
        try:
            done, _pending = await asyncio.wait(
                waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if communicate_task in done:
                stdout_bytes, _ = communicate_task.result()
            elif abort_task is not None and abort_task in done:
                aborted = True
                await _terminate()
                stdout_bytes, _ = await communicate_task
            else:
                timed_out = True
                await _terminate()
                stdout_bytes, _ = await communicate_task
        finally:
            if abort_task is not None and not abort_task.done():
                abort_task.cancel()
            if not communicate_task.done():
                communicate_task.cancel()

        output = _truncate_middle((stdout_bytes or b"").decode("utf-8", errors="replace"),
                                  max_output_bytes)
        exit_code = process.returncode

        if aborted:
            raise RuntimeError("Operation aborted")

        pieces: list[str] = []
        if timed_out:
            pieces.append(f"[command timed out after {timeout:g}s and was killed]")
        pieces.append(output if output.strip() else "(no output)")
        if exit_code not in (0, None):
            pieces.append(f"[exit code: {exit_code}]")

        return AgentToolResult(
            content=[TextContent(text="\n".join(pieces))],
            details={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "cwd": str(cwd),
                "shell": config.shell_name,
            },
        )

    return AgentTool(
        name=config.name,
        label=config.label,
        description=config.description,
        execute=execute,
        parameters=build_shell_schema(default_timeout),
        prompt_snippet=config.prompt_snippet,
        prompt_guidelines=config.prompt_guidelines,
    )

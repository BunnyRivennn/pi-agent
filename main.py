"""pi-agent 交互式命令行。

    uv run python main.py                 # 新建会话
    uv run python main.py --resume        # 续聊最近一次会话
    uv run python main.py --session <id>  # 续聊指定会话
    uv run python main.py --list          # 列出所有会话

配置走 .env（参见 .env.example）：
    OPENAI_API_KEY        必填
    OPENAI_MODEL_NAME     模型名，默认 gpt-5-mini
    OPENAI_API_BASE_URL   兼容端点（DeepSeek / Ark 等）
    OPENAI_API_STYLE      responses（默认）| completions

对话历史落在 ``<项目根>/session/storage/sessions.db``（全部会话共用这一个
库文件，靠 session_id 隔离），退出后可用 --resume 续聊。
上下文超阈值时自动压缩（摘要走同一个模型）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pi_agent.agent_core import (
    Agent,
    AgentTool,
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
)
from pi_agent.agent_core.compaction import CompactionSettings
from pi_agent.agent_core.types import AgentEvent
from pi_agent.pi_ai import create_agent_stream_fn, create_default_registry
from pi_agent.pi_ai.runtime import complete_simple
from pi_agent.session import Context, SqliteSessionRepo, create_agent_session
from pi_agent.session.factory import AgentSession
from pi_agent.tools import (
    build_tools_prompt,
    create_all_tools,
    create_coding_tools,
    create_read_only_tools,
)

load_dotenv()

# 会话 db 的落点。钉在本文件所在目录（项目根）而不是 cwd，
# 否则从别的目录起进程会到处散建 session 目录。
# 注意：这是运行时数据目录，与源码包 src/pi_agent/session/storage/ 同名但无关。
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "session" / "storage" / "sessions.db"
# 摘要 prompt + 输出要占额度，给压缩留的余量见 CompactionSettings.reserve_tokens。
DEFAULT_CONTEXT_WINDOW = 128_000

HELP = """\
可用命令：
  /help          显示本帮助
  /history       打印当前上下文里的消息
  /compact       立即压缩上下文（需要已启用压缩）
  /session       显示当前 session id 与 db 路径
  /clear         清空屏幕
  /exit, /quit   退出（历史已落盘，下次 --resume 可续聊）
按 Ctrl+C 中断当前回答，Ctrl+D 退出。"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def build_tools(workdir: Path, *, sandbox: bool, preset: str) -> list[AgentTool]:
    """装配内置工具集。

    Args:
        workdir: 工具的工作目录，相对路径以它为基准。
        sandbox: 为真时把文件读写限制在 ``workdir`` 内。注意 shell 类工具绕不过
            这层护栏（它们能执行任意命令），真要隔离得靠容器。
        preset: ``coding``（read/bash/edit/write）、``readonly``（read/grep/find/ls）
            或 ``all``（全部 8 个）。
    """
    root = workdir if sandbox else None
    # shell 类工具（bash/powershell）没有 root 参数——边界由 cwd 和超时决定。
    sandboxed = ("read", "edit", "write", "grep", "find", "ls")
    options: dict[str, dict[str, Any]] = {name: {"root": root} for name in sandboxed}

    factories = {
        "coding": create_coding_tools,
        "readonly": create_read_only_tools,
        "all": create_all_tools,
    }
    return factories[preset](workdir, options)


# ---------------------------------------------------------------------------
# 流式打印
# ---------------------------------------------------------------------------


def make_printer() -> Callable[[AgentEvent], None]:
    """把 Agent 事件流渲染成终端打字机输出。"""
    state = {"streamed_text": False}

    def on_event(event: AgentEvent) -> None:
        etype = event["type"]

        if etype == "message_start":
            if isinstance(event.get("message"), AssistantMessage):
                state["streamed_text"] = False
                print("\033[92m助手 >\033[0m ", end="", flush=True)
            return

        if etype == "message_update":
            inner = event.get("assistant_message_event") or {}
            itype = inner.get("type")
            if itype == "text_delta":
                print(inner["delta"], end="", flush=True)
                state["streamed_text"] = True
            elif itype == "toolcall_end":
                call = inner.get("tool_call")
                if isinstance(call, ToolCall):
                    print(f"\n  \033[90m⚙ {call.name}({call.arguments})\033[0m")
            return

        if etype == "tool_execution_end":
            flag = "✗" if event.get("is_error") else "✓"
            print(f"  \033[90m{flag} {event['tool_name']}\033[0m")
            return

        if etype == "message_end":
            msg = event.get("message")
            if isinstance(msg, AssistantMessage):
                if state["streamed_text"]:
                    print()
                else:
                    # 纯工具调用回合没有文本，把前缀行收掉
                    print("\r", end="")
                if msg.stop_reason == "error" and msg.error_message:
                    print(f"\033[91m[错误] {msg.error_message}\033[0m")
            return

    return on_event


# ---------------------------------------------------------------------------
# 模型接线
# ---------------------------------------------------------------------------


def build_model() -> tuple[Model, Any]:
    """按 .env 选择 provider 风格，返回 (model, registry)。"""
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini")
    base_url = os.getenv("OPENAI_API_BASE_URL", "")
    style = os.getenv("OPENAI_API_STYLE", "responses").lower()

    provider = "openai-completions" if style == "completions" else "openai"
    registry = create_default_registry()
    model = Model(id=model_name, provider=provider, api=provider, base_url=base_url)
    return model, registry


def make_summarizer(model: Model, registry: Any) -> Callable[[str], Any]:
    """压缩用的摘要函数：用同一个模型起一次性调用，不带工具、不进主历史。"""
    from pi_agent.agent_core.compaction import (
        SUMMARIZATION_PROMPT,
        SUMMARIZATION_SYSTEM_PROMPT,
    )

    async def summarize(transcript: str) -> str:
        print("\n\033[90m[正在压缩上下文…]\033[0m", flush=True)
        message = await complete_simple(
            f"{transcript}\n\n{SUMMARIZATION_PROMPT}",
            model=model,
            registry=registry,
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        )
        text = "\n".join(
            b.text for b in message.content if isinstance(b, TextContent)
        ).strip()
        print("\033[90m[压缩完成]\033[0m", flush=True)
        return text

    return summarize


def build_compaction_settings(context_window: int) -> CompactionSettings:
    """按窗口大小缩放压缩参数。

    默认的 reserve=16384 / keep_recent=20000 是按 128k 窗口定的。直接用在小窗口上
    会出现两个毛病：阈值 `window - reserve` 变负数；且 keep_recent 比整个上下文还大，
    使 find_cut_point 总是短路返回 0（无可摘要的 head）——表现为“永远不压缩”。
    这里按比例取：预留 1/8、保留近期 1/4。
    """
    return CompactionSettings(
        enabled=True,
        reserve_tokens=max(256, context_window // 8),
        keep_recent_tokens=max(256, context_window // 4),
    )


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------


def print_history(agent: Agent) -> None:
    messages = agent.state.messages
    if not messages:
        print("(上下文为空)")
        return
    print(f"--- 当前上下文（{len(messages)} 条）---")
    for i, msg in enumerate(messages):
        role = getattr(msg, "role", "?")
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text = content
        else:
            parts = []
            for b in content:
                if isinstance(b, TextContent):
                    parts.append(b.text)
                elif isinstance(b, ToolCall):
                    parts.append(f"<tool {b.name}>")
            text = " ".join(parts)
        text = text.replace("\n", " ")
        if len(text) > 100:
            text = text[:100] + "…"
        print(f"  [{i}] {role}: {text}")
    print("---")


async def handle_command(
    line: str, agent: Agent, sess: AgentSession, session_id: str, db_path: Path
) -> bool:
    """处理 / 开头的命令。返回 True 表示应退出。"""
    cmd = line.strip().lower()

    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        print(HELP)
    elif cmd == "/history":
        print_history(agent)
    elif cmd == "/session":
        print(f"session id : {session_id}")
        print(f"db 文件    : {db_path}")
        print(f"压缩启用   : {sess.compaction_enabled}")
    elif cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
    elif cmd == "/compact":
        if not sess.compaction_enabled:
            print("压缩未启用。")
        else:
            last = next(
                (
                    m
                    for m in reversed(agent.state.messages)
                    if isinstance(m, AssistantMessage)
                ),
                None,
            )
            if last is None:
                print("还没有可压缩的对话。")
            else:
                did = await sess._maybe_compact(last)
                print("已压缩。" if did else "当前上下文未达压缩阈值。")
    else:
        print(f"未知命令 {cmd}，输入 /help 查看可用命令。")
    return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def pick_session_id(
    repo: SqliteSessionRepo, ctx: Context, args: Any
) -> str | None:
    if args.session:
        return str(args.session)
    if args.resume:
        # list_ids 已按 created_at 倒序（最新在前），直接取第一个。
        ids = await repo.list_ids(ctx)
        if not ids:
            print("没有历史会话，将新建一个。")
            return None
        return ids[0]
    return None


async def run() -> None:
    parser = argparse.ArgumentParser(description="pi-agent 交互式命令行")
    parser.add_argument("--session", help="续聊指定 session id")
    parser.add_argument("--resume", action="store_true", help="续聊最近一次会话")
    parser.add_argument("--list", action="store_true", help="列出所有会话后退出")
    parser.add_argument(
        "--no-compaction", action="store_true", help="关闭上下文自动压缩"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"会话 db 文件（全部会话共用），默认 {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"模型上下文窗口，默认 {DEFAULT_CONTEXT_WINDOW}",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="工具的工作目录，默认当前目录",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="允许读写 --workdir 之外的路径（默认限制在工作目录内）",
    )
    parser.add_argument(
        "--tools",
        choices=("coding", "readonly", "all"),
        default="coding",
        help="工具套餐：coding=读写改跑，readonly=只读探索，all=全部（默认 coding）",
    )
    args = parser.parse_args()

    db_path: Path = args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    repo = SqliteSessionRepo(str(db_path))

    if args.list:
        rows = await repo.list_sessions(ctx)
        if not rows:
            print("（暂无会话）")
        for sid, created_at in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(created_at / 1000))
            print(f"{sid}  {when}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("未找到 OPENAI_API_KEY。请在 .env 里配置（可参考 .env.example）。")
        sys.exit(1)

    model, registry = build_model()
    agent = Agent(stream_fn=create_agent_stream_fn(registry))
    agent.set_model(model)

    workdir: Path = args.workdir.resolve()
    tools = build_tools(workdir, sandbox=not args.no_sandbox, preset=args.tools)
    # 工具提示词由工具自带的元数据拼装，加工具不用回来手改这段。
    agent.set_system_prompt(
        "You are a coding assistant operating in a real filesystem. "
        f"Your working directory is {workdir}.\n\n"
        f"{build_tools_prompt(tools)}"
    )
    agent.set_tools(tools)
    agent.subscribe(make_printer())

    summarize = None if args.no_compaction else make_summarizer(model, registry)
    session_id = await pick_session_id(repo, ctx, args)

    sess = await create_agent_session(
        agent,
        repo,
        ctx,
        session_id=session_id,
        summarize=summarize,
        context_window=None if args.no_compaction else args.context_window,
        compaction_settings=(
            None if args.no_compaction else build_compaction_settings(args.context_window)
        ),
    )
    sid = sess.session.metadata.id

    print(f"\033[1mpi-agent\033[0m · 模型 {model.id} · session {sid}")
    print(f"会话 db: {db_path}")
    restored = len(agent.state.messages)
    if restored:
        print(f"已恢复 {restored} 条历史消息。")
    print("输入问题开始对话，/help 查看命令，/exit 退出。\n")

    try:
        while True:
            try:
                line = input("\033[94m你 >\033[0m ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n（输入 /exit 退出）")
                continue

            if not line:
                continue

            if line.startswith("/"):
                if await handle_command(line, agent, sess, sid, db_path):
                    break
                continue

            try:
                # 走 AgentSession.prompt：落库 + 轮末串行压缩检查
                await sess.prompt(line)
            except KeyboardInterrupt:
                agent.abort()
                print("\n\033[90m[已中断]\033[0m")
            except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底，不让单轮错误杀掉进程
                print(f"\033[91m[出错] {type(exc).__name__}: {exc}\033[0m")
    finally:
        await sess.dispose()
        print(f"\n会话已保存。下次续聊：uv run python main.py --session {sid}")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

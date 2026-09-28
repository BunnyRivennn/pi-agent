"""AgentSession 门面 + create_agent_session 主入口（对齐官方 A4 层）。

把 Agent（agent_core）与 StorageBackedSession（持久化）接线：
- 订阅 Agent 的 message_end → 把新消息 append 到当前 branch（持久化）。
- laneConfig：把 model/thinking_level 存进 value 侧存储，重开时恢复。
- 自动恢复：open 已存在会话时，读回 branch 历史 messages 灌入 Agent 状态。
- dispose：flush 在途落库任务 + 取消订阅 + 关会话。

红线遵守：
- subscribe 回调是同步的，落库是 async → 用 fire-and-forget task 跟踪，dispose flush。
- 加载历史 ≠ 重跑工具：只 replace_messages，不触发 agent_loop。
- 不挂会话时 Agent 行为零变化（本模块是可选装配层）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..agent_core.agent import Agent
from ..agent_core.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    estimate_context_tokens,
    serialize_conversation,
    should_compact,
)
from ..agent_core.types import AgentMessage, AssistantMessage
from .compaction import collect_retained_tail, find_cut_point
from .context import build_session_context
from .errors import CompactionRaceError
from .repo import SessionRepo
from .session import DEFAULT_BRANCH, StorageBackedBranch, StorageBackedSession
from .storage import BranchScan, Context
from .types import Entry
from .values import Value

# laneConfig 侧存储地址：存 model/thinking 等不进 entry 流的运行配置。
_LANE_NAMESPACE = "pi.lane"

# summarize 回调：输入序列化后的会话文本，返回摘要文本。由构造方注入（用同一
# registry/model 起一次性 LLM 调用）。不注入则不启用压缩 → Agent 行为零变化。
Summarize = Callable[[str], Awaitable[str]]


def _lane_config_address(branch: str) -> Value[dict[str, Any]]:
    return Value(namespace=_LANE_NAMESPACE, key=branch)


class AgentSession:
    """Agent + 持久化会话的组合门面。"""

    def __init__(
        self,
        agent: Agent,
        session: StorageBackedSession,
        branch: StorageBackedBranch,
        context: Context,
        *,
        summarize: Summarize | None = None,
        context_window: int | None = None,
        compaction_settings: CompactionSettings | None = None,
    ) -> None:
        self._agent = agent
        self._session = session
        self._branch = branch
        self._context = context
        self._pending: set[asyncio.Task[None]] = set()
        self._disposed = False
        # 压缩：仅当同时注入 summarize 与 context_window 时启用（否则零变化）。
        self._summarize = summarize
        self._context_window = context_window
        self._compaction_settings = compaction_settings or DEFAULT_COMPACTION_SETTINGS
        # 对齐官方 agent-session.ts：事件回调只记录最后一条 assistant 消息，
        # 压缩检查放在 prompt() 主控制流里串行 await（见 _check_compaction）。
        # 「取走即清空」使一轮内多次 LLM 调用只检查一次，结构性去重。
        self._last_assistant: AssistantMessage | None = None
        self._unsubscribe = agent.subscribe(self._on_event)

    @property
    def compaction_enabled(self) -> bool:
        return (
            self._summarize is not None
            and self._context_window is not None
            and self._compaction_settings.enabled
        )

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def session(self) -> StorageBackedSession:
        return self._session

    # -- 事件处理：message_end → 落库 ----------------------------------------

    def _on_event(self, event: dict[str, Any]) -> None:
        if self._disposed:
            return
        if event.get("type") != "message_end":
            return
        message = event.get("message")
        if message is None:
            return
        # 只记录 assistant 消息供压缩检查使用；压缩本身不在这里触发。
        if isinstance(message, AssistantMessage):
            self._last_assistant = message
        task = asyncio.create_task(self._persist_message(message))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _persist_message(self, message: AgentMessage) -> None:
        await self._branch.append_message(message, self._context)

    # -- 对话入口（压缩在此串行编排） ----------------------------------------

    async def prompt(
        self,
        input_value: str | AgentMessage | list[AgentMessage],
        **kwargs: Any,
    ) -> None:
        """发起一轮对话，轮末串行检查压缩（对齐官方 _runAgentPrompt）。

        与直接调 ``agent.prompt()`` 的区别：本方法在 agent 跑完后 flush 落库，
        再在主控制流上 await 压缩检查——无 fire-and-forget，故无并发窗口。
        """
        await self._agent.prompt(input_value, **kwargs)
        await self._agent.wait_for_idle()
        await self.flush()
        await self._check_compaction()

    # -- 压缩编排 ------------------------------------------------------------

    async def _check_compaction(self) -> bool:
        """轮末压缩检查（对齐官方 _checkCompaction）。

        取走并清空 ``_last_assistant``：一轮内 LLM 可能被调用多次（工具循环），
        只对最后一条做检查，天然去重。中止/错误的轮次不触发压缩。
        """
        message = self._last_assistant
        self._last_assistant = None
        if not self.compaction_enabled or self._disposed:
            return False
        if message is None or message.stop_reason in ("aborted", "error"):
            return False
        return await self._maybe_compact(message)

    async def _maybe_compact(self, trigger: AssistantMessage) -> bool:
        """检查上下文是否超阈值，超则算切点、生成摘要、乐观提交 CompactionEntry。

        慢 IO（扫路径、序列化、LLM 摘要）在 mutation 屏障外做；提交时以
        expected_tip 乐观校验，tip 已变（有新消息落库）则本轮放弃（CompactionRaceError）。
        原始 entry 从不删除。
        """
        assert self._summarize is not None
        assert self._context_window is not None
        tip = await self._branch.get_tip_id(self._context)
        if tip is None:
            return False
        path: list[Entry] = await self._session.scan_branch(
            BranchScan(start=tip, order="oldestFirst"), self._context
        )

        # 触发消息比最近一次压缩更老 → 它携带的是压缩前的陈旧上下文规模，
        # 据此压缩会在刚压完后立刻再压一次（死循环）。对齐官方
        # assistantIsFromBeforeCompaction 检查。
        last_compaction_ts = -1
        for entry in reversed(path):
            if entry.type == "compaction":
                last_compaction_ts = entry.timestamp
                break
        if last_compaction_ts >= 0 and trigger.timestamp <= last_compaction_ts:
            return False

        messages = await build_session_context(path, None)
        estimate = estimate_context_tokens(messages)
        if not should_compact(
            estimate.tokens, self._context_window, self._compaction_settings
        ):
            return False

        cut = find_cut_point(
            path, 0, len(path), self._compaction_settings.keep_recent_tokens
        )
        # 切点在最前 → 没有可摘要的历史，跳过（避免摘要空内容/自引用）。
        if cut.first_kept_entry_index <= 0:
            return False

        head = path[: cut.first_kept_entry_index]
        head_messages = await build_session_context(head, None)
        if not head_messages:
            return False
        retained_tail = collect_retained_tail(
            path, cut.first_kept_entry_index, len(path)
        )

        transcript = serialize_conversation(head_messages)
        summary = await self._summarize(transcript)

        try:
            await self._session.append_compaction_to_branch(
                self._branch.name,
                summary=summary,
                retained_tail=retained_tail,
                tokens_before=estimate.tokens,
                expected_tip=tip,
                context=self._context,
            )
        except CompactionRaceError:
            # tip 在生成摘要期间被推进：良性竞态，下一轮末重试。
            return False

        # 压缩后重建 Agent 上下文，使后续轮次从摘要处继续。
        new_tip = await self._branch.get_tip_id(self._context)
        if new_tip is not None:
            new_path = await self._session.scan_branch(
                BranchScan(start=new_tip, order="oldestFirst"), self._context
            )
            rebuilt = await build_session_context(new_path, None)
            self._agent.replace_messages(rebuilt)
        return True

    # -- 配置持久化（laneConfig） --------------------------------------------

    async def save_lane_config(self) -> None:
        state = self._agent.state
        config: dict[str, Any] = {
            "model_id": state.model.id,
            "model_api": state.model.api,
            "model_provider": state.model.provider,
            "thinking_level": state.thinking_level,
            "system_prompt": state.system_prompt,
        }
        await self._session.set_value(
            _lane_config_address(self._branch.name), config, self._context
        )

    # -- lifecycle -----------------------------------------------------------

    async def flush(self) -> None:
        """等待所有在途落库任务完成。

        即时快照并清空，避免 done_callback 的 discard 尚未被事件循环调度时
        while 空转；await 后再检查期间新生成的任务。
        """
        while self._pending:
            pending = list(self._pending)
            self._pending.clear()
            await asyncio.gather(*pending, return_exceptions=True)

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._unsubscribe()
        await self.flush()
        await self._session.close(self._context)


async def _restore_lane_config(
    agent: Agent, session: StorageBackedSession, branch_name: str, context: Context
) -> None:
    stored = await session.get_value(_lane_config_address(branch_name), context)
    if stored is None or not isinstance(stored.value, dict):
        return
    config = stored.value
    prompt = config.get("system_prompt")
    if isinstance(prompt, str):
        agent.set_system_prompt(prompt)
    thinking = config.get("thinking_level")
    if isinstance(thinking, str):
        agent.set_thinking_level(thinking)  # type: ignore[arg-type]


async def _restore_messages(
    agent: Agent, session: StorageBackedSession, branch: StorageBackedBranch, context: Context
) -> int:
    tip = await branch.get_tip_id(context)
    if tip is None:
        return 0
    path = await session.scan_branch(
        BranchScan(start=tip, order="oldestFirst"), context
    )
    messages = await build_session_context(path, None)
    if messages:
        agent.replace_messages(messages)
    return len(messages)


async def create_agent_session(
    agent: Agent,
    repo: SessionRepo,
    context: Context,
    session_id: str | None = None,
    branch_name: str = DEFAULT_BRANCH,
    *,
    summarize: Summarize | None = None,
    context_window: int | None = None,
    compaction_settings: CompactionSettings | None = None,
) -> AgentSession:
    """组合 Agent + 持久化会话。

    - session_id 为 None → 新建会话。
    - session_id 指向已存在会话 → 打开并自动恢复（历史消息 + laneConfig）。
    - 传入 summarize + context_window → 启用短期记忆压缩（每轮末自动检查）。
    """
    session: StorageBackedSession | None = None
    if session_id is not None:
        session = await repo.open(session_id, context)
    if session is None:
        session = await repo.create(context, session_id=session_id)

    branch = await session.branch(branch_name, context)
    if branch is None:
        branch = await session.create_branch(branch_name, None, context)

    # 自动恢复：历史消息 + 运行配置。
    await _restore_lane_config(agent, session, branch_name, context)
    await _restore_messages(agent, session, branch, context)

    agent_session = AgentSession(
        agent,
        session,
        branch,
        context,
        summarize=summarize,
        context_window=context_window,
        compaction_settings=compaction_settings,
    )
    await agent_session.save_lane_config()
    return agent_session


__all__ = ["AgentSession", "Summarize", "create_agent_session"]

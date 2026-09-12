# PLAN3：Checkpointer（会话持久化，SQLite 先行）

> 状态：**规划中，未施工**。对应 PLAN.md Phase 3 的 session persistence 项。
>
> **第一版刻意做小：把消息存进 SQLite，进程重启后能读回历史继续聊——先跑起来。**
> 断点续跑（崩在工具之间接着走）、分叉树、AgentSession 门面都留到后续切片，见 §9。
>
> **存储选型：用 SQLite 代替 PLAN.md 原话的 JSONL 文件**。理由：
> - sqlite3 是 Python 标准库，**零新依赖**；
> - SQL 接口与未来的 PG 后端同构——换 PG 时只换连接和方言，应用层接口不变；
> - 原子事务（assistant + tool results 要么全存要么全不存）、并发读写、损坏诊断、按时间查询，都比手写 append-only 文件可靠；
> - 代价（建表 schema、cursor 管理）对长期是赚的。

---

## 1. 背景

当前 `Agent` 是**有状态但无记忆**的：

- 对话历史在内存 `Agent._state.messages`（[agent.py](src/pi_agent/agent_core/agent.py) 的 `AgentState.messages`），进程结束即丢。
- `Agent(initial_state=...)`、`replace_messages()` 是现成的状态注入点，但没有代码把状态"存下来再装回来"。
- `session_id` 已从 `Agent` → `AgentLoopConfig` → `PiAIRequest` → provider 一路透传，唯一用途是 OpenAI 请求的 `metadata`（[openai.py:905](src/pi_agent/pi_ai/providers/openai.py#L905)），**没有任何 save/load 逻辑**。

第一版目标一句话：**挂载 checkpointer 的 Agent，每回合消息自动落 SQLite；新进程用同一 session_id 能取回完整历史继续对话。**

## 2. 目标与非目标

### 目标（第一版）

1. 消息 dataclass 的序列化层：dataclass ⇄ dict roundtrip（含 ToolCall / ToolResult / Usage 等嵌套结构）。
2. 数据库中立的 **Checkpointer 接口**：`save_messages` / `load_messages` / `list_sessions`。
3. 两个实现：
   - `InMemoryCheckpointer`：dict 存储，测试用；
   - **`SqliteCheckpointer`**：标准库 sqlite3，标准库够用就不引第三方；async 接口内部用 `asyncio.to_thread` 包同步调用，不阻塞 event loop。
4. 消息级 `id` / `parent_id` 由存储层分配（不改 agent_core dataclass），为未来分叉留好字段；第一版只走线性链。
5. Agent 可选挂载，不挂载时零影响（现有 75 个测试行为不变）。
6. 一个恢复入口 + 一个免 key demo：新进程重建 Agent 后继续对话，模型收到的上下文与"从未重启"一致。

### 非目标（第一版明确不做，排期见 §9）

- ❌ **断点续跑**：崩在 turn 中途（LLM 调用中、工具执行中/之间）不保证接着走——第一版只在 **turn 边界**落盘，崩溃时当前 turn 整体丢失，重来。续跑是下一切片，思路已验证（§9.1）。
- ❌ **分叉树语义**：schema 里有 `parent_id` 字段，但不做"从历史某条消息 fork 新分支"的 API；第一版 tip 永远是最新消息。
- ❌ **AgentSession 门面**（`create_agent_session`，Phase 3①）：本版只提供"存/取消息"的最小能力。
- ❌ PG 后端（接口预留，另案）。
- ❌ 模型配置（system_prompt/model/tools）持久化：tools 是代码对象不可序列化；恢复时由调用方重新构造 Agent，checkpointer 只回填消息（见 §8.1）。
- ❌ 持久化 `AgentState` 易失字段（`is_streaming` / `stream_message` / `pending_tool_calls: set` / `error`）——运行时瞬态。
- ❌ 自定义 Mapping 消息（`CustomAgentMessage`）：`default_convert_to_llm` 本就过滤它们；遇到时跳过并计入 `skipped` 计数，可观测，不静默也不中断。
- ❌ 不改 provider、不改事件协议、不引第三方依赖。

## 3. 现状关键事实（设计约束）

来自代码调研（行号基于 2026-09）：

1. 消息全部是 `@dataclass(slots=True)`，在 [types.py](src/pi_agent/agent_core/types.py)：
   - `UserMessage(content: str | list[TextContent|ImageContent], timestamp, role="user")`
   - `AssistantMessage(content: list[Text|Thinking|ToolCall], api, provider, model, usage: Usage, stop_reason, error_message, timestamp, role="assistant")`
   - `ToolResultMessage(tool_call_id, tool_name, content: list[Text|Image], is_error, details, timestamp, role="toolResult")`
   - content block：`TextContent / ThinkingContent / ImageContent / ToolCall`，都带 `type` 字面量标签——天然的多态反序列化判别字段。
2. **消息没有 id**，只有 `timestamp: int`（epoch ms）。配对键是 `ToolCall.id` ↔ `ToolResultMessage.tool_call_id`，原样进 payload，回放后配对自然成立。
3. 消息顺序（[agent_loop.py](src/pi_agent/agent_core/agent_loop.py)）：时间序平铺；AssistantMessage 后紧跟各条 ToolResultMessage；`LlmContext.messages` 就是这个扁平 list。
4. **写入时机（第一版，turn 边界）**：`Agent._execute` 消费事件循环，`turn_end` 事件的 payload 同时带 `message`（AssistantMessage）和 `tool_results`（list[ToolResultMessage]）——一次事务批量写入；user 消息在 `prompt()` 入口写（保证崩溃也不丢用户输入，留下"问了没答"是正确行为）。
5. **恢复路径现成**：`Agent(initial_state=AgentState(system_prompt=..., model=..., tools=..., messages=[...]))`；`_clone_state` 浅克隆即可。
6. 运行依赖仅 `jsonschema` + `python-dotenv`；Python ≥ 3.11；mypy strict；pytest + `asyncio_mode="auto"`，无 conftest、无 fixture 文化，用模块级工厂函数 + 手写 fake stream。

## 4. 概念模型

### 4.1 分层

```
┌──────────────────────────────────────────────┐
│ Agent（已有）：turn_end 时批量写 / prompt 写 user │
├──────────────────┬───────────────────────────┤
│ 序列化层 serialize │ Checkpointer 接口（DB 中立） │
│ dataclass ⇄ dict  │ save/load/list            │
├──────────────────┴───────────────────────────┤
│ InMemoryCheckpointer（测试）  SqliteCheckpointer │
│                               （标准库 sqlite3）  │
└──────────────────────────────────────────────┘
         未来：PostgresCheckpointer（实现同一接口）
```

### 4.2 存什么：消息记录

一条消息 = 一行记录，按 turn 批量、按 session 线性追加：

| 列 | 内容 |
|---|---|
| `id` | 存储层分配，`msg_<ms>-<rand6>`，时间序肉眼可读 |
| `session_id` | Agent 已有的 session_id（没有则首次写入时生成 `sess_<uuid8>`） |
| `parent_id` | 上一条消息 id；首条为 NULL（**字段先在，分叉后用**） |
| `seq` | 会话内自增序号，回放顺序的权威依据（不依赖毫秒时间戳碰撞） |
| `timestamp` | 消息自带 epoch ms |
| `role` | `user` / `assistant` / `toolResult`（冗余列，方便查询/调试） |
| `payload` | 消息 dataclass 按 type 标签序列化的**完整 JSON 文本** |

一个事务写一个 turn：`[assistant, tool_result...]` 要么全进要么全不进，不会存出"有调用无结果"的半截 turn。

### 4.3 恢复 = 按 seq 回放

`load_messages(session_id)` → `SELECT payload FROM messages WHERE session_id=? ORDER BY seq` → 逐条 `message_from_dict` → 回填 `AgentState.messages`。没有快照、没有游标、没有额外状态。

### 4.4 失败语义（红线，沿用 JSON 容错的教训）

- payload 反序列化失败、未知 `version`/`type`：**显式抛 `CheckpointCorruptError`**（带 session_id 和 seq），绝不悄悄截断历史让 Agent"看似恢复成功"。
- 写库失败向上抛，不假装已持久化。
- load 到的消息引用的工具将来恢复时不存在——第一版不检查（调用方负责用相同工具集重建 Agent）；续跑切片再做显式校验。

## 5. SQLite schema

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,          -- sess_xxx
    created_at  INTEGER NOT NULL,          -- epoch ms
    tip_seq     INTEGER NOT NULL DEFAULT 0 -- 最新消息 seq（线性版即 COUNT，冗余留作快速 tip 查询）
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,          -- msg_xxx
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    parent_id   TEXT,
    seq         INTEGER NOT NULL,
    timestamp   INTEGER NOT NULL,
    role        TEXT NOT NULL,
    payload     TEXT NOT NULL,             -- 完整消息 JSON
    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages (session_id, seq);
```

- DB 文件默认 `.pi_agent/sessions.db`（单库多 session；也支持 `:memory:` 供测试）。
- 连接开启 `PRAGMA foreign_keys=ON`、`PRAGMA journal_mode=WAL`（读写并发更稳）、`PRAGMA busy_timeout=5000`。
- payload JSON 自带 `version: 1`，schema 演进靠版本号 + `message_from_dict` 的显式分支。

## 6. 接口草图

```python
# src/pi_agent/session/types.py

@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    parent_id: str | None
    seq: int
    session_id: str
    timestamp: int
    role: str                 # "user" | "assistant" | "toolResult"
    message: AgentMessage     # 还原后的 dataclass

@dataclass(frozen=True, slots=True)
class SessionInfo:
    id: str
    created_at: int
    message_count: int

@runtime_checkable
class Checkpointer(Protocol):
    async def save_messages(
        self, session_id: str, messages: list[AgentMessage]
    ) -> list[str]:
        """一个 turn 的消息原子追加（user 单条，或 assistant + tool results）。
        session 不存在则自动创建。返回分配的消息 id（按入参顺序）。"""

    async def load_messages(self, session_id: str) -> list[StoredMessage]:
        """按 seq 回放全部消息；session 不存在返回 []；
        记录损坏抛 CheckpointCorruptError（带 seq）。"""

    async def list_sessions(self) -> list[SessionInfo]: ...

class CheckpointError(Exception): ...
class CheckpointCorruptError(CheckpointError): ...   # 含 session_id / seq
```

序列化层是独立纯函数，两后端共用：

```python
# src/pi_agent/session/serialize.py
def message_to_dict(message: AgentMessage) -> dict[str, Any]: ...
def message_from_dict(payload: dict[str, Any]) -> AgentMessage:
    """按 role + content[].type 标签还原具体 dataclass。
    未知 type/version → 抛 UnknownRecordVersion，不猜、不丢。"""
```

Agent 挂载（最小侵入，构造参数可选）：

```python
Agent(
    ...,
    checkpointer=checkpointer or None,   # None = 现状行为
    # session_id 已存在；checkpointer 首次写入时若为空则自动生成
)
```

恢复辅助（不做门面，只提供数据取回；Agent 重建仍由调用方完成）：

```python
# src/pi_agent/session/restore.py
async def restore_messages(checkpointer: Checkpointer, session_id: str) -> list[AgentMessage]:
    return [stored.message for stored in await checkpointer.load_messages(session_id)]
```

## 7. 文件框架

```
src/pi_agent/session/                    # 新包，与 agent_core/、pi_ai/ 平级
├── __init__.py                          # Checkpointer / InMemory / Sqlite / 异常
├── types.py                             # StoredMessage、SessionInfo、Checkpointer Protocol、异常
├── ids.py                               # msg_/sess_ id 生成（纯函数，易测）
├── serialize.py                         # dataclass ⇄ dict roundtrip（按 type 标签多态）
├── memory.py                            # InMemoryCheckpointer：dict 存记录，异步接口
├── sqlite.py                            # SqliteCheckpointer：建表、事务追加、按 seq 回放
└── restore.py                           # restore_messages 辅助函数

src/pi_agent/agent_core/
├── agent.py                             # 改：__init__ 加可选 checkpointer；prompt 入口写 user；
│                                        #     _execute 事件循环 turn_end 批量写
└── types.py                             # AgentLoopConfig 不动（第一版写入点在 Agent 层，不进 loop）

tests/session/                           # 沿用现有风格：无 fixture、模块级工厂、中文 id、-s 可观察
├── test_serialize.py                    # 三种消息 × 各 block 的 roundtrip、未知 type/version 显式失败
├── test_ids.py                          # id 唯一/前缀/时间序
├── test_memory_checkpointer.py          # save/load/seq 顺序/自动建 session/批量原子性
├── test_sqlite_checkpointer.py          # tmp_path 真 SQLite：写入→新建实例（模拟重启）回放、
│                                        #   :memory: 模式、损坏 payload 报错带 seq、WAL/外键 pragma
└── test_agent_checkpointing.py          # mock provider 两圈工具 loop 落库 → 新 Agent 实例恢复 →
                                         #   再聊一轮，断言 provider 收到的消息序列与不重启一致；
                                         #   不挂载 checkpointer 时行为零变化

examples/
└── sqlite_session_demo.py               # mock 免 key：第一轮对话 → 退出 → 新 agent 同库恢复 →
                                         #   "我刚才说了什么？" 验证历史在场
```

不动：`agent_loop.py`、event_stream、provider、事件协议、`AgentState` 字段。

## 8. 开放问题（施工时逐个定）

1. **模型配置恢复**：第一版 checkpointer 只存消息；调用方用相同 system_prompt/model/tools 重新构造 Agent，再经 `restore_messages` 回填。demo 里演示完整写法；将来 AgentSession + settings/registry（Phase 3③）再收编配置。
2. **session_id 自动生成位置**：Agent 挂载 checkpointer 且没传 session_id 时，首次 `save_messages` 由后端生成并回传，Agent 更新自身 session_id property。需注意并发 prompt（现有代码本就禁止流式中 prompt，问题不大）。
3. **DB 文件路径配置**：构造 `SqliteCheckpointer(path=...)` 必传路径，默认值放在哪（环境变量 `PI_AGENT_DB`？还是只给 demo 用 `.pi_agent/sessions.db`）——倾向必传，不搞全局默认。
4. **ImageContent.data 可能很大**：第一版照存（SQLite TEXT 无实际压力），未来再评估外置 blob 存储。

## 9. 后续演进（本版跑起来后再一点点加）

### 9.1 下一步：断点续跑（思路已验证，不需要 LangGraph 式快照）

LangGraph 显式存 `pending_writes`，是因为通用 DAG 的节点输出不一定是消息。**pi-agent 的工具结果本身就是消息**，所以"还欠哪些工具"可以从消息日志纯集合差推导：

```
missing = 末条 AssistantMessage 的 toolcall 集合
        − 其后已配对的 ToolResultMessage 集合（按 tool_call_id）
```

| 崩溃位置 | 末条消息 | 恢复动作 |
|---|---|---|
| user 存了、LLM 没跑完 | UserMessage | 走现有 `agent_loop_continue` 调 LLM |
| LLM 完成、工具 0 执行 | Assistant（0 result） | 执行全部 missing 工具，**不重调 LLM** |
| 工具 1、2 之间 | …1 result | 只执行剩余 missing |
| 工具全完、下轮 LLM 前 | result 配齐（missing=0） | 走 `agent_loop_continue` |
| 正常 stop | 无 toolcall | 空闲等输入 |

前置改动只有两个：① 写入粒度从 turn 收紧到节点（`message_end` 写 assistant、每个 `tool_execution_end` 立即写 result）；② 新增 `resume()` 推导入口。**不需要 phase 字段、不需要 checkpoint 表、不需要快照状态机，消息日志是唯一事实来源。**

边界（躲不掉，LangGraph 也一样）：工具执行中途崩 → 该工具重跑（write/bash 类靠工具自身幂等：临时文件 rename、先查后写）；LLM 流式中崩 → 重调一次 LLM。

### 9.2 再往后

- **分叉树**：schema 的 `parent_id` 已就位；加 tip 概念 + `load_messages(tip=...)` + 写入时显式 parent，即可从历史消息 fork。
- **PG 后端**：`PostgresCheckpointer` 实现同一 Protocol（asyncpg），SQL 几乎直译；序列化层和接口零改动——这就是选 SQLite 不选 JSONL 的核心原因。
- **AgentSession / `create_agent_session`**（PLAN Phase 3①）：门面负责模型配置存取、一行 resume。
- **HITL 审批**：在 missing 工具上加"等待审批"标记 + resume value，是 9.1 的自然延伸。
- **sub-agent 隔离**：session_id 加命名空间前缀。

## 10. 分阶段施工计划

每阶段独立提交、独立验收；叫停不留下半成品行为。

| Slice | 内容 | 验收 | 依赖 |
|---|---|---|---|
| **0** | `ids.py` + `serialize.py` 纯函数 | 三消息 + 全 block/Usage/ToolCall 嵌套 roundtrip；未知 type/version 显式报错；TDD | 无 |
| **1** | `types.py` 接口 + `InMemoryCheckpointer` | save/load、seq 顺序、自动建 session、批量原子、空 session=[] | 0 |
| **2** | `SqliteCheckpointer` | tmp_path 落盘；新建实例读同库（模拟重启）回放一致；外键/WAL pragma；损坏 payload 报 `CheckpointCorruptError` 带 seq；mypy strict | 1 |
| **3** | Agent 挂载 + 恢复路径：prompt 写 user、turn_end 批量写、`restore_messages` | mock 两圈工具 loop 落库 → 新 Agent 恢复再聊一轮，provider 收到的消息与不重启逐序一致；不挂载时现有 75 测试零变化 | 2 |
| **4** | `examples/sqlite_session_demo.py` + 文档片段（如何挂 checkpointer、如何恢复） | 免 key 手工跑通"退出-重启-续聊" | 3 |
| **未来** | §9.1 断点续跑 → 分叉 → PG → AgentSession → HITL | — | — |

## 11. 测试约定

- pytest，async 测试显式 `@pytest.mark.asyncio`；不用 fixture，用模块级工厂（参考 [test_agent_loop.py](tests/agent_core/test_agent_loop.py) 的 `make_model` / `make_stream`）。
- 模型侧一律 mock provider / 手写 fake stream，不打真实 API。
- SQLite 测试用 pytest 内置 `tmp_path` 建库文件；"重启"用"新建 checkpointer 实例指同一路径"模拟。
- 参数化用例中文 id，关键断言前 `print()` 输入/输出，`pytest -s` 能看清测了什么。
- 每 Slice 三连：`uv run pytest tests/ -q` + `uv run ruff check` + `uv run mypy`（strict）。

## 12. 与 PLAN.md 的对应与偏离

- 对应 Phase 3「session persistence with tree semantics (id/parentId)」的**持久化本体**；树语义只保留 schema 字段，分叉 API 推后（§9.2）。
- **有意偏离**：存储后端从 PLAN.md 字面的 JSONL 换成 SQLite——零新增依赖、与未来 PG 同构、事务和并发更可靠；`create_agent_session`（第①项）在 Slice 4 之后。
- Settings/auth/model registry（第③项）、内置 coding 工具（第④项）与本计划解耦，可并行。

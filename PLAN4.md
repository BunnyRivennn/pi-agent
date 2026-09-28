# PLAN4：记忆系统统一路线图（对齐官方架构 B：StorageBackedSession + SessionRepo，SQLite 后端）

> 状态：**施工中**。持久化轨道 A0–A5 已完成并验收；短期记忆/压缩 D1–D2 已完成（三连通过）。待做：B0（hook 总线）、C1–C2（Skill）、E1（长期记忆）、F（断点续跑，后期）。本文已用官方 TS 源码（`docs/pi-main`）逐条核对，取代前两版基于 skill 文档转述的设计。
>
> **对齐决策**：官方存在两套并存会话架构，本项目对齐**新架构 B**（`packages/agent/src/harness/session/`），非老的 `SessionManager`（架构 A）。存储后端用 **SQLite**（实现架构 B 的 `Storage` 接口）。
>
> 核对来源（官方源码，行号 2026-xx）：
> - `packages/agent/src/harness/session/types.ts`（Entry/Storage/Session/SessionRepo 契约）
> - `packages/agent/src/harness/session/values.ts`（KV+List 侧存储）
> - `packages/agent/src/harness/session/memory.ts` / `session.ts`（MemoryStorage / StorageBackedSession）
> - `packages/agent/src/harness/compaction/{compaction,utils,branch-summarization}.ts`
> - `packages/coding-agent/src/core/{sdk,agent-session,session-manager}.ts`（架构 A + 门面参照）

---

## 0. 一句话结论

对齐官方**架构 B 的存储契约层**：`Entry`(4 种) + `Storage` 事务接口 + `Session`/`SessionRepo` + `values/lists` 侧存储 + `StorageBackedSession`，用 `SqliteStorage implements Storage` 落地。其上叠加 `AgentSession` 门面、compaction（retainedTail 内联模型）、长期记忆、Skill。

**明确切分**：架构 B 还捆了一台 **13-leaf 可断点续跑操作状态机**（`OperationState` + drive/lane/reconcile/effect-gate）。那是崩溃续跑引擎，**单列为后期轨道 F，不进第一版**。第一版只在 turn 边界持久化。

---

## 1. 架构 B 的会话数据模型（对齐目标，全部来自官方源码）

### 1.1 Entry：只有 4 种（types.ts:16-64）

```
EntryType = "message" | "compaction" | "branch_summary" | "custom"

EntryBase { id, parentId: string|null, seq: number, timestamp: number, type, customType? }
MessageEntry     : { message: AgentMessage, terminate?: true }
CompactionEntry  : { summary, retainedTail: AgentMessage[], tokensBefore, details?, usage?, fromHook }
BranchSummaryEntry: { fromId: string|null, summary, details?, usage?, fromHook }
CustomEntry      : { customType, data?: JsonValue }
```

- **seq/timestamp 由存储层分配**：`NewEntry = Omit<Entry,"seq"|"timestamp">`。
- **compaction 用 `retainedTail` 内联**保留近期消息（自包含），**不用架构 A 的 `firstKeptEntryId` 跨条查找**——移植更简单。
- **model/thinking/name/label/usage 不是 entry**——见 §1.3 KV 侧存储。
- `CustomEntry` 靠 `EntryProjector(entry, context) → AgentMessage[]|undefined` 决定是否进 LLM 上下文（对应架构 A 的 custom vs custom_message 之分）。

### 1.2 Storage 接口（types.ts:455-471）—— SQLite 要实现的契约

```
Storage:
  commit(writes: Write[], ctx): Promise<CommitResult>        # 事务化批量写（entry + value + list）
  getEntries(ids[], ctx): Promise<Map<id,Entry>>
  getValue<T>(addr, ctx) / scanValues<T>(prefix, ctx)
  readList<T>(addr, opts, ctx): Promise<ListElement<T>[]>
  scanBranch(query{start,...}, ctx): Promise<Entry[]>        # 从某 entry 沿 parentId 上溯
  scanBranchStructure(query, ctx): Promise<EntryStructure[]>
  scanEntries(query{fromSeq,toSeq,order,limit}, ctx): Promise<Entry[]>
  scanUsage(query, ctx): Promise<UsageRow[]>
  getStats(ctx): Promise<SessionStats{messageCount, usage}>
  close(ctx)
```

`Write` 是联合类型：entry 写 + `ValueWrite`(set/delete) + `ListWrite`(append/delete)。一次 `commit` 原子落多种写——天然映射 SQLite 事务。

### 1.3 values + lists 侧存储（values.ts）—— 配置/元数据的家

类型化地址：`value<T>(namespace, key)` / `list<T>(namespace, key)`。官方用它存：

| 地址 | 存什么 | 我们第一版是否需要 |
|---|---|---|
| `laneConfig(lane) = value<LaneConfiguration>("pi.lane.config",lane)` | **{model, thinkingLevel, activeToolNames}** | ✅ 配置恢复的核心——读一个 value，不回放 entry |
| `sessionName = value<string>("pi.session.name")` | 会话显示名 | ✅ 轻量 |
| `entryLabel(id) = value<string>("pi.entry.label",id)` | 书签/标注 | 可后置 |
| `branchTip(branch) = value<string|null>("pi.branch.tip",branch)` | 分支 tip 指针 | ✅ 分叉需要 |
| `operationMeta/State/...`、`pending*` | **续跑引擎的运行态** | ❌ 属轨道 F，第一版不碰 |

**关键差异（对上一版 PLAN4 的修正）**：架构 A 靠回放 `model_change`/`thinking_level_change` entry 恢复配置；**架构 B 直接读 `laneConfig` value**。我们对齐 B，配置恢复走 KV。

### 1.4 Session / SessionRepo（types.ts:530-602）

```
Session(extends SessionReader):
  metadata, idGenerator
  getEntry(id) / findEntries(query) / findEntry(query) / getStats / getName / getLabel
  branch(name) / createBranch(name, at) → Branch
  beginMutation() → SessionMutation      # 独占互斥写屏障：commit 恰好 0/1 次，end 释放
  mutate(cb)                             # 受托独占回调，单次 commit
  setValue/deleteValue/appendList/deleteList/setName/setLabel
  close()

Branch: name, getTipId(), findEntries/findEntry, appendMessage(msg)→id, appendCustomEntry(type,data)→id

SessionRepo:
  create(opts) / open(metadata) / list(opts) / delete(metadata) / fork(source, ForkOptions)
```

- **写走 mutation 屏障**（`beginMutation`/`mutate`）保证单会话写串行化 + 事务。
- **branch 是一等公民**：`createBranch(name, at)` + `Branch.appendMessage`；tip 存在 `branchTip` value。
- `SessionRepo` 是"多会话目录/仓库"层，对齐 `MemorySessionRepo` / `JsonlSessionRepo`——我们加 `SqliteSessionRepo`。

### 1.5 build context（compaction-aware，session/context.ts）

`buildContextEntries`：沿 branch tip 上溯，找**最后一个 compaction**，输出 `[compaction, ...tip之后的 entry]`；compaction 投影为 `[compactionSummaryMessage, ...retainedTail.filter(isContextMessage)]`（`isContextMessage` 过滤 error/aborted/deferred 的 assistant）。配置（model/thinkingLevel）从 `laneConfig` value 读，**不从 path 推**。

---

## 2. 对上一版 PLAN4 的修正（已用源码证据钉死）

| # | 上一版写的 | 官方实际 | 证据 |
|---|---|---|---|
| 1 | entry 9 种 | 架构 A 是 **10 种**（漏 `usage`）；**架构 B 是 4 种** | session-manager.ts:165 / types.ts:16 |
| 2 | turn_end 批量写 | **message_end 逐条 appendMessage**；turn_end 只 flush 延迟 custom | agent-session.ts:707-757 |
| 3 | "序列化失败显式抛异常"是"官方共识" | **官方静默跳过损坏行**（`parseSessionEntries`），只校验 header。fail-loud 是 PLAN3 主张，非官方 | session-manager.ts:321-336 |
| 4 | "首条 assistant 才落盘→丢用户输入"是有意差异 | 官方**缓冲含 user 的全部 entry，首条 assistant 到达一次性 flush**，user 不丢 | session-manager.ts:1071-1098 |
| 5 | 配置恢复靠回放 model_change entry | 架构 B 读 **`laneConfig` value** | values.ts:160 |
| 6 | 压缩用 firstKeptEntryId | 架构 B 用 **retainedTail 内联** | types.ts:36 |

**保留正确**：AgentSession 组合（非继承）Agent、靠订阅持久化、Agent 对持久化零耦合；`compact()` 后用 build context 重写 messages；`reserveTokens=16384`/`keepRecentTokens=20000`/`TOOL_RESULT_MAX_CHARS=2000`/图片 4800/char4 估算。

---

## 3. 依赖关系图（DAG）

```
        ┌──────────────────────────────────────────────────────┐
        │ A 会话持久化（架构B契约, SQLite）                       │
        │ serialize(4 entry)+ids → Storage接口+SqliteStorage →    │
        │ StorageBackedSession+SessionRepo+values/lists →         │
        │ build_context → AgentSession门面+create_agent_session   │
        └──────┬──────────────────────────────────┬──────────────┘
               │                                   │
     ┌─────────┘                                   └─────────┐
     ▼                                                       ▼
┌────────────────────┐                          ┌────────────────────────┐
│ D 短期记忆/压缩      │                          │ F 断点续跑引擎(后期,大)  │
│ D1 摘要引擎(retained)│                          │ 13-leaf OperationState + │
│ D2 append_compaction │                          │ drive/lane/reconcile/    │
│  +build_context 处理 │                          │ effect-gate + pending*   │
└─────────┬──────────┘                          │ (PLAN3 §9.1 续跑)        │
          │ (summary 是长期记忆原料)              └────────────────────────┘
          ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ B hook 总线 M0               │───▶│ E 长期记忆外挂 M5             │
│ (event_stream 统一扩展点)    │    │ MemoryStore/Extractor/Injector│
└─────────────────────────────┘    └─────────────────────────────┘

┌─────────────────────────────┐
│ C Skill 渐进加载 M1/M2       │  ← 与 A/B 解耦
└─────────────────────────────┘
```

依赖一句话：A 无前置、立即开工、是 D2/E1 底座（分叉、门面、配置恢复在 A 线内建）；B 可与 A 并行、是 E1 前提；D2 依赖 A+B；E1 依赖 B+A(+D)；C 独立；**F（续跑引擎）依赖 A，独立大工程，最后做**。

---

## 4. 统一里程碑表

验收三连：`uv run pytest -q` + `uv run ruff check` + `uv run mypy`（strict）。

图例：✔ 已完成并三连验收通过 ｜ ⏳ 进行中 ｜ ☐ 待办

| 阶段 | 状态 | 轨道 | 内容 | 依赖 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| **A0** | ✔ | 持久化 | `types.py`（4 种 Entry + EntryBase + NewEntry）+ `serialize.py`（4 种 Entry + AgentMessage 全 block ⇄ dict）+ `ids.py`（session UUIDv7 + entry id）+ `values.py`（Value/ValueList 类型化地址、Write 联合）+ `errors.py` | 无 | `src/pi_agent/session/` | 全 entry+消息嵌套 roundtrip；NewEntry 无 seq/timestamp；未知 type 显式抛异常 |
| **A1** | ✔ | 持久化 | `Storage` 接口（Protocol）+ 完整 `Write`（entry/usage/value/list）/`CommitResult`/`UsageRow`/`SessionStats`/scans 类型 + `commit.py`（seq 分配 + 校验）+ `InMemoryStorage`（commit/getEntries/scanBranch/scanEntries/getValue/readList/scanUsage/getStats） | A0 | 同上 | 事务 commit 原子性 + 校验（非单调/重复 id/缺父）；scanBranch 沿 parentId 上溯；value set/delete + list append 保序 |
| **A2** | ✔ | 持久化 | `sqlite.py`：`SqliteStorage implements Storage`，表 `entries`/`values_kv`/`lists_kv`/`usage`/`session_stats`；事务化 commit（复用 prepare/validate，SQL EXISTS 校验视图，事务内更新聚合行）；scanBranch 迭代上溯；WAL/外键；SQL 直查不物化内存 | A1 | 同上 | tmp_path 落盘→重开回放/seq/配置/统计一致；重复 id/缺父校验；mypy strict |
| **A3** | ✔ | 持久化 | `session.py`（MutationLine 串行屏障 / mutate / SessionMutation / Branch / createBranch / setValue/appendList / 错误类）+ `context.py`（压缩感知 build_context）+ `repo.py`（SessionRepo Protocol + InMemorySessionRepo + SqliteSessionRepo，每会话一个 `.db`） | A2 | 同上 | mutation 串行不交错；appendToBranch 原子提交 entry+tip；build_context 从最后 compaction 截断、跳过 error/aborted；repo 落盘重开续读 |
| **A4** | ✔ | 持久化 | `factory.py`：`AgentSession` 门面 + `create_agent_session`：订阅 message_end→append（同步回调 + 后台 task，`flush()` 排空）；laneConfig 存取恢复 system_prompt/thinking；重开自动灌回历史；`dispose()` | A3 | `session/factory.py` | mock 工具 loop 落库→重开续聊逐条一致；恢复不重跑工具；不挂会话现有测试零变化 |
| **A5** | ✔ | 持久化 | `examples/session_demo.py`（免 key 跑通"退出-重启-续聊"，真落盘 SQLite）+ README 文档（Package layout + 接线示例） | A4 | `examples/` + `README.md` | 实跑验证：4 条落库→全新 Agent 恢复 4 条+system_prompt→续聊累计 6 条、tip 前进 |
| **B0** | ☐ | hook 总线 | `event_stream` 升级：生命周期事件 + 多 handler 按序派发 + fire-and-forget | 无（并行） | `agent_core/event_stream.py` | handler await 不阻塞；agent_settled 每 prompt 一次 |
| **C1** | ☐ | Skill | Skill dataclass + frontmatter 解析 + discover + format_for_prompt | 无 | `agent_core/skills.py` | 空 desc 不加载；索引无正文 |
| **C2** | ☐ | Skill | `read_skill` 工具（白名单）+ system prompt 索引区 | C1 | skills.py+agent.py | mock 读 skill 后回答 |
| **D1** | ✔ | 短期记忆 | token 估算（CJK 感知，官方均一 chars/4 会低估中文）+ should_compact + serialize_conversation(2000 截断) + 结构化摘要 prompt。摘要 LLM 调用改为**注入式** `summarize` 回调（`Model` 无 context_window，注入 context_window） | 无（核心并行） | `agent_core/compaction.py` | 超长会话触发；不切 toolResult；retainedTail 正确；19 例测试绿 |
| **D2** | ✔ | 短期记忆 | append compaction entry（retainedTail，乐观 expected_tip 防竞态 `CompactionRaceError`）+ build_context 处理（已在 A3 context.py 就绪）+ 每轮末 `_maybe_compact` 检查 + split-turn 前缀。**落点改为 factory 层 `_maybe_compact`（非 agent_loop）**；增量摘要(previousSummary) 与 before_compact 钩子待 B0 到位后补 | D1+A2 | `session/factory.py`+`session/compaction.py`+`session.py` | 多轮压缩后 context 正确；原始 entry 仍在库；未注入 summarize/context_window 时零行为变化；14 例测试绿 |
| **E1** | ☐ | 长期记忆 | MemoryStore + Extractor(agent_settled 后台) + Injector(before_prompt 回灌一次) | B0+A4(+D2) | `agent_core/memory.py` | 跨两 session 实例，第二个首 prompt 前见 `[Previous Session Summary]` |
| **F** | ☐ | 续跑引擎 | 13-leaf OperationState + drive/lane/reconcile/effect-gate + pending* 值 + 崩溃恢复（崩在工具间接着跑） | A（全部） | 新 `session/runtime/` | 大工程，另立子计划；对齐 packages/agent/src/harness/runtime |

---

## 5. SQLite 表设计（实现架构 B Storage）

```sql
sessions(id TEXT PK, storage_version INT, created_at INT, cwd TEXT, parent_session_id TEXT)
entries(id TEXT, session_id TEXT, parent_id TEXT, seq INT, timestamp INT,
        type TEXT, custom_type TEXT, payload TEXT,          -- payload=JSON(entry 剩余字段)
        PRIMARY KEY(session_id, id))
values(session_id TEXT, namespace TEXT, key TEXT, seq INT, value TEXT,   -- value=JSON
       PRIMARY KEY(session_id, namespace, key))
lists(session_id TEXT, namespace TEXT, key TEXT, seq INT, value TEXT,    -- append-only
      PRIMARY KEY(session_id, namespace, key, seq))
usage(session_id TEXT, seq INT, provider TEXT, model TEXT, kind TEXT, usage TEXT)
```

- `commit(writes)` = 单事务内按 write 类型分派到 entries/values/lists；seq 由 `MAX(seq)+1` 或自增分配。
- `scanBranch({start})` = 从 start 沿 `parent_id` 上溯（递归 CTE 或应用层迭代）。
- `scanEntries({fromSeq,toSeq,order,limit})` = 按 seq 范围查询。
- `PRAGMA foreign_keys=ON; journal_mode=WAL; busy_timeout=5000`。
- 接口对齐官方，换 PG 只改 store 实现（SQL 直译）。

---

## 6. 与现有代码的接触面

| 阶段 | 新增 | 修改 | 不动 |
|---|---|---|---|
| A0-A3 | `src/pi_agent/session/`（types/ids/values/serialize/errors/context/session/repo）+ `session/storage/` 子包（storage_types/commit/memory/sqlite） | 无 | agent_core、pi_ai 全部 |
| A4 | `session/factory.py`、`AgentSession` | `agent_core/agent.py`（AgentSession 组合 Agent，订阅 message_end） | agent_loop、事件协议、AgentState 字段、消息 dataclass |
| B0 | 无 | `event_stream.py`（派发语义） | provider、消息 dataclass |
| C1-C2 | `agent_core/skills.py` | agent.py（prompt 索引、read_skill） | session 层 |
| D1-D2 | `agent_core/compaction.py` | agent_loop.py、session（append compaction + build_context） | provider |
| E1 | `agent_core/memory.py` | 挂 B0 hook | session 接口 |
| F | `session/runtime/` | 深度介入 loop | — |

**红线（源码核对后的准确版）**：
- 原始 entry **永不物理删除**，压缩是 append compaction entry（带 retainedTail）+ build_context 重算。append-only。
- **持久化经事件订阅**（message_end→append），Agent 对持久化零耦合；配置经 `laneConfig` value 命令式写。
- 慢 I/O（摘要 LLM、写库）**fire-and-forget**，不阻塞。
- 长期记忆注入只在 `before_prompt` 一次，**禁** turn_start 自动 steer。
- 不挂载记忆组件时**现有 75 测试零变化**。
- **加载历史 ≠ 重跑工具**：恢复只读回 tool_result。
- 损坏数据策略**需拍板**（§7 决策 2）：官方架构 A 静默跳过；我们可选更严。

---

## 7. 需拍板的决策点

1. **续跑引擎 F 的范围**：第一版是否完全不做 F，只在 turn 边界落盘（崩在 turn 中途整轮重来）？（建议：是。F 太大，先交付可用的持久化+记忆。）
2. **损坏数据策略**：官方架构 A 静默跳过损坏行（容错）；PLAN3 主张显式抛异常（fail-loud）。SQLite 下损坏形态不同（单行 JSON 解析失败）。建议：**单 entry payload 解析失败→抛 `SessionCorruptError` 带 entry id**（比 JSONL 更该 fail-loud，因为 SQLite 不该出坏行）。待你确认。
3. **落盘时机**：官方"首条 assistant 才 flush"是文件 IO 优化；SQLite 有事务，可**每次 commit 即落**（更简单、更不易丢）。建议后者。
4. **customType 投影**：`EntryProjector` 机制第一版是否实现，还是 custom entry 一律不进上下文？建议先不进，留接口。
5. **Skill/长期记忆是否纳入本期**：可整体后置，先交付 A（持久化+门面）+ D（压缩）。

---

## 8. 官方（架构 B）⇄ Python 命名对照

| 官方 TS | Python | 备注 |
|---|---|---|
| `Storage` / `commit/getEntries/scanBranch/scanEntries/getValue/readList/scanUsage/getStats/close` | 同名 snake_case，`Protocol` | SQLite 实现之 |
| `StorageBackedSession` / `SessionRepo` | 同名 | 会话 + 仓库 |
| `Entry`(4 种) / `EntryBase` / `NewEntry` | dataclass | seq/timestamp 存储分配 |
| `beginMutation/mutate/branch/createBranch/setValue/appendList/setName/setLabel` | snake_case | 写走屏障 |
| `value<T>()/list<T>()/setValue/appendList` + `laneConfig/sessionName/entryLabel/branchTip` | `value()/list_()`+具名地址工厂 | KV 侧存储 |
| `buildContextEntries/buildSessionContext` | `build_context_entries/build_session_context` | compaction-aware |
| `MemoryStorage/JsonlStorage` | `InMemoryStorage/SqliteStorage` | 后端 |
| `createAgentSession/AgentSession/dispose` | `create_agent_session/AgentSession/dispose` | 门面 |

**暂不实现**：`OperationState` 13-leaf 状态机及 drive/lane/reconcile/effect-gate（轨道 F）；ModelRuntime/SettingsManager/ResourceLoader/CacheWarmer；版本迁移；PG 后端（接口预留）。

---

## 9. 与 PLAN.md 的对应

- Phase 3：`create_agent_session`/`AgentSession`（A 线）+ tree-semantic persistence（**架构 B 的 branch 模型**）。
- Phase 4：auto-compaction（D）、context hooks（B/E）、**断点续跑（F，对应 PLAN3 §9.1）**。
- **有意偏离**：JSONL → SQLite（实现架构 B 的 `Storage` 接口，理由见 §5）。
- Skill（C）对应 Phase 5，可后置。

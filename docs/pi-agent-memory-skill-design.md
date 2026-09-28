# pi-agent 长短期记忆机制 & Skill 渐进式加载 —— 完整分析与 Python 移植方案

> 基线版本：`@earendil-works/pi-coding-agent` **v0.85.1**（与 `skill/dg-piagent` 的 API 描述对齐）
>
> 参考：`skill/dg-piagent/`（`09-skills.md`、`12-session-manager.md`、`18-compaction.md`、`scenarios/C01`、`G01`、`G02`、`G03`）
> 目标：讲清"官方怎么实现 / 怎么召回 / 检索层怎么设计"，并翻译为 Python 集成到 `src/pi_agent/`

---

## 0. 一句话总结

> **短期记忆 = 会话树原生持有，compaction 负责"装不下时压缩成结构化摘要"；长期记忆官方只给了两块砖（AGENTS.md 静态规则 + JSONL 会话存档），动态的跨会话记忆需要扩展自建"提取 → 持久化 → 回灌"闭环。Skill 渐进式加载则让领域知识"省 token"地进入上下文。**
>
> **compaction 是"逻辑压缩"不是"物理删除"：原文永远留在会话树上，但对 LLM 单向有损——官方没有任何自动召回机制，也没有关键词/向量检索。检索层是官方刻意留白的设计空间。**

---

## 1. 全景图：pi-agent 没有独立的 memory / skill 模块

```
                    ┌──────────────────────────────────────────┐
                    │  ResourceLoader（资源层，会话启动时）        │
                    │  · Skills 渐进式加载  ← 渐进式披露在这里      │
                    │  · Context Files (AGENTS.md)              │
                    │  · 产出 → system prompt（索引区）           │
                    └────────────────┬─────────────────────────┘
                                     ↓
   user prompt ──→ agent loop（每一轮 LLM 调用）
                     │  · context hook：改本轮 LLM 输入快照   ← 短期注入点
                     │  · token 检查 → compaction           ← 短期记忆管理
                     ↓
                    ┌──────────────────────────────────────────┐
                    │  SessionManager（会话树持久化，jsonl）       │
                    │  · Message / CompactionEntry / CustomEntry│
                    └──────────────────────────────────────────┘
                                     ↑
   扩展层事件（turn_end / agent_settled）→ 提取要点 → 外部存储  ← 长期记忆写入
   before_agent_start → 加载历史摘要回灌                     ← 长期记忆读取
```

官方的"记忆"由四块机制拼成，"技能"由渐进式加载机制承载：

| 层 | 机制 | 载体 | 生命周期 | 检索方式 |
|---|------|------|---------|---------|
| 短期 | 会话树（SessionManager） | JSONL 文件 / 内存 | 单个会话 | 结构遍历 |
| 短期 | Compaction 压缩摘要 | 会话树内 `CompactionEntry` | 单个会话（entry 随树持久化） | 无（摘要全量进上下文） |
| 长期 | Context Files（AGENTS.md / CLAUDE.md） | 文件系统 | 跨会话、跨重启 | 无（启动时全量注入） |
| 长期 | 扩展外挂（G03 模式） | 外部存储 / 会话树 CustomEntry | 跨会话（自己实现） | **自己实现** |
| 资源 | Skill 渐进式加载 | SKILL.md + system prompt 索引 | 会话启动注入索引，运行期按需读正文 | 模型按需 read |

一句话分工：

| 机制 | 层 | 时机 | 作用 |
|------|----|------|------|
| Skill 渐进式加载 | 资源层 | 会话启动注入索引，运行期按需读正文 | 领域知识**省 token** 地进入上下文 |
| Compaction | 会话层 | token 逼近窗口上限时自动触发 | **短期记忆**：窗口内旧历史 → 结构化摘要 |
| 记忆提取+回灌 | 扩展层 | turn_end / agent_settled 提取；下次 before_agent_start 回灌 | **长期记忆**：跨会话保留要点 |

---

## 2. 短期记忆：会话树 + Compaction

### 2.1 会话树是记忆的"底座"

- 每个会话是一个 **JSONL 文件**：`~/.pi/agent/sessions/<编码后cwd>/<timestamp>_<sessionId>.jsonl`
  - 首行 `SessionHeader`，后续每行一个 `SessionEntry`
  - 环境变量 `PI_CODING_AGENT_DIR` / `PI_CODING_AGENT_SESSION_DIR` 可覆盖存储位置
- **append-only 树结构**：每条 entry 有 `id` / `parentId`，`leaf` 指针标记当前位置
  - 分叉 = 移动 leaf，原分支不删 → **历史永远不会真正丢失**
- 发给 LLM 的上下文由 `buildSessionContext()` 沿 root→leaf 路径**每次实时重算**，处理 compaction / branch_summary 后输出最终 `messages`
- entry 类型：`message` / `compaction` / `branch_summary` / `thinking_level_change` / `model_change` / `custom`（不进 LLM 上下文）/ `custom_message`（进 LLM 上下文）/ `label` / `session_info`

### 2.2 Compaction：窗口快满时的自动压缩

触发链路（agent loop 内自动）：

```
每轮结束 → shouldCompact(contextTokens > contextWindow - reserveTokens)？
  → prepareCompaction() 找切点 → compact() 调 LLM 生成摘要
  → 写入 CompactionEntry → 后续上下文 = 摘要 + 保留的近期消息
```

三种触发原因（`reason` 字段）：

| reason | 含义 |
|--------|------|
| `threshold` | 上下文 token 触及阈值，自动触发 |
| `overflow` | 已超窗紧急压缩；仅 `stopReason !== "stop"` 时 `willRetry = true` 重试该轮 |
| `manual` | 用户执行 `/compact` |

关键设计点：

- **默认参数**（`CompactionSettings`）：
  - `reserveTokens = 16384`：双重用途——① 触发阈值 = `contextWindow - reserveTokens`；② 摘要 LLM 输出上限 = `min(0.8 × reserveTokens, model.maxTokens)`。128K 窗口约 112K 时触发
  - `keepRecentTokens = 20000`：压缩后保留的近期上下文近似 token 数
- **切点安全**：`findCutPoint` 只在 user / assistant / bashExecution / custom / branchSummary / compactionSummary 等安全位置切割，**绝不切在 toolResult 中间**、不切在已有 compaction entry 上；切在轮次中间时（`isSplitTurn`）生成"历史摘要 + `---` 分隔 + 轮次前缀摘要"两段合并
- **增量摘要**：已存在 `CompactionEntry` 时，`previousSummary` 传入下一次摘要调用——只对新增消息做增量合并，不重摘全史
- **摘要模型可独立**：`compact()` 接受独立 model 参数，可用便宜的小模型做摘要
- **toolResult 截断只发生在摘要 prompt**：`serializeConversation` 里截断到 2000 字符（保留开头 + `[... N more characters truncated]`），**落盘的 entry 不受影响，存的是完整原始内容**
- **文件操作追踪**：从压缩区间的工具调用中提取 `read` / `write` / `edit`，去重后以 `<read-files>` / `<modified-files>` 标签附在摘要末尾
  - 去重语义：`readFiles` = 只读未改；`modifiedFiles` = edited ∪ written。既 read 又 edit 的文件只进 modifiedFiles

摘要结构化格式（这就是官方"赌摘要足够好"的全部本钱）：

```
## Goal / ## Constraints & Preferences
## Progress（Done / In Progress / Blocked）
## Key Decisions / ## Next Steps / ## Critical Context
<read-files>…</read-files>
<modified-files>…</modified-files>
```

### 2.3 两层 API 不要混用（高频坑）

| 维度 | subscribe 层（`AgentSession`） | 扩展层（`pi.on` 回调里的 `ctx`） |
|------|------|------|
| 触发压缩 | `await session.compact(customInstructions?)` → `Promise<CompactionResult>` | `ctx.compact(options)` —— **fire-and-forget 返回 void**，结果走 `onComplete`/`onError` 回调 |
| 事件 | `agentSession.on(...)`：`compaction_start` / `compaction_end` / `summarization_retry_*` | `pi.on("session_before_compact")`（可返回 `{cancel}` 或 `{compaction}` 替换摘要）/ `pi.on("session_compact")` |
| 读会话 | `session.sessionManager.getBranch()` 等 | `ctx.sessionManager`（**ReadonlySessionManager**，无 append*/branch*） |

### 2.4 Branch Summary（分支摘要）

用户跳转/放弃一个分支时，对被离开的分支做摘要（`branchWithSummary`），与 compaction 共用同一摘要引擎。相当于"这条支路到过这里，结论是……"的存档。

---

## 3. 压缩前的原文去哪了？——保留，但不召回

### 3.1 核心事实：逻辑压缩，不是物理删除

compaction 只做两件事：

1. **追加**一条 `CompactionEntry`（含 `summary` + `firstKeptEntryId`）
2. 之后 `buildContextEntries()` 重算上下文时，**跳过** `firstKeptEntryId` 之前的旧 entry

```
root → msg1 → msg2 → msg3 → msg4 → CompactionEntry(firstKeptEntryId=msg3) → msg5 ...
                ↑ 被"压缩"的 entry 仍在树上、仍在 JSONL 里
```

- **磁盘上**：完整原文一直在（append-only，什么都没删，含完整 toolResult）
- **LLM 眼里**：只剩摘要 + 保留消息，**一去不返**

### 3.2 官方为什么不自动召回？

这是刻意取舍：召回决策交给系统只有两个坏结局——

- 每次全量带回 → 窗口爆炸，压缩白做
- 引入语义检索 → 非确定性，行为抖动

官方选择把代价压缩在"摘要质量"一个点上，赌结构化摘要足够装下继续干活所需的信息。摘要末尾的 `<read-files>` / `<modified-files>` 就是留的钩子：LLM 看不到原文，但知道哪些文件被动过，需要时用 `read` 工具重新读盘——**文件本身就是编码 Agent 最大的持久化记忆**。

### 3.3 需要原文时的三条路（全部是主动触发，无自动机制）

1. **分叉回去**（交互层）：`sm.branch(oldEntryId)` 把 leaf 移到压缩前节点，新分支上下文原样恢复；原分支不受影响。代价是丢掉压缩后的那段对话
2. **扩展读树注入**（开发层）：`ctx.sessionManager.getEntries()` / `getEntry(id)` 随时能拿原文，通过 `before_agent_start` 返回 `{ message }` 或 `context` hook 拼回 LLM 输入。**"什么时候注入哪段"的判断逻辑官方不给，即第 6 节的检索层**
3. **摘要钩子**（信息层）：靠 `<read-files>` 标签引导 LLM 自己 `read` 重建

---

## 4. 官方召回现状：零检索

**关键词检索和向量检索在官方 SDK 里一处都没有**——没有内置向量库、没有 embedding 调用、没有 BM25。官方的"召回"全是**结构性的，不是语义性的**：

| 官方机制 | 召回方式 | 本质 |
|---------|---------|------|
| Compaction 摘要 | 不召回，摘要**全量**进上下文 | 用摘要换空间 |
| 分叉到压缩前 | 结构性回放（`branch(entryId)`） | 整段重新可见 |
| `continueRecent` / `inMemory(cwd, {id}, entries)` | 整个会话**全量**恢复 | 文件级，不挑内容 |
| `SessionManager.list()` 的 `firstMessage` / `allMessagesText` | 会话选择 UI 的元数据 | 不是给 Agent 检索用的 |

官方能这么做，是因为编码场景的上下文有天然结构边界（entry / 轮次 / 分支），且文件系统本身就是可重建的记忆。

---

## 5. 长期记忆：官方内置的部分 + 扩展外挂

### 5.1 Context Files——唯一官方内置的跨会话记忆

- 启动时发现：全局 `~/.pi/agent/AGENTS.md` + cwd 向根目录逐级找 `AGENTS.md > CLAUDE.md`
- 注入 system prompt 的 `<project_context>` 块（父级 → 子级顺序）
- 本质是**静态规则**：几乎不变的项目约定，零运行时开销
- 相关陷阱：
  - 即使设置 `customPrompt`，context files / skills / cwd 仍会被追加到 system prompt 末尾（`buildSystemPrompt`）
  - context files 加载**不受 projectTrusted 门槛**——宿主环境的 CLAUDE.md 会被误注入，二开时用 `noContextFiles: true` 关闭

### 5.2 会话持久化本身是长期记忆的"原料"

- `SessionManager.continueRecent(cwd)` 续接最近会话
- `SessionManager.forkFrom(sourcePath, targetCwd)` 跨项目继承会话（header 记录 `parentSession` 溯源）
- v0.85.0+：`SessionManager.inMemory(cwd, {id}, entries)` 可从自己的数据库恢复历史（Web 多用户"内存运行 + 外部落库"闭环；运行期不自动写盘，需自行 `getEntries()` 序列化落库）
- 陷阱：文件模式 `create()` 后**首条 assistant 消息到达前不落盘**（`_persist` 的 `hasAssistant` 守卫）

### 5.3 动态长期记忆 = G03 扩展外挂（官方明确说这不是内置功能）

官方提供的只是素材（事件 hook + 会话树 API），记忆闭环要自己组装：

```
提取（写侧）                          注入（读侧）
pi.on("agent_settled")        ──→    pi.on("before_agent_start")
  或 turn_end 提取要点                 从外部存储读回历史摘要
├─ pi.appendEntry("turn_summary", …)  return { message } 注入对话流
│    → CustomEntry 落到会话 JSONL      或 return { systemPrompt } 拼进系统提示词
└─ 或写到外部存储（DB / .agent/summaries/*.md）
```

配套官方工具函数（主入口导出）：

- `serializeConversation(convertToLlm(messages))`：消息转纯文本（toolResult 截 2000 字符）
- `generateSummary(messages, model, reserveTokens, apiKey, headers?, signal?, customInstructions?, previousSummary?, …)`：复用内置 compaction 的结构化摘要能力；传 `previousSummary` 走增量 prompt
- `generateSummaryWithUsage(...)`：同签名，返回 `{ text, usage }`

**G03 关键陷阱清单**：

1. `turn_end` 的 `event.message` 永远是 **assistant** 消息，拿不到 user 提问（去 `message_end` 抓 role==="user"，或 `agent_end` 里反向找）
2. `AssistantMessage.content` 是数组（`TextContent | ThinkingContent | ToolCall`），不是 string；`ToolResultMessage` 的字段是 `toolName` 不是 `name`
3. `CustomEntry`（`appendEntry`）**不进 LLM 上下文**——`buildSessionContext` 忽略自定义类型，必须靠 `before_agent_start` 主动注入才能被"看见"
4. 落库时机用 **`agent_settled`** 不用 `agent_end`（后者三条退出路径都触发、retry 场景多次触发）
5. `pi.on` handler 被派发方 `await`——LLM 摘要 / 写库必须 **fire-and-forget**（`queueMicrotask` 推后台），否则拖慢整个 `prompt()` resolve。若不需要扩展层 `ctx`，用 `session.subscribe("agent_settled", ...)` 落库更省心（subscribe 不 await listener）
6. 注入用 `before_agent_start`（每次 prompt 一次）；**千万别用 `turn_start` + `sendUserMessage({deliverAs:"steer"})`**——steer 触发新 turn → 新 turn 触发 turn_start → 死循环
7. 扩展替换摘要（`session_before_compact` 返回 `{compaction}`）时，`<read-files>`/`<modified-files>` 标签**不会自动附加**（只有内置 `compact()` 分支会拼），需自己用 `computeFileLists` + `formatFileOperations` 拼接

### 5.4 写入端：何时提取、提取什么

| 时机 | 粒度 | 拿到什么 | 备注 |
|------|------|---------|------|
| `turn_end` | 每轮 turn | 本轮 **assistant** 消息 + toolResults + turnIndex | ⚠️ **拿不到 user message**（常见误期待）|
| `message_end` | 每次 LLM 调用 | 单条消息；靠 role 配对 user/assistant | ⚠️ 单轮会触发多次（预文本/空消息/最终回答），配对需过滤空 assistant |
| `agent_settled` | 每次 prompt 一次 | 会话树还原出的完整 messages | **最终摘要最佳落点**：所有 retry/compaction/queue 处理完才触发 |

三个官方认证的坑（移植成 Python 事件系统时同样适用）：

1. **`turn_end.message` 永远是 assistant**，没有 `userMessage` 字段——抓 user 要用 `message_end`(role=user) 或 `agent_settled` 里反查。
2. **`agent_end` ≠ 结束**：error/abort/正常三条退出路径都触发，retry 时触发多次。做"最终摘要落库"必须用 `agent_settled`。
3. **事件 handler 被派发方 await**——摘要 LLM 调用、写库这类慢 I/O 必须 fire-and-forget（丢进后台任务），否则阻塞整个 agent loop。

### 5.5 存储端：存到哪

两种官方给出的落点，语义完全不同：

| 落点 | LLM 可见？ | 用途 |
|------|-----------|------|
| **外部存储**（文件 `.agent/summaries/*.md`、DB、KV） | 不可见，需回灌 | 主推。策略完全自主，用户/宿主可审计 |
| **CustomEntry**（`appendEntry(customType, data)` 写进会话树 jsonl） | **不可见**——构建 LLM 上下文时自定义 entry 被忽略 | 仅作持久化日志/审计；想被看见仍需回灌 |

> 关键认知：写进会话树 ≠ 进上下文。**任何长期记忆都要显式回灌**。

### 5.6 读取端：怎么回灌（注入机制选型）

G01 的四机制选型表（长期记忆回灌主要用前两个）：

| 注入机制 | 层次 | 时机 | 适用 |
|----------|------|------|------|
| **`before_agent_start` 返回 `{message}` 或 `{systemPrompt}`** | 消息历史 / system prompt | 每次 prompt 一次，**无死循环风险** | ★ 会话启动加载历史摘要的**标准落点** |
| **`context` hook 返回 `{messages}`** | 本轮 LLM 输入快照（不进历史） | 每轮 LLM 调用前 | 高频变化的外部数据（git status、最新记忆条目） |
| Context Files（AGENTS.md） | system prompt `<project_context>` 块 | 启动一次 | 几乎不变的静态规则 |
| `steer()` / sendUserMessage(deliverAs=steer) | user message 进历史，触发新 turn | 运行中插队 | 交互补充指示；**别用于自动注入** |

**死循环陷阱**（Python 事件系统同样要防）：在 `turn_start` 里自动 `sendUserMessage(steer)` → steer 触发新 turn → 又触发 turn_start → 无限循环烧 token。注入上下文一律走 `before_agent_start` 或 `context` hook。

官方推荐组合（模式 A+D）：

```
turn_end/agent_settled  ──fire-and-forget──→  摘要外部存储（每轮增量/每 prompt 终稿）
before_agent_start（loaded 标志位，每会话一次）─→ 读外部存储 → { message: [Previous Session Summary]… }
```

---

## 6. 检索式召回层设计（二开方案）

### 6.1 核心架构：向量索引是"派生缓存"，不是"记忆本体"

因为会话树是 append-only 的 ground truth，**不需要担心写侧完备性**——这是本架构比常规 RAG 简单一个量级的根本原因：

```
会话树（JSONL，永远完整）＝ 真相层
      │  离线/按需重建，按 entry id 幂等 upsert
      ▼
向量索引（entry id → embedding）＝ 派生物、缓存
      │  查询
      ▼
召回 top-k → 注入上下文（带出处）
      │  需要原文时
      ▼
getEntry(id) 精确取原文 / branch(entryId) 整段回放
```

推论：

- 写侧可以**完全异步化**：不用在 agent_settled 里实时 embed，后台任务扫 `getEntries()`，按 entry id 幂等 upsert
- 挂了、漏了、换 embedding 模型 → **删了重建**，无损
- 召回结果必须**带出处**（会话 id、entry id、时间戳），与 pi 的会话树对得上，支持深挖

### 6.2 索引什么？——只索引"离开文件系统就消失的信息"

**工具结果（toolResult）恰恰最不该进向量库**：文件可重读、git log 可重跑，索引它们只稀释召回质量、膨胀存储。

| 索引什么 | 为什么 | 建议粒度 |
|---------|--------|---------|
| 用户消息（改需求、否决、偏好） | "上次说不要 JWT"这种决策链，文件里没有 | 每条 user message |
| assistant 的关键结论 / 方案 reasoning | "为什么选方案 B"，文件里没有 | 提取后按轮次落，别按 token 切块 |
| CompactionEntry 摘要 | 天然高浓度，适合直接做检索单元 | 整条 |

经验法则：**能被 `read` 工具或 git 重建的，不索引；只有对话里"说过的话"才索引**。体积压掉 90%+，召回精度反而上去。

### 6.3 检索选型：关键词 / 向量 / 混合

- **关键词（BM25 / SQLite FTS）够用**：记忆条目是短文本（决策、约定、结论），query 带明确词（项目名、文件名、功能名）。编码场景大多如此。优点：零 embedding 成本、可解释、无漂移
- **必须向量**：记忆是长叙述、query 是口语化改写（"之前那个登录的事"→"认证方案讨论"）。缺点：embedding 成本 + chunk 粒度问题
- **实战推荐**：**BM25 打底 + 向量重排**的混合——关键词粗筛 top-50（便宜），向量重排取 top-k（只跑小量）

### 6.4 注入时机与方式

| 插槽 | 时机 | 适用 | 成本 |
|------|------|------|------|
| `before_agent_start` → `return { message }` 或 `{ systemPrompt }` | 每次 prompt 一次 | 默认选择，覆盖 90% 场景 | 低 |
| `pi.on("context")` → `return { messages }` | 每轮 LLM 调用前 | 多轮工具调用后记忆漂移时才升级 | 每轮 embedding + 检索，翻几倍 |

注意事项：

1. **查询构造别拿原始 prompt 直接 embed**——用户说"还是用上次那个办法"时 query 里啥都没有。拼上最近 1~2 轮对话作为 query 上下文再检索
2. **注入格式**：top-k 控制在 3~5 条、每条几百 token；用明确分隔标记（如 `<recalled-memory source="session-xxx entry-yyy">`），让模型知道这是"回忆"不是"当前指令"
3. **召回质量差时宁可不注入**（空结果别塞"没有相关记忆"这类噪音）
4. **`before_agent_start` 是扩展独有事件**——`session.subscribe` 监听会静默失败，必须走 `pi.on`（6 个扩展独有事件之一）

### 6.5 落地路径：先做"穷人版"验证

不要一上来建向量库。验证顺序：

1. **V0（穷人版）**：把历届 `CompactionEntry.summary`（不切块、不 embed、全文）在 `before_agent_start` 全量注入——摘要本来就是压缩过的，十几个会话也就几千 token。若解决 → 到此为止
2. **V1（关键词）**：摘要注入超预算但不精准 → SQLite FTS / BM25 索引上表的"三类该索引的内容"
3. **V2（混合）**：出现口语化改写命中失败 → 加向量重排

### 6.6 决策点检查表（动手前自问）

- [ ] 用户场景里是否真的存在"回头引用超过一个压缩周期之前的对话"？（多数编码助手实际很少——用户会重新说一遍）
- [ ] V0 全量摘要注入是否已经够用？
- [ ] 索引范围是否排除了 toolResult？
- [ ] 召回条目是否带出处（entry id）？
- [ ] 注入是否有分隔标记、是否允许空结果？
- [ ] 索引是否可整体重建（缓存语义）而非唯一真相？

---

## 7. Skill 渐进式加载（Progressive Disclosure）

### 7.1 核心思想

**只把"目录"放进 system prompt，"正文"留给模型按需读取。**

- 注入 system prompt 的是每条 Skill 的**索引**：`name + description + location`（XML 块），几十 token 一条。
- SKILL.md **正文从不自动注入**。索引附带一行指令：
  > Use the read tool to load a skill's file when the task matches its description.
- 模型判断任务匹配某条 description 后，自己用 **read 工具**读 `<location>` 指向的 SKILL.md 全文。
- 相对路径约定：SKILL.md 内引用的相对路径，以 skill 目录（SKILL.md 的父目录，即 `baseDir`）为根解析——索引里明确告知模型这一点。

收益：10 个 skill 每个正文 2K token，全量注入 = 2 万 token 常驻；渐进式 = 索引约 500 token 常驻 + 命中时才付出正文成本。

### 7.2 Skill 数据模型

```python
@dataclass
class Skill:
    name: str            # [a-z0-9-]，≤64 字符，不能 - 开头/结尾、不能连续 --
    description: str     # ≤1024 字符；为空则该 skill 不加载（官方行为）
    file_path: str       # SKILL.md 路径，模型按此读取正文
    base_dir: str        # skill 根目录，相对路径解析基准
    disable_model_invocation: bool  # True=不进索引，仅显式 /skill:name 可调
    # source_info: 来源元信息（user/project/temporary scope），用于冲突排序与调试
```

Frontmatter（YAML）字段：`name`、`description` 必填；`disable-model-invocation`、`license`、`metadata`、`allowed-tools` 可选（后四者 pi 不消费，仅 spec 约定）。frontmatter 缺 `name` 时回退为所在目录名。

### 7.3 发现规则

- **目录布局**：`<root>/<skill-name>/SKILL.md`。
- **递归规则**：某目录含 `SKILL.md` → 即为 skill 根，**不再向下递归**；否则扫描直接子 `.md`（pi 模式）并递归进子目录找 `SKILL.md`。
- **加载来源顺序**：用户全局 `~/.pi/agent/skills/` → 项目 `.pi/skills/` → 显式额外路径；`.agents/skills/` 有独立的祖先遍历规则（对齐 git root）。跳过 `node_modules`、隐藏目录、gitignore 匹配项。
- **同名冲突**：后加载者被丢弃并记 diagnostic；SDK 集成路径下优先级 project > user > package。
- **名称校验失败不 throw**：记 warning diagnostic，skill 照常加载但 `/skill:name` 解析可能异常。
- **信任门槛**：项目级 skill 受 `projectTrusted` 控制（SDK 默认 True）；而 Context Files（AGENTS.md）**不受**该门槛——移植时若保留 AGENTS.md 兼容，要意识到这是注入风险点。

### 7.4 索引注入格式（formatSkillsForPrompt）

```
The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory
(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.

<available_skills>
  <skill>
    <name>data-schema</name>
    <description>Provides database table schemas and field descriptions</description>
    <location>/my-project/.pi/skills/data-schema/SKILL.md</location>
  </skill>
  ...
</available_skills>
```

`disable_model_invocation=True` 的 skill 不出现在此块中。

### 7.5 正文进入模型上下文的三条路径

| 路径 | 触发方 | 机制 |
|------|--------|------|
| ① read 工具按需读取 | 模型自主 | 索引 description 匹配 → read(file_path)。**主路径** |
| ② `/skill:name` 显式调用 | 用户 | 读取 SKILL.md 全文、strip frontmatter，包成 `<skill name="..." location="...">正文</skill>` 块注入用户消息；args 追加在块后。`disable_model_invocation=True` 也能显式调 |
| ③ context hook / 主动注入 | 宿主代码 | 运行时按业务逻辑把某 skill 正文塞进消息快照（G01 模式 B） |

### 7.6 最大集成坑（移植时必须内置防护）

**工具白名单必须含 read，否则 skill 形同虚设**：模型看到索引却永远拿不到正文，表现为"反复用工具探查已知信息，一次查询多耗 5+ 次调用"。官方建议：

- 白名单加 `read`；若项目含 `.env`/源码等敏感文件，用**包装过的 read**（限制只允许读 skills 目录）替代内置 read。
- 验证方法：打印格式化后的索引 XML——应只有索引没有正文；trace 里看 read 调用。

---

## 8. 实现载体分层：hooks vs 内置（★ 先读，决定每块代码放哪）

官方 pi 的做法不是"全都做成 hooks"，而是分层：hooks 是**可选逻辑的触发点**，核心流程**内嵌**在 loop/组装链里。移植时先按此表归位：

| 机制 | 载体 | 是否 hook | 理由 |
|------|------|-----------|------|
| Skill 渐进式加载 | 资源层：`load_skills()` → prompt 组装拼索引 → 注册 `read_skill` 工具 | ❌ | 会话启动时静态完成，硬套 hooks 会把"启动注入一次"变成"每轮动态判断"，浪费且不符语义 |
| Compaction 主体 | **内嵌 `agent_loop.py`** 每轮末尾 | ❌ | 核心控制流；做成 hook 会出现"没人注册就没人压缩"的空转。官方同样是内置 + 拦截钩子组合 |
| Compaction 定制点 | `before_compact`（cancel/替换摘要）、`after_compact`（通知/落库） | ✅ | 换摘要模型、聚焦指令、可观测性都是可选逻辑 |
| 长期记忆外挂 | `agent_settled` 后台提取落盘 + `before_prompt` 回灌 | ✅ | 整体就是一组 hook handler，是 hooks 的典型用户 |
| hook 总线本身 | `event_stream.py` 升级为统一扩展点 | —— | 地基。先有它，上面的才挂得上去 |

**归位判断标准**：「某个时点要触发一段**可选**逻辑」→ hook；「每次调用**必然发生**的核心流程」→ 内置进 loop/组装链，最多暴露拦截点。

**hook 总线设计要点**（M0 落实）：

- 生命周期点集合：`turn_start` / `turn_end` / `message_end` / `agent_end` / `agent_settled` / `before_prompt` / `before_compact` / `after_compact`。
- 同一事件点支持多个 handler，按注册顺序派发。
- **慢 I/O fire-and-forget**：handler 内做摘要 LLM 调用、写库时用 `asyncio.create_task` 丢后台，派发方不 await——这是官方最重要的约定（`pi.on` handler 被派发方 await，同步慢 I/O 会阻塞整个 agent loop）。
- 直接实现 `agent_settled` 语义（所有 retry/compaction/queue 处理完才触发一次），跳过官方 `agent_end` 多次触发的坑。
- 防死循环约束写进总线文档：自动注入上下文只允许 `before_prompt`（每次 prompt 一次）或 context 快照改写，**严禁**在 `turn_start` 自动 steer 类注入。

---

## 9. Python 移植方案

### 9.1 模块落点总览

```
src/pi_agent/
├── agent_core/
│   ├── skills.py        ← 渐进式加载（M1/M2）
│   ├── compaction.py    ← 短期记忆压缩（M3/M4）
│   └── memory.py        ← 长期记忆外挂（M5）
├── event_stream.py      ← hook 总线（M0，地基）
└── agent_loop.py        ← 核心循环（内嵌压缩检查）
```

### 9.2 Skill 模块（`skills.py`，M1/M2）

```
Skill / SkillFrontmatter (dataclass)
load_skills(roots, extra_paths) -> SkillLoadResult(skills, diagnostics)
discover_skills(dir)            # SKILL.md 优先、递归、忽略规则
parse_skill_md(path)            # YAML frontmatter + 正文
format_skills_for_prompt(skills)# 7.4 的索引 XML
expand_skill_command(text)      # ①② 路径：/skill:name 展开（若保留该语法）
```

集成点：
- `agent.py` 组装 system prompt 时调 `format_skills_for_prompt()` 追加索引区。
- 提供 `read_skill` 工具（路径白名单限定 skill 根目录），注册进工具表——比开放通用 read 更安全，等价于官方"包装 read"建议。
- skill 正文按需读走工具调用，天然契合现有 `agent_loop`。

### 9.3 Compaction 模块（`compaction.py`，M3/M4）

```
CompactionSettings(enabled, reserve_tokens=16384, keep_recent_tokens=20000)
should_compact(context_tokens, context_window, settings) -> bool
estimate_context_tokens(messages, last_usage) -> int   # usage 精确 + char/4 兜底
find_cut_point(entries, keep_recent_tokens) -> CutPointResult  # 安全切点
serialize_conversation(messages) -> str                # 含 toolResult 2000 截断
SUMMARY_PROMPT                                         # 2.3 结构化模板
async compact(preparation, model, custom_instructions, previous_summary) -> CompactionResult
CompactionEntry(dataclass)                             # 写入会话存储
build_context(entries)                                 # summary + first_kept 之后的消息
```

集成点：`agent_loop.py` 每轮 turn 结束后插入 token 检查 → 触发压缩 → 原地更新消息列表。会话树持久化依赖会话存储层（当前 mock/内存起步，接口预留 `first_kept_entry_id` 即可）。

### 9.4 Memory 模块（`memory.py`，M5）

```
MemoryStore(Protocol)            # save(summary) / load() / load_recent(n)
FileMemoryStore / SqliteMemoryStore   # 落点自选
MemoryExtractor                  # 在 agent_settled 后台任务中调摘要模型
MemoryInjector                   # 每次 prompt 前（等价 before_agent_start）注入一次
```

集成点：
- 现有 `event_stream.py` / `agent_loop.py` 的事件派发改为/确认 **asyncio.create_task** 语义：订阅者慢 I/O 不阻塞主循环（对应官方 fire-and-forget 约定）。
- 事件集合对齐：`turn_end`、`message_end`、`agent_end`、`agent_settled`（建议直接实现 settled 语义，跳过官方踩过的坑）。
- 注入时机放在 `prompt()` 入口、agent loop 启动前（等价 before_agent_start），每会话 loaded 标志一次 + 可配 TTL。

### 9.5 分阶段开发计划（按 MD 逐步实施）

> 每阶段可独立验证，mock provider 即可测（参考 examples/mock_demo.py）。

**M0 — hook 总线**（改造 `event_stream.py`）
- 统一生命周期事件点（`turn_start`/`turn_end`/`message_end`/`agent_end`/`agent_settled`/`before_prompt`/`before_compact`/`after_compact`），多 handler 按序派发
- 慢 I/O fire-and-forget 语义（`asyncio.create_task`），写死在派发器里
- 验证：单测——handler 内 `await asyncio.sleep` 不阻塞主循环；`agent_settled` 每 prompt 只触发一次

**M1 — Skill 数据层**（skills.py）
- Skill dataclass + frontmatter 解析（PyYAML）+ name/description 校验（违规记 diagnostic 不抛异常）
- `discover_skills()` 递归发现 + `format_skills_for_prompt()` 索引 XML
- 验证：单测——空 description 不加载、非法 name 记 warning、索引无正文

**M2 — Skill 读取工具与接线**
- `read_skill` 工具（路径白名单限定 skill 目录）注册进工具表
- system prompt 组装追加索引区；验证：mock 测试模型能"读 skill 后回答"
- 可选：`/skill:name` 展开语法（strip frontmatter 包 `<skill>` 块）

**M3 — Compaction 核心**（compaction.py）
- token 估算 + `should_compact` + `serialize_conversation`（toolResult 2000 截断）
- 结构化摘要 prompt + `compact()`（接 pi_ai provider，摘要模型可与对话模型不同）
- 验证：mock 超长会话触发压缩，检查 summary 结构、切点不落在 toolResult 中间

**M4 — Compaction 接入 agent loop + 会话树**
- `CompactionEntry` + `build_context()`（summary + first_kept 之后）；agent_loop 每轮末检查
- 增量摘要（previousSummary）；`before_compact` 钩子（cancel/替换）
- 验证：两轮压缩后上下文正确、原始终端 jsonl 仍保留全史

**M5 — 长期记忆外挂**（memory.py，挂 M0 的 hook 总线）
- MemoryStore + Extractor（`agent_settled` 后台任务）+ Injector（`before_prompt` 回灌一次）
- 验证：跨两个 session 实例，第二个 session 首条 prompt 前能看到 `[Previous Session Summary]`

---

## 10. 移植时必须遵守的行为契约（坑清单汇总）

1. Skill 正文**永不**自动注入；索引必须附"用 read 工具按需加载"指令 + 相对路径解析说明。
2. 工具白名单没有 read（或 read_skill）时，skill 系统整体失效——集成层应做启动检查并告警。
3. name 校验失败 / description 为空：**不抛异常**，记 diagnostic；description 空则不加载。
4. `reserveTokens` 一参两用（触发阈值 + 摘要 maxTokens），改默认值前想清楚。
5. 切点绝不能落在 toolResult 中间或已有 compaction 上；split turn 要合并两段摘要。
6. 原始历史**不删除**——CompactionEntry 是追加，上下文重算时跳过旧 entry。
7. 摘要输入侧 toolResult 截断 2000 字符是**预算控制**，不得扩散到消息本体或落盘数据（砍尾只许发生在摘要输入，且必须带 `[... N more characters truncated]` 标记）。
8. 摘要/写库等慢 I/O 一律后台任务，不阻塞事件派发与 agent loop。
9. 长期记忆自动注入只允许 prompt 前一次（或 context hook），**严禁**在 turn_start 语义里自动 steer 注入（死循环）。
10. 写入会话树的 CustomEntry 默认不进 LLM 上下文——"存了"≠"看见了"。

---

## 11. 关键 API 速查

```ts
// 会话读取（含被压缩 entry）
sm.getEntries()                    // 全部 entry（浅拷贝，append-only 不可改）
sm.getBranch(fromId?)              // root→leaf 全路径
sm.getTree()                       // 树形结构（含 label）
sm.getEntry(id)                    // 精确取某条
sm.buildContextEntries()           // 处理了 compaction 的可见 entry 列表（LLM 视角）
sm.buildSessionContext()           // → SessionContext { messages, thinkingLevel, model }

// 分叉
sm.branch(entryId)                 // leaf 移到任意历史 entry（含压缩前的）

// 压缩
await agentSession.compact(customInstructions?)   // subscribe 层
ctx.compact({ customInstructions, onComplete, onError })  // 扩展层，fire-and-forget
shouldCompact(contextTokens, contextWindow, settings)
generateSummary(messages, model, reserveTokens, apiKey, …, previousSummary?)
generateSummaryWithUsage(...)
serializeConversation(convertToLlm(messages))

// 扩展记忆闭环
pi.on("agent_settled", …)          // 写侧信号（所有 retry/compaction/queue 处理完）
pi.appendEntry(customType, data)   // 写 CustomEntry（持久化但不进 LLM 上下文）
pi.on("before_agent_start", …)    // 读侧注入（return { message } 或 { systemPrompt }）
pi.on("context", …)                // 每轮改 LLM 输入快照（structuredClone 深拷贝，必须 return）
```

---

## 12. 参考

- skill 文档：`skill/dg-piagent/`
  - `references/sdk_doc/09-skills.md`（技能机制）
  - `references/sdk_doc/18-compaction.md`（压缩机制全文）
  - `references/sdk_doc/12-session-manager.md`（会话树全文）
  - `references/scenarios/C01`、`G01`（注入机制四件套）、`G02`、`G03`（记忆外挂模式）
- 官方源码：`packages/coding-agent/src/core/compaction/`、`src/core/session-manager.ts`（GitHub: `github.com/earendil-works/pi`）

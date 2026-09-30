# pi-agent 全面 Bug 扫描与代码优化（ultracode workflow 提示词）

> 使用方式：在本项目会话中，把下面「==== 提示词正文开始 ====」之后的全部内容作为一条消息发出
> （开头必须保留 `ultracode:` 关键字，它会触发多 agent workflow 编排）。
> 也可以直接对我说："按 CODE_REVIEW_PROMPT.md 跑"。

---

==== 提示词正文开始 ====

ultracode: 对这个 Python Agent 项目做全面的 bug 扫描和代码优化分析。

项目根路径：D:\yxr_files\pi-agent\

项目背景（先读 README.md / PLAN.md / PLAN2.md / PLAN3.md /PLAN4.md 建立上下文）：
这是官方 TypeScript SDK `@earendil-works/pi-coding-agent` v0.85.1 的 Python 移植学习项目
（Python >= 3.11，src 布局，uv 管理，pytest + ruff + mypy strict）。
两层架构：
- `pi_ai`：LLM provider 层。registry（模型/stream_fn 装配）、runtime（薄适配）、
  providers/openai.py（OpenAI Responses API + SSE 翻译，1469 行，最大文件）、
  providers/openai_completions.py（chat/completions 协议，1054 行）、
  providers/mock.py（测试用假 provider）、providers/_json_repair.py（流式残缺 JSON 修复）
- `agent_core`：agent 编排层。
  event_stream.py（生产者/消费者：asyncio.Queue + result future + 哨兵）、
  agent_loop.py（回合循环：调 LLM → 工具链 → steering/follow-up 双队列插队，576 行）、
  agent.py（对外门面 Agent：prompt/continue_/abort/subscribe，消费事件维护 state，404 行）、
  types.py（消息/内容块/配置的 dataclass 与字面量类型，373 行）
官方 TS 的中文解读在 skill/dg-piagent/（SKILL.md + references/sdk_doc + scenarios），
可作为"官方应有行为"的规格参照，但注意它是对 TS 版的解读，不是对 Python 代码的描述。

**已知问题（不要当作新发现重复上报；可以验证当前修复是否彻底）：**
1. agent_loop 生产者 task 异常导致 prompt() 永久挂起——已用 try/except + _fail_with_error 修复
2. Agent._emit 对 listener 抛异常无隔离（per-listener try/except 尚未加，测试已先行写好）
3. abort_event 只在发请求前检查，取消不了在途 HTTP/SSE 请求

**项目红线（发现违反必须报 Critical 或 High）：**
- 容错 JSON 修复绝不允许"砍尾"：不得通过截断/丢弃模型已生成字段来换取可解析，
  救不回只能返回 {} 让 jsonschema 回灌重试。审查 _json_repair.py 时把这一点作为头号检查项。
- 任何生产者异常路径都必须产出终态事件（agent_end + 哨兵），不准让消费者永久挂起。

请按以下方式组织扫描：

## Phase 1【结构侦察 - 必须输出结构化报告】

你是一个代码考古 agent，任务是深度阅读这个项目的所有核心 Python 模块，
输出一份供后续 bug 扫描 agent 使用的项目地图。

需要阅读的文件（按优先级），全部在 D:\yxr_files\pi-agent\src\pi_agent\ 下：
1. agent_core/event_stream.py（93 行）— 最底层，先读，所有并发问题的根
2. agent_core/types.py（373 行）
3. agent_core/agent_loop.py（576 行）— 最重要，重点读
4. agent_core/agent.py（404 行）— 最重要，重点读
5. pi_ai/providers/openai.py（1469 行）— 最大文件，重点读
6. pi_ai/providers/openai_completions.py（1054 行）
7. pi_ai/providers/_json_repair.py（141 行）
8. pi_ai/providers/mock.py（178 行）
9. pi_ai/runtime.py（171 行）
10. pi_ai/registry.py（59 行）
11. pi_ai/types.py（67 行）
12. 两个 __init__.py 的导出面
另外参考（不是被扫描对象，是规格）：skill/dg-piagent/SKILL.md 与
skill/dg-piagent/references/ 中和事件协议、持久化、工具相关的文档；
tests/ 下的测试用来反推契约。

必须按以下格式逐一分析每个模块：

=== 模块分析 ===

【模块名】（文件名）
【职责】一句话描述它做什么
【核心类/函数】列出最重要的 3-5 个类或函数及其作用
【对外接口】其他模块调用它的入口是什么（含 import 路径）
【依赖关系】它 import 了哪些本项目内部模块
【关键状态】它维护了哪些持久状态（dataclass 字段、闭包变量、asyncio 对象：Queue/Event/Future/Task）
【异步形态】它是 async 还是 sync？后台 task 在哪创建、由谁兜底？事件在哪个 task 生产、哪个 task 消费？
【潜在风险点】阅读时直觉感到"这里可能有问题"的地方，列 1-3 条

（以上格式对每个模块重复）

=== 全局分析 ===

【模块依赖图】用文字描述模块间调用关系，指出：
- 核心调用链：Agent.prompt → agent_loop → stream_fn（registry 装配）→ provider SSE → 回推事件
- 循环依赖风险
- 最高耦合点：types.py 被多少模块依赖、config 对象如何一路透传

【事件协议清单】列出系统中全部事件类型（agent 级事件 + provider 内 12 个流式子事件），
每个事件：谁生产、谁消费、携带什么 payload、是否终态。指出 payload 命名不一致
（snake_case vs TS camelCase）和同一事件在两条 provider 路径上的形状差异。

【共享状态清单】跨模块/跨 task 共享的对象：EventStream 队列、Agent.state、
partial_message 替换、new_messages/current_context.messages 两份消息列表、
provider 内的 state 映射（如 tool_item_id_to_call_id）。在哪里创建、被谁读写。

【异步边界】sync/async 交界（_maybe_await、stream_fn 同步返回 stream、
openai SDK 的迭代器、asyncio.create_task 生产模式），
列出所有"后台 task 抛出的异常可能无人 retrieve"的位置。

【TS 对照偏离清单】结合 skill/dg-piagent 文档，列出 Python 版相对官方 TS v0.85.1
已知或可疑的行为偏离（事件缺种、语义不同、未实现机制），标注哪些是"有意简化"、
哪些可能是"翻译走样"。

【高风险模块排序】按 bug 可能性从高到低排列，说明理由
（体量、状态复杂度、并发/SSE 解析、字符串修补等）。

禁止：不要总结"这是一个 AI agent 项目"这类废话。
要求：每个字段必须填写，不能空着，不能写"详见代码"。

## Phase 2【并行 Bug 扫描】

每个 agent 执行前，先获取 Phase 1 的项目地图作为上下文，再按维度分工，
7 个 agent 同时并行执行：

- Agent A 异步/并发：
  生产者/消费者误用、queue 死锁、哨兵/终态遗漏、create_task 异常无人消费、
  future 完成两次或永不完成、listener 同步广播阻塞消费循环、
  并发 prompt 守卫（already processing）、steering/follow-up 调度竞争
- Agent B 事件协议与状态机：
  partial/final 消息替换（context.messages[-1] = ...）是否可能错槽、
  added_partial 分支、stop_reason 各取值路径、turn_start/turn_end/agent_end 配对、
  异常路径补发第二个 agent_end 的契约冲突、state.is_streaming/stream_message 残留、
  toolcall_start/delta/end 与 text/thinking 三组子事件的配对
- Agent C 资源管理：
  openai SDK stream/response 是否在所有路径（含异常、提前 break、abort）关闭、
  httpx client 生命周期、async generator 未耗尽、abort 在途请求无法取消、
  事件订阅未取消订阅导致的listener泄漏
- Agent D 异常处理：
  裸 except/broad except、异常被吞只打日志、错误类型不匹配、缺少 finally、
  工具执行异常被包装成 tool result 后 is_error 是否一致、
  get_api_key/registry 解析失败的传播路径、后台 task "task exception never retrieved"
- Agent E 空值与边界：
  SSE 缺字段/None（历史教训：火山 response.created 缺 output 导致 NoneType 崩溃）、
  delta 累积器初始值、空 content/空 arguments、output_index/call_id 查不到映射、
  off-by-one、消息列表为空时 continue_、JSON 修复边界（空串/纯中断/嵌套转义/中文）
- Agent F 配置、registry 与初始化：
  api_key/base_url/env 缺失处理、ProviderRegistry 名实不符（dict 而非真 registry）、
  透传但零消费的假旋钮（thinking_budgets、max_retry_delay_ms 等）、
  默认模型/provider、dataclass 默认值可变共享、config 被 replace/原地改写的混淆
- Agent G Provider SSE 翻译层专项（openai.py + openai_completions.py + _json_repair.py）：
  SSE 事件到 12 子事件的映射遗漏/错配、DeepSeek 双 ID（item.id vs call_id）映射表的
  登记时机与漏登记路径、流式 partial JSON 修复调用点、**砍尾红线审计**、
  usage/stop_reason 提取、两条 provider 路径（Responses vs chat/completions）行为分叉、
  与官方 TS 翻译器（output_index 槽位法）的差异

每个 agent 的输出格式：
[文件名:行号] 问题描述 | 触发条件（给出具体输入/时序） | 严重程度: Critical/High/Medium/Low

严重程度标准：
- Critical：必现、主流程中断/挂死、数据丢失、违反砍尾红线、错误的工具参数被真的执行
- High：大概率触发、影响稳定性（资源泄漏、异常被吞导致静默失败、状态残留）
- Medium：偶发、影响体验或边界行为
- Low：代码质量、命名、与 TS 偏离但不影响运行

## Phase 3【Adversarial 验证】

仅对 Phase 2 中标记为 Critical 和 High 的 finding 进行对抗验证，
Medium 和 Low 直接归入 Uncertain，不做验证。

对每个 Critical/High finding，派 2 个独立 agent 尝试反驳：

Refuter 1（代码路径角度）：
→ 这个 bug 的触发路径在实际调用链中存在吗？从 Agent.prompt/agent_loop 一路追得到吗？
→ 有没有其他代码已经做了防护（上层 try/except、前置校验、SDK 自身保证）？
→ tests/ 里是否已有测试钉死了该行为（有测试覆盖且通过 → 倾向反驳成功）？

Refuter 2（运行时角度）：
→ 即使代码有问题，在真实运行（DeepSeek/OpenAI 兼容端点、mock、工具调用链、abort、插话）
  中会触发吗？需要什么时序条件？
→ 触发频率是偶发还是必现？后果会不会被下游兜住？

每个 Refuter 必须在回答最后一行输出以下其中一个，不得省略，后面不得有任何文字：
【结论】反驳成功：该 bug 不存在或无法触发
【结论】反驳失败：该 bug 确实存在且可触发

投票判定：
- 两个都"反驳成功" → 误报，丢弃
- 一个成功一个失败 → Uncertain，保留
- 两个都"反驳失败" → Confirmed Bug，进入 REPORT.md
- 任一 Refuter 未输出结论标签 → 视为反驳失败，保守处理

## Phase 4【汇总输出】

输入材料：Phase 3 Confirmed（Critical/High）、Uncertain（未验证的 Medium/Low + 降级项）、
Phase 2 全部原始 findings（优化线索）、Phase 1 项目地图。

任务 1：Bug Report。每个 Confirmed Bug 输出：
- 文件名和行号
- 问题描述
- 触发条件
- 修复方案：必须具体到"在 xxx 函数的 yyy 位置，用什么机制改"，
  禁止"需要加锁/需要处理一下"这类空话；异步问题要指明 task/queue/future 的具体操作

任务 2：优化建议。每条必须指向具体文件和函数（能回答"改哪个文件的哪个函数，怎么改"）：
- 性能：SSE 翻译热路径上的字符串/列表操作、delta 累积、JSON 重复解析、
  不必要的消息深拷贝
- 代码结构：两层之间的耦合、两条 provider 路径可抽象复用的翻译逻辑、
  config 透传链过长、types.py 膨胀的拆分方向
- 可维护性：事件 payload 命名统一（对齐 TS camelCase 的迁移策略）、
  假旋钮要么接线要么删除、错误处理风格统一、日志缺口、与官方 TS 的差距优先级清单
  （retry/agent_settled → 扩展层事件 → create_agent_session → 内置工具 → compaction）

禁止：
- "建议优化性能"这类无意义的话
- 把"已知问题"三条当新发现（只能验证修复是否彻底）

输出：写入项目根目录 REPORT.md，严格按以下模板：

# REPORT.md

## Part 1: Bug Report

### Critical（必须修）
| 文件:行号 | 问题描述 | 触发条件 | 修复方案 |
|---|---|---|---|

### High（强烈建议修）
| 文件:行号 | 问题描述 | 触发条件 | 修复方案 |
|---|---|---|---|

### Uncertain（存疑，人工复核）
| 文件:行号 | 问题描述 | 存疑原因 |
|---|---|---|

## Part 2: 优化建议

### 性能优化
| 文件:函数 | 当前问题 | 优化方案 |
|---|---|---|

### 代码结构改进
| 涉及模块 | 当前问题 | 改进方向 |
|---|---|---|

### 可维护性提升
| 涉及模块 | 当前问题 | 改进方向 |
|---|---|---|

约束：
- 禁止修改任何文件（REPORT.md 除外）
- 禁止运行测试、构建或网络请求命令（纯静态只读扫描；以现有 tests/ 内容作为契约证据）
- 禁止给任何文件提交 git commit
- 只读模式扫描

==== 提示词正文结束 ====

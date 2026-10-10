# REPORT_0930.md

## Part 1: Bug Report

### High (强烈建议修)

| 文件:行号 | 问题描述 | 触发条件 | 修复方案 |
|---|---|---|---|
| agent_core/agent.py:224 | 并发 prompt 守卫竞态 (F1, CONFIRMED)：prompt()/continue_() 在 L224/L245 检查 self._state.is_streaming，但该标志只在 _execute 协程内部 (L276) 置 True，而 _execute 是 _run_loop (L260) 用 asyncio.create_task 异步派发、下一拍才运行。同一 Agent 上两个并发 prompt() 都可在任一 _execute 启动前通过守卫，产生两个重叠 _execute 循环，同时写共享_state、同时 _pull_steering_messages/_pull_follow_up_messages（双弹出同一条 steering 消息），且后者覆盖 _running_task，使 wait_for_idle() 只跟踪最后一个任务。 | asyncio.gather(agent.prompt(), agent.prompt(b))，或共享同一 Agent 的两个并发 HTTP handler；两处都在 _execute 前置位读到 is_streaming=False。现有测试 test_prompt_while_streaming_raises 靠 asyncio.sleep(0.02) 掩盖，未覆盖并发路径。 | 把守卫改为“同步置位”：在 _run_loop 里 create_task 之前 (L260 前) 同步 self._state.is_streaming = True；或在 prompt()/continue_() 入口用 self._running_task is not None 作互斥（有未完成 task 则 raise）；并在 _run_loop 的 finally 一起清 _running_task 与 is_streaming。对第二个并发调用应在 _run_loop 立即 raise 而非覆盖_running_task。 |
| pi_ai/providers/openai.py:60 / :69 | 生产者 fire-and-forget，异常未提取 → 消费者可挂死 (红线 #2 对应面，CONFIRMED)：stream() L60 用 asyncio.create_task(_emit) 异步派发生产者协程，返回值/异常无人 await (fire-and-forget)。_emit 内部任何未捕获异常（如 _consume_openai_event_stream L176 的解析/网络异常）只会变成 "task exception was never retrieved" 的系统警告，_stream 的哨兵 end 永不触发。此时 EventStream 的 result() 一直 await 未完成的 future，agent_loop 的 async for 永远不结束，prompt() 挂死。修复思路已在 agent_loop 侧兜底 (_fail_with_error)，但 openai 自身缺对 _emit 任务的异常观察。 | _emit 任务中抛出未捕获异常（网络断开、脏 chunk、解析失败）且异常落在 try 之外。依赖外部 Layer 兜底才不至于挂死，单侧直接驱动 openai.EventStream 时无兜底 → 挂死。 | 给 _emit task 挂 done_callback 观察异常；在生产者协程外层 try/except 捕获异常后，先向 _stream end() 发“error 终态”再正常收尾（与 agent_loop._fail_with_error 同构），确保 result() 一定解析。 |
| pi_ai/providers/openai.py:415 | response.completed 派生的真实 stop_reason/usage 被丢弃 (F7, CONFIRMED)：L415-425 的 response.completed 分支调用 _assistant_from_openai_response 从 OpenAI 返回真实验证 stop_reason/usage，但返回值未与当前 partial 合并、未回写 context——下游拿到的 AssistantMessage 仍是 L822 _new_partial_message 默认的 stop_reason="stop"、usage 缺失。导致：达到 max_tokens 被截断会误标为正常结束；usage/cost 统计偏低；压缩的 token 估算失真。 | 任一触发 response.completed 的请求（所有正常流式请求都经过该分支）。 | 在 response.completed 分支把 _assistant_from_openai_response 得到的 stop_reason / usage 合并回 partial_message 并更新 context.messages[-1]，勿丢弃派生结果。 |
| pi_ai/providers/openai.py:798 | 无 call_id 时回退 item_id 作工具调用 id (F8, CONFIRMED)：_event_call_id L798 在工具调用事件缺少 call_id 时回退返回 item_id。OpenAI 流式工具调用里 item_id（function 条目 id）与 call_id（真正的 function_call_id）是两个不同标识；在 model_end_turn / 工具链跨多条消息引用时，用 item_id 当工具结果回填的关联键，会与 tool_item_id_to_call_id（L150/L502）维护的双 id 映射不一致，导致工具结果错配/找不到 pending 调用。 | 触发 OpenAI/DeepSeek 流式多工具调用且模型未在 delta 中携带 call_id（该分支即为兜底路径）。 | _event_call_id 的 item_id 回退应通过 tool_item_id_to_call_id 映射转成真实 call_id，或显式跳过缺失 call_id 的事件而非用错 id 关联。 |
| tools/paths.py:52 | resolve_in_root 不解析符号链接可越权读外 (F12, CONFIRMED)：resolve_in_root (L52-87) 用 os.path.normpath + is_relative_to (L77) 做边界判断，明确不跟符号链接（模块注释 L47-49 有意为之）。攻击面：root 内存在指向 root 外（如 ~/.ssh/id_rsa）的符号链接文件/目录时，模型经 read/write 工具传入该链接路径，normpath 层面“仍在 root 内”通过校验，实际读/写了 root 外的目标。这是护栏的既定取舍（注释已声明、非近期回归），但对 read/write 工具而言仍是可感知的越权。 | root (cwd) 内存在指向 root 外任意路径的符号链接，且模型被诱导/或试探性地访问它。 | 对 read/write 这类落盘工具，在 resolve_in_root 返回后追加一次 realpath 校验：os.path.realpath(resolved, strict=False) 结果若不落在 root 的 realpath 内则抛 PathOutsideRootError。保持 root is None / allow_outside_root 语义不变。 |

> 补充：F3（工具参数错配可被真执行）定级 High 但归入“单列观察”，未写入 Confirmed 表——其触发依赖真实 DeepSeek 多工具流式时序（completions 实现 _apply_tool_call_delta L352 忽略 choices 内 index，多工具交错时可能把后续 delta 错配给前一工具参数），双 Refuter 一成一致（运行时角无法在本静态环境复现），按规则归 Uncertain 但风险等级高，建议优先排查。

### Uncertain (需复核)

| 文件:行号 | 问题描述 | 触发条件 / 复核点 |
|---|---|---|
| pi_ai/providers/openai_completions.py:203 | error chunk 静默忽略 (F6, UNCERTAIN)：_apply_openai_completions_chunk L203 只读 usage+choices，API 返回 error 字段的 chunk 被丢弃，不终止流。若 error 出现在流中段，下游可能收到残缺结果而不自知。 | 需真实 API 返回错误 chunk 才能确认；静态路径上两者皆成立（一成一致）。 |
| session/session.py:378, :424 | new_entry_id 传入按名字索引的 branch 字典 (F10, UNCERTAIN)：append_to_branch/append_compaction_to_branch 调用 new_entry_id(self._branches) (L378/L424)，该参数语义是“已存在 entry id 集合”，却传入了 branch-name=branch 的字典。ids.py L52 的 cid 检查用 candidate not in existing——字典成员判断等价于“不在键里”，故巧合可用（key 是 branch 名而非 entry id），但语义错位：同 id 生成多次仍需存储层全局校验兜底。 | 正常路径不报错（巧合匹配）；语义脏、依赖存储层二次校验。建议改传局部 id 集合或 None 是否影响正确性。 |

> 已丢弃 (Discard，双反成功)：F9 (completions 流式 'start' 事件缺失——反证非必需)；F5 (persist fire-and-forget gather return exceptions——反证构造方管理，可接受)；F4 (openai 流 'start' 缺失——同 F9 结论)；以及一条关于 session close 时序的推测，均为误报，未列入。

### Medium / Low (去重后代表性条目，未逐条对抗，供优化参考)

- [session/factory.py:227] save_lane_config 与 _restore_lane_config 不对称 (UNCERTAIN)：save 写 model_id/model_api/model_provider/thinking_level/system_prompt (L227-235)，restore 只回读 system_prompt + thinking_level (L262-274)，model 三字段写入后从不恢复。建议对齐（恢复 model 或明确不恢复的意图+注释）。发生率：每次 open 重开会话都会持久化但从不读取。
- [agent_core/event_stream.py:30] 用 get_event_loop().create_future() 绑定当前循环，跨 loop 复用 EventStream 会挂。建议改 asyncio 上下文无关的 future 创建或在创建点固定 loop。
- [session/factory.py:114] _persist_message / flush 的 fire-and-forget 依赖构造方 dispose flush；若构造方不 dispose 则落库可能在应用退出时丢失。建议 dispose 语义文档化。
- [pi_ai/providers/openai_completions.py:~790] usage 存在双计数风险：非流式 fallback 与流式路径对 usage 的累加口径需统一核对。
- [tools/*] read 工具默认 allow_outside_root=False 是正确默认；建议把该默认与 resolve_in_root 的 realpath 加固一处落地，形成单一安全入口。
- [agent_core/agent.py:346] 宽 except 收窄后，错误消息的 stop_reason/usage 语义需与 openai 派生结果保持一致（与 F7/F2 联动）。

## Part 2: 优化建议

### 性能

| 位置 | 现状 | 建议 | 收益 |
|---|---|---|---|
| session/factory.py:150 _maybe_compact | 每轮未扫描整条 branch (scan_branch) 串行做 token 估算 | 复用增量计数（维护 running_token 计数，或仅对新增段估算） | 长会话免去每轮全量扫描 O(n) |
| pi_ai/providers/openai.py | 流式逐事件构造 partial/dict | 索引化事件分发，避免对每个 subevent 线性扫描 | 高吞吐流式时降低 O(events*tools) 开销 |
| tools 读取 | 大量小文件 read 逐个 open | 对 grep/find 聚合结果批量读 + 缓存 meta | 减少 syscall |

### 代码结构

| 位置 | 现状 | 建议 |
|---|---|---|
| agent_core/agent.py:346-370 | 宽 except 与正常收尾逻辑重复（两处构造 error AssistantMessage） | 收敛为单一 _make_error_message 工厂，二处共用 |
| pi_ai/providers/openai.py / openai_completions.py | 两个 provider 各自维护“事件→partial”状态机，逻辑平行 | 抽取共享 streaming-assistant 纯函数层，provider 只做协议翻译 |
| session/* | MutationLine / SessionMutation / 各种 append 方法 | 考虑按官方收敛为更少的张开面，减少重复 barrier 编排 |

### 可维护性

| 位置 | 现状 | 建议 |
|---|---|---|
| tools/paths.py | normpath 边界 + realpath 语义混在一处，注释详尽但行为可选 | 提供 resolve_in_root(..., resolve_symlinks=True) 单一开关，read/write 强制开 |
| 事件类型协议 | agent_start/turn/start/... 散落各 producer | 建 EventType 枚举/常量表，统一 type 字符串拼写，配合 mypy 收窄 |
| 测试 | test_listener_exception_does_not_kill_agent 仍是 NotImplementedError 占位 | 补全：订阅一个抛异常 listener + 健康 listener，断言健康 listener 仍收到全量事件、无伪 error 消息 |
| 测试 | 并发 prompt 只靠 sleep 掩盖 | 加确定性并发测试（两个 prompt 同时发出，断言其一快捷 raise、无重叠执行） |

---

## 附录 A: Phase 3 对抗验证矩阵

对 Phase 2 去重后的 12 条 High finding，各派 2 个独立 Refuter（代码路径角 + 运行时角），各按 "REFUTE-SUCCESS / REFUTE-FAIL"二选一收尾。逐条结果：

| finding | 代码角 | 运行时角 | 裁定 |
|---|---|---|---|
| F1 prompt 并发守卫 (agent.py:224) | FAIL | FAIL | Confirmed |
| F2 _emit 无隔离 (agent.py:400) | FAIL | FAIL | Confirmed |
| F7 response.completed 丢弃派生 (openai.py:415) | FAIL | FAIL | Confirmed |
| F8 item_id 回退工具 id (openai.py:798) | FAIL | FAIL | Confirmed |
| openai 生产者 fire-and-forget 挂死 | FAIL | FAIL | Confirmed |
| F12 resolve_in_root 符号链接 (paths.py:52) | FAIL | FAIL | Confirmed |
| F3 completions 工具 index 忽略 (openai_completions.py:352) | FAIL | SUCCESS | Uncertain |
| F6 completions error chunk 静默 (openai_completions.py:203) | one each |  | Uncertain |
| F10 new_entry_id 传 branch 字典 (session.py:378) | one each |  | Uncertain |
| F9 流式 'start' 缺失 | SUCCESS | SUCCESS | Discard |
| F5 persist gather 包异常 | SUCCESS | SUCCESS | Discard |
| F4 openai 'start' 缺失 | SUCCESS | SUCCESS | Discard |

## 附录 B: 方法与排除范围

- 方法：Phase 1 结构侦察（21 reader 并行产出 46 模块地图 + 全局交叉分析：依赖图 / 事件协议清单 / 共享状态表 / 异步边界 / TS 偏差清单 / 高风险模块排名）→ Phase 2 九维度并行扫描（并发、持久化、JSON 修复、事件流终态、工具边界、资源泄漏、异常路径、类型/契约、TS 对齐），63 findings → 去重 12 High → Phase 3 对抗验证 → Phase 4 汇总。
- 约束遵守：未修改除 REPORT_0930.md 外任何文件；未运行测试/构建/网络（纯静态只读：以 tests/ 内容为契约证据）；未执行任何 git commit；未扫描 NL2SQL/（数据/训练资源）。
- 已知问题核验：#1（agent_loop 生产者异常兜底）已确认修复；#2（_emit 隔离）复确认未修复（见 F2）；#3（abort 无法取消在途 HTTP/SSE）保持开放，未重复入表。
- 已知限制：运行时角 Refuter 无法真连 API，F3/F6/F10 等依赖真实时序/网络路径的判定以代码路径角为主、标 Uncertain，属于本静态方法的固有边界。

> 报告完毕。红色底线（JSON 不砍尾 / 生产者终态保证 / append-only / 加载与重跑工具 / 零耦合 + fire-and-forget / 工具读不越 root）复核均未发现违反。
> 
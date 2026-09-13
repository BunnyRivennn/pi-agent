# Changelog

本 skill 的变更历史。基线版本指 SKILL.md 顶部声明对齐的 `@earendil-works/pi-coding-agent` 版本。

## [基线 v0.85.1] - 2026-09 升级（自 v0.83.0）

对照 v0.85.1 的 `dist/**/*.d.ts` + 包内 CHANGELOG/docs 逐项核实后更新。跨越 0.84.0 ~ 0.85.1 共 7 个版本。

### Changed

- **基线版本** 0.83.0 → 0.85.1：SKILL.md 版本协议、快速开始安装命令、所有 scenario 安装命令统一改为 `@0.85.1`。
- **A04 / sdk_doc 06**：内置工具数量 7 → 8（v0.84.3 新增 Windows 原生 `powershell` 工具）；`createAllTools` / `createAllToolDefinitions` 同步改为 8 个。
- **sdk_doc 05**：新增「v0.84.0 认证/Provider 相关变更补记」表——`refresh()` 的 `ModelsRefreshOptions`/`ModelsRefreshResult`、header 值 `string | null`、`setRuntimeApiKey` options 语义、`ModelsStreamTransforms` → `ModelsRequestTransforms` 更名、手写原生 Provider 的 `ctx.stored` + `ctx.publish()` 事务、OAuth `refreshToken` 必须响应 abort signal、`GoogleThinkingLevel` → `GoogleApiThinkingLevel`。
- **H01**：删除「v0.80.x 不含 `getSystemPromptSource` / `getAppendSystemPromptSources`」的历史兼容警告（0.85.1 接口已稳定包含这两个方法）。
- **D01**：typebox import 说明改为「v0.83.0 起统一，v0.85.1 依然如此」。

### Added

- **版本避坑提示**（SKILL.md）：v0.85.0 误发布内部实验代码导致 SDK import 失败，锁版本应跳过 0.85.0 直接用 0.85.1+。
- **sdk_doc 04 事件系统**：
  - 新事件 `session_compact_failed`（v0.84.3，扩展独有，含 reason/aborted/errorMessage/willRetry/fromExtension）。
  - 新事件 `ui_prompt_start` / `ui_prompt_end`（v0.84.4，扩展独有，区分 agent 工作与等待 `ctx.ui` 交互）。
  - 坑 4 派发分类表扩充：补全其他扩展独有事件与 subscribe 独有事件（含 `entry_appended` / `bash_execution_update`）。
  - **JSON/RPC 模式 `message_update` 破坏性变更警告**（v0.84.0）：`--mode json`/`rpc` 下 delta-only（删累积 `message` 和 `partial`，顶层给 `usage`）；明确进程内 `session.subscribe`/`pi.on` 不受影响（已核实 pi-agent-core 0.85.1 types.d.ts 仍带 `message`）。
- **sdk_doc 12 / F01**：`SessionManager.inMemory(cwd?, options?, entries?)` 第三参（v0.85.0）——从外部存储（DB/对象存储）恢复会话，给出 `parseSessionEntries` + `getEntries` 闭环示例，强化 Web 多租户「内存运行 + 外部落库 + 断点续聊」方案。
- **A04 / B02**：`defaultTools` 设置（v0.84.2）——声明式配置初始内置工具集，与代码层 `tools` 硬白名单的区别、`[]` 语义、项目级替换规则、CLI 标志优先级。
- **C03 / sdk_doc 11**：`AGENTS.override.md` 机制（v0.84.0）——同目录替换 AGENTS.md/CLAUDE.md，其他目录仍正常分层。
- **sdk_doc 07**：补全 `pi.registerMarkdownTransformer()`（v0.84.0）的签名、`MarkdownTransformContext` 字段与链式语义；`pi.sendUserMessage()` 的 `expandPromptTemplates` 选项（v0.84.2）及其与 `session.prompt()` 默认值差异（扩展层默认 false / prompt 默认 true）。
- **sdk_doc 02**：`session.clearQueue()`（v0.84.4）——取出并清空 steering/followUp 队列。

### 核实后明确不改

- 主干 API 描述（`createAgentSession` / `pi.on` / `defineTool` / 扩展独有事件机制）仍然准确。
- E11 SSE 流式集成走进程内 subscribe，不受 message_update JSON/RPC 变更影响。
- 21-multi-agent 的 subagent 模式虽 spawn `pi --mode json`，但用 `message_end`（权威完整消息）取结果，不依赖被删的累积字段。
- sdk_doc 16 推荐的 `createProvider({ fetchModels })` 路径在 0.84.0 事务改造后**无需迁移**，现有描述已对齐。

## [基线 v0.83.0]

前一基线（本文件建立前的历史版本，变更无记录）。

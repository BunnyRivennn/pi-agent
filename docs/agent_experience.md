## 0、Agent harness的开发经验总结

我觉得这个问题如果从 LangGraph、AutoGen、CrewAI、OpenAI Agents SDK、Claude Agent SDK 到底谁最好用开始讨论，很容易跑偏。因为 Agent 框架还在快速变化，今天缺的功能，可能下个月就补上。真正应该先问的是：Agent 到底是什么？一个 Agent 框架究竟应该替我们解决什么问题？想清楚这个，再看框架，很多东西就很清楚了。

## 一、Agent 的本质其实没有那么复杂

我更愿意把 Agent 简化成：

```
当前状态
   ↓
Model
   ↓
下一步 Action
   ↓
Environment
   ↓
Observation
   ↓
新的状态
   ↓
Model
   ↓
下一步 Action
   ...
```

也就是说：Agent 的核心能力，是根据当前状态不断决定“下一步做什么”。这个 Action 可以是：

- 搜索网页
- 读文件
- 写代码
- 运行程序
- 调用 API
- 操作浏览器
- 创建子任务
- 询问用户

所以从这个角度看，LLM 是：Next Token Predictor  
Agent 更像：Next Action Predictor

这也解释了 Agent 为什么能完成复杂任务。它不一定需要一开始就规划未来 100 步。更多时候是：

```
做一步
↓
看结果
↓
重新判断
↓
再做一步
```

这其实和人工作很像。程序员修 Bug，也不会一开始就知道未来 37 个操作分别是什么。而是：看日志 → 怀疑数据库 → 查询 → 排除 → 看缓存 → 修改 → 测试 → 再修改。

Anthropic 对 Agent 和 Workflow 的一个区分也很接近这个意思：Workflow 的路径主要由代码预先规定，而 Agent 是由模型动态决定自己的过程和工具使用。(Anthropic)

## 二、既然 Agent 只是“不断决定下一步”，为什么还需要 Agent 框架？

因为真正困难的部分其实并不是：模型会不会产生一个 Tool Call。这个现在已经很容易了。真正麻烦的是 Tool Call 前后这一大堆东西。

例如模型说：`send_payment(10000)`  
然后发生了网络超时。现在你怎么办？重新调用？但也许付款其实已经成功了，只是结果没有返回。

再比如 Agent 工作了两个小时：搜索 → 下载 → 写代码 → 测试 → 修改 → 再测试  
这时候 Context 满了怎么办？程序崩了怎么办？用户半小时以后才批准某个操作怎么办？模型连续调用同一个 Tool 20 次怎么办？模型说：“任务已经完成。”到底信不信？

所以：Agent Framework 真正解决的，不是“怎么调用 LLM”，而是怎么让一个概率性的模型持续、安全、可靠地作用于现实世界。这也是为什么现在越来越多人开始谈 Harness。

## 三、Harness 到底是什么？

我觉得可以用一句话理解：Model 是大脑，Harness 是大脑赖以工作的整个环境。

一个最简单的 Agent：`LLM + Tools`

真正工程化以后会慢慢长成：

```
Model
                  │
        ┌─────────┴─────────┐
        │                   │
     Context             Actions
        │                   │
     Memory              Tools
        │                   │
     State               Sandbox
        │                   │
        └──── Runtime ──────┘
                 │
        Observation / Eval
                 │
              Recovery
```

这些东西加起来，就是 Harness。

OpenAI 今年专门把这种工作称为 Harness Engineering。他们在使用 Codex 大规模开发软件时发现，很多失败并不是“模型不够聪明”，而是 Agent 缺少能够理解和使用的工具、环境结构和反馈机制。因此工程工作的重点逐渐从“让模型再试一次”，变成了“模型缺少什么能力，怎样让这种能力对 Agent 可见并且可以被机械执行”。(OpenAI)

所以 Harness 的核心思想其实是：不要只优化大脑，也要优化大脑所处的世界。

## 四、那么一个真正的 Agent 框架至少应该解决哪些问题？

我觉得至少有下面几个层次。

### 1. Context：Agent 现在知道什么？

这是最基础的。Agent 必须知道：

- 目标是什么
- 已经完成什么
- 当前状态是什么
- 有哪些约束
- 有哪些可用能力
- 哪些事实已经确认

所以一个 Agent Framework 首先应该有完整的 Context Management：

- system instruction
- conversation
- tool result
- memory
- files
- knowledge
- task state
- context compression
- progressive disclosure

这里最重要的不是：Context Window 越大越好。而是：当前这一步需要知道的信息，是否能够正确进入 Context。

OpenAI 在 Harness Engineering 的实践中甚至专门把巨大的 AGENTS.md 改成了一个“地图”，详细知识放到结构化文档中，让 Agent 按需寻找，而不是一次性塞进上下文。(OpenAI)

### 2. Capability：Agent 能做什么？

Agent 和 Chatbot 最大的区别就是：它能行动。所以框架必须有：

- Tool
- Function Calling
- MCP
- Shell
- Browser
- Filesystem
- Database
- HTTP
- Code Execution

但 Tool 多并不代表框架好。真正的问题是：Agent 能不能在需要的时候找到正确的能力，并且理解如何使用？100 个 Tool 全塞到 Prompt 里，很多时候还不如只给 10 个。因此 Tool discovery、Skill、MCP、渐进式能力加载会越来越重要。

OpenAI 最新 Agents SDK 的运行框架也开始把 Memory、MCP、Skills、AGENTS.md、Shell、文件系统等原语统一纳入 harness。(OpenAI)

### 3. Runtime：Agent Loop 怎么运行？

这是我认为非常重要、但很多所谓 Agent Framework 做得很薄的一层。Demo 可以这样写：

```python
while True:
    response = model(messages)

    if response.tool_call:
        result = run_tool()
        messages.append(result)
    else:
        break
```

几十行代码就能做一个 Agent。真正运行以后会发现需要处理：

- timeout
- retry
- cancel
- parallel tool calls
- write barrier
- step limit
- duplicate calls
- stuck detection
- streaming
- approval
- checkpoint
- resume

这时候才真正进入：Agent Runtime 而不是 LLM SDK。

## 五、第四个问题可能更加重要：Agent 怎么知道自己做对了？

假设 Agent 写了一段代码。最差的 Harness 是：

```
写代码
↓
模型觉得不错
↓
完成
```

好的 Harness 是：

```
写代码
↓
编译
↓
测试
↓
失败
↓
读取错误
↓
修改
↓
测试
↓
Code Review
↓
完成
```

区别在哪里？不是模型变聪明了。而是：反馈回路变好了。

所以一个 Agent 系统非常核心的东西应该是：

```
Action
↓
Observation
↓
Evaluation
↓
Next Action
```

Agent 并不一定要第一次做对。关键是：系统有没有能力让它知道自己错了。某种意义上，这甚至比 Planner 更重要。

## 六、第五个问题：长任务怎么活下去？

这可能是下一代 Agent Framework 真正开始拉开差距的地方。LLM 调用天然是：短生命周期。但现实任务可能持续：10 分钟、2 小时、2 天。于是会出现：

- Context 满
- 进程退出
- 网络失败
- 机器重启
- 等待人工批准
- 等待外部系统

Anthropic 在研究 long-running agent 时专门指出：仅靠 Context Compaction 并不能解决长任务，Agent 还需要通过任务列表、文件和其他结构化 Artifact 把工作状态传递到下一次上下文中。(Anthropic)

所以一个成熟 Agent Framework 最终一定会出现：

- State
- Checkpoint
- Artifact
- TaskResult
- Memory
- Resume
- Retry
- Idempotency

你会发现：Agent Framework 做到后面，越来越不像一个“AI 库”。反而越来越像：操作系统 + 工作流引擎 + 数据库 + 分布式系统。

## 七、第六个问题：概率模型和确定性软件怎么划边界？

这是我认为判断 Agent Framework 水平非常重要的一条。

例如：“不要重复提交订单。”你可以写进 Prompt。但好的 Harness 应该做：`operation_id + idempotency`

再比如：“修改代码以后一定要测试。”当然可以告诉模型。但更可靠的是：CI 不通过 → 任务不能标记完成

再比如：“付款之前请慎重。”不如：`payment → human approval`

所以我觉得 Agent Engineering 一个非常重要的原则是：能够由软件确定性保证的事情，不要交给 LLM 自觉。

模型负责：判断和决策。  
Harness 负责：边界和不变量。

## 八、这也解释了为什么 Harness 会出现

LLM 有三个天然特征：

1. 它是概率性的。同样 Context，两次结果可能不同。
2. 它本身基本是无状态的。真正长期存在的 State 必须放在模型之外。
3. 它不知道现实世界到底发生了什么。除非 Harness 把 Observation 重新告诉它。

但是现实中的复杂任务恰恰需要：

- 长期状态
- 确定性约束
- 现实反馈
- 错误恢复
- 副作用控制

所以：Harness 本质上就是给概率性智能搭建一个确定性的外骨骼。

模型负责产生智能。Harness 负责让这种智能：能够安全、稳定、长期地兑现成结果。

## 九、那么到底怎么判断一个 Agent 框架好不好？

我自己不会首先看：Star 数多少？也不会首先看：支持几个模型、几个 Agent、几个 Tool？我会问下面几个问题。

### 1. 模型能不能真正决定下一步？

如果所有路径都已经写死：A → B → C → D，那它更像 Workflow Engine，而不是通用 Agent Runtime。

### 2. Context 管理是否是一等公民？

有没有：
- Memory
- Compaction
- Artifact
- Knowledge
- Task State
- Context Budget

而不是简单粗暴地：`messages.append(...)`

### 3. Tool 有没有真正的执行语义？

不只是 function calling，还应该考虑：
- 权限
- 副作用
- 并发
- 超时
- 取消
- 重试
- 去重
- 审批

### 4. 有没有 Feedback Loop？

Agent 做完事情以后：谁告诉它到底做对没有？如果只能依靠模型自评，可靠性上限会很低。

### 5. 长任务能不能恢复？

Context Window 用完、进程退出以后：任务还能不能继续？如果不能，很多所谓 long-running Agent 本质上只是一个很长的 while loop。

### 6. State 是否独立于模型？

今天 GPT，明天 Claude，后天 DeepSeek。换模型以后：任务、Memory、文件、Tool、运行状态还在不在？模型应该是可以替换的计算资源，而不是整个 Agent 本身。

### 7. 能不能限制模型，而不是只劝模型？

这是我特别看重的一点。好的框架应该能够表达：

- 这个 Tool 不允许调用
- 这个操作必须审批
- 这个 Action 不能 Retry
- 这个结果必须经过测试

而不是全部写进 Prompt：“请注意……”

### 8. 可观察性怎么样？

一个 Agent 运行两小时失败了，你至少应该能够回答：

- 它做过什么？
- 为什么这么做？
- Tool 返回了什么？
- 在哪里开始失败？
- 重试过多少次？
- 当前 State 是什么？

否则生产环境基本无法 Debug。

## 十、因此，“最好用的 Agent 框架是什么？”其实没有一个固定答案

如果任务非常确定：提取 → 分类 → 总结 → 写数据库，那么可能根本不应该使用复杂 Agent。一个 Workflow 就够了。Anthropic 很早就建议：优先使用最简单的可行方案，只有真正需要模型动态决策时才增加 Agent 复杂度。(Anthropic)

如果你的任务是：Coding Agent  
你会特别看重：Shell、Filesystem、Git、Tests、Checkpoint、Code Review

如果是 Browser Agent：DOM、Screenshot、Tab、Permission、Download、Authentication、Human Approval

如果是企业 Agent：Identity、Permission、Audit、Task、Human Gate、Long-running State

所以不存在脱离场景的“最好 Agent Framework”。但我觉得存在一个越来越清晰的判断方向：不要看一个框架能不能把模型调起来，要看它能不能把模型的“下一步判断”，变成一个可靠的闭环系统。

最后如果让我用一句话定义一个好的 Agent Framework：它应该尽可能释放模型的决策能力，同时用确定性的 Harness 管理状态、环境、工具、反馈、权限和失败。

过度约束，就退化成 Workflow。约束太少，就变成一个不断 Tool Calling 的随机游走。

真正好的 Agent Framework 应该处在中间：

```
Model
        负责判断下一步
               │
               ▼
        ┌──────────────┐
        │   Harness    │
        │ Context      │
        │ Tools        │
        │ State        │
        │ Memory       │
        │ Evaluation   │
        │ Recovery     │
        │ Security     │
        └──────────────┘
               │
               ▼
          现实世界
```

模型越强，Framework 越不应该替模型“想”。但与此同时：模型越强、能够做的事情越多，Harness 反而越重要。

因为当一个模型只能写几句话的时候，出错最多是回答错了。当一个 Agent 可以：修改代码、删除文件、操作浏览器、调用数据库、发送邮件、花钱，问题就完全不同了。

所以我认为 Agent Framework 下一阶段真正竞争的，恐怕不是：谁包装的 Agent 类最多。而是：谁能用最少但最有效的 Harness，把模型的智能稳定地转化成现实世界中的工作成果。

这可能才是判断一个 Agent 框架好坏最有用的标准。



import json, time
from pathlib import Path

class TrajectoryLogger:
    def __init__(self, path: Path):
        self._fp = path.open("a", encoding="utf-8")

    def __call__(self, event: dict) -> None:
        try:
            self._fp.write(json.dumps(
                {"ts": time.time(), **event},
                ensure_ascii=False, default=str,
            ) + "\n")
            self._fp.flush()
        except Exception:
            pass   # 日志坏了不能把 agent 带走

    def close(self) -> None:
        self._fp.close()

logger = TrajectoryLogger(Path("trajectory.jsonl"))
unsub = agent.subscribe(logger)

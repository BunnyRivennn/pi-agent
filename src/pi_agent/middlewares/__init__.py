"""中间件：挂 hooks，会改数据流。

本包放通过 ``session.hooks.on(...)`` 挂上的 handler。它们能改流经的数据、能
喊停，所以坏了会改变 agent 的行为——与 ``extensions`` 的只读定位相反。

已有：

- ``memory.py``     长期记忆/用户画像。两头都要：注入走 ``before_run``（每轮
                    开头往 prompt 里塞画像），提取走 ``before_run_end``（一轮
                    跑完拿到完整消息）。两个点都接好了，挂上即生效。
                    提取逻辑目前是规则版，待梳理清楚后换模型调用。

按计划的落位：
- ``permission.py`` 权限门，走 ``before_tool``。危险命令返回
                    ``{"block": {"reason": ...}}``。这是 hook 独有的能力——
                    注入和订阅都做不到「喊停」。
- ``redaction.py``  工具结果脱敏，走 ``after_tool``。

写之前先确认插入点接了没。已接：``before_run`` / ``before_run_end``（都在
``agent_loop.py``）、``before_compaction``（在 ``session/factory.py``）。未接：
``transform_context`` / ``before_payload`` / ``after_response`` /
``before_tool`` / ``after_tool``——运行器方法写好了但循环里还没调用，挂上去是
静默不响的，不报错。

摘要**不**在这里：它是压缩实现的一部分，走依赖注入（``pi_agent.summarizer``
的 ``LlmSummarizer``）。判断标准是「卸掉它功能还在不在」——摘要器卸掉压缩就
停了，那是实现；中间件卸掉 agent 照跑，只是少一层加工。
"""

from __future__ import annotations

from .memory import (
    ExtractProfile,
    JsonProfileStore,
    MemoryMiddleware,
    ProfileStore,
    rule_extract,
)

__all__ = [
    "ExtractProfile",
    "JsonProfileStore",
    "MemoryMiddleware",
    "ProfileStore",
    "rule_extract",
]

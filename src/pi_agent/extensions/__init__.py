"""扩展：只观察、不改数据流。

本包的东西全部通过 ``agent.subscribe()`` 挂上，只能看见事件流，改不了任何东西。
日志、监控、统计、UI 驱动属于这里。

与 ``middlewares`` 的分界线是**读写**：
- ``extensions``  → ``subscribe``，只读。坏了不影响 agent 的产出。
- ``middlewares`` → hooks，会改数据流。坏了会改变 agent 的行为。

两者都是叶子包：谁都不 import 它们，宿主程序挑着装，核心不知道它们存在。
"""

from __future__ import annotations

from .trajectory import TrajectoryLogger

__all__ = ["TrajectoryLogger"]

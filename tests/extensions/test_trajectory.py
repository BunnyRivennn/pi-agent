"""轨迹落盘测试：写入格式 + 失败不扩散。"""

from __future__ import annotations

import json
from pathlib import Path

from pi_agent.extensions import TrajectoryLogger


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_逐条写成jsonl(tmp_path: Path) -> None:
    log = TrajectoryLogger(tmp_path / "t.jsonl")
    log({"type": "agent_start"})
    log({"type": "turn_end", "tool_results": []})
    log.close()

    lines = _read_lines(tmp_path / "t.jsonl")
    assert [e["type"] for e in lines] == ["agent_start", "turn_end"]
    assert all(isinstance(e["ts"], float) for e in lines)
    assert log.written == 2
    assert log.dropped == 0


def test_不可序列化对象不丢事件(tmp_path: Path) -> None:
    """default=str 兜底：message 对象进来也得落盘，不能整条丢掉。"""

    class Weird:
        def __str__(self) -> str:
            return "<weird>"

    log = TrajectoryLogger(tmp_path / "t.jsonl")
    log({"type": "message_end", "message": Weird()})
    log.close()

    (entry,) = _read_lines(tmp_path / "t.jsonl")
    assert entry["message"] == "<weird>"
    assert log.dropped == 0


def test_中文不转义(tmp_path: Path) -> None:
    log = TrajectoryLogger(tmp_path / "t.jsonl")
    log({"type": "text_delta", "text": "你好"})
    log.close()

    assert "你好" in (tmp_path / "t.jsonl").read_text(encoding="utf-8")


def test_自动建父目录(tmp_path: Path) -> None:
    log = TrajectoryLogger(tmp_path / "a" / "b" / "t.jsonl")
    log({"type": "agent_start"})
    log.close()
    assert (tmp_path / "a" / "b" / "t.jsonl").exists()


def test_续写不覆盖(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    first = TrajectoryLogger(p)
    first({"type": "agent_start"})
    first.close()

    second = TrajectoryLogger(p)
    second({"type": "agent_end"})
    second.close()

    assert [e["type"] for e in _read_lines(p)] == ["agent_start", "agent_end"]


def test_关闭后不抛异常只计数(tmp_path: Path) -> None:
    """日志坏了不能把 agent 带走——listener 抛异常会炸事件流。"""
    log = TrajectoryLogger(tmp_path / "t.jsonl")
    log.close()
    log({"type": "agent_start"})  # 不抛
    assert log.dropped == 1
    assert log.written == 0


def test_上报自己抛异常也不扩散(tmp_path: Path) -> None:
    def bad_reporter(_exc: Exception, _event: dict) -> None:
        raise RuntimeError("上报也炸了")

    log = TrajectoryLogger(tmp_path / "t.jsonl", on_error=bad_reporter)
    log.close()
    log({"type": "agent_start"})  # 不抛
    assert log.dropped == 1


def test_可作上下文管理器(tmp_path: Path) -> None:
    with TrajectoryLogger(tmp_path / "t.jsonl") as log:
        log({"type": "agent_start"})
    assert log.written == 1
    log({"type": "agent_end"})  # 已关闭
    assert log.dropped == 1

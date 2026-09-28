"""内置工具集的测试。

重点覆盖：

- 注册表的注册/查找/预设套餐；
- ``edit`` 的三条硬约束（唯一匹配、不重叠、对原文匹配）与格式兼容层；
- BOM 与 CRLF 保持；
- 路径护栏；
- 同文件写串行化；
- ``read`` 的 GBK 探测与目录列举；
- ``bash`` 的退出码/超时/输出截断。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_agent.tools import (
    PathOutsideRootError,
    build_tools_prompt,
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_read_tool,
    create_tool,
    create_tools,
    create_write_tool,
    register_tool,
    tool_names,
)
from pi_agent.tools.edit import prepare_edit_arguments
from pi_agent.tools.edit_diff import Edit, apply_edits
from pi_agent.tools.mutation_queue import with_file_mutation_lock


async def _run(tool, params):
    """调用工具并返回首个文本块。"""
    result = await tool.execute("call-1", params, None, None)
    return result.content[0].text, result.details


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_四个内置工具都已注册():
    names = tool_names()
    assert set(names) >= {"read", "bash", "edit", "write"}


def test_coding套餐顺序固定(tmp_path: Path):
    tools = create_coding_tools(tmp_path)
    assert [t.name for t in tools] == ["read", "bash", "edit", "write"]


def test_未知工具名报错并列出可用工具(tmp_path: Path):
    with pytest.raises(KeyError, match="Unknown tool: nope"):
        create_tool("nope", tmp_path)


def test_重复注册同名工具被拒绝():
    with pytest.raises(ValueError, match="Tool already registered"):

        @register_tool("read")
        def _dup(cwd, **kwargs):  # pragma: no cover - 只验证抛错
            raise AssertionError


def test_自定义工具可注册并取用(tmp_path: Path):
    from pi_agent.agent_core.types import AgentTool, AgentToolResult, TextContent

    @register_tool("custom_probe")
    def _create(cwd: Path, *, suffix: str = "!") -> AgentTool:
        async def execute(tool_call_id, params, abort_event=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(text=f"{params['msg']}{suffix}")], details={}
            )

        return AgentTool(
            name="custom_probe", label="probe", description="test", execute=execute
        )

    tool = create_tool("custom_probe", tmp_path, {"suffix": "?"})
    assert tool.name == "custom_probe"
    assert "custom_probe" in tool_names()


def test_提示词由启用的工具拼装(tmp_path: Path):
    prompt = build_tools_prompt(create_tools(["read", "write"], tmp_path))
    assert "read:" in prompt
    assert "write:" in prompt
    assert "Guidelines:" in prompt
    # 没启用的工具不该出现
    assert "bash:" not in prompt


# ---------------------------------------------------------------------------
# edit 的匹配约束
# ---------------------------------------------------------------------------


def test_apply_edits拒绝多处匹配():
    with pytest.raises(ValueError, match="matched 2 places"):
        apply_edits("foo\nfoo\n", [Edit("foo", "bar")], "t.py")


def test_apply_edits拒绝匹配不到():
    with pytest.raises(ValueError, match="did not match"):
        apply_edits("hello\n", [Edit("nope", "x")], "t.py")


def test_apply_edits拒绝重叠区间():
    content = "abcdef"
    with pytest.raises(ValueError, match="overlap"):
        apply_edits(content, [Edit("abcd", "X"), Edit("cdef", "Y")], "t.py")


def test_apply_edits拒绝空oldText():
    with pytest.raises(ValueError, match="empty oldText"):
        apply_edits("abc", [Edit("", "x")], "t.py")


def test_apply_edits拒绝无变化():
    with pytest.raises(ValueError, match="no change"):
        apply_edits("abc", [Edit("abc", "abc")], "t.py")


def test_多处edit都对原文匹配而非增量():
    # 若是增量应用，第二个 edit 会在已替换的文本上找不到目标。
    content = "alpha\nbeta\n"
    result = apply_edits(content, [Edit("alpha", "beta"), Edit("beta", "alpha")], "t.py")
    assert result.new_content == "beta\nalpha\n"


# ---------------------------------------------------------------------------
# edit 的参数兼容层
# ---------------------------------------------------------------------------


def test_prepare把JSON字符串还原成数组():
    out = prepare_edit_arguments({"path": "a", "edits": '[{"oldText":"a","newText":"b"}]'})
    assert out["edits"] == [{"oldText": "a", "newText": "b"}]


def test_prepare把单个对象包成数组():
    out = prepare_edit_arguments({"path": "a", "edits": {"oldText": "a", "newText": "b"}})
    assert out["edits"] == [{"oldText": "a", "newText": "b"}]


def test_prepare把顶层平铺参数并进数组():
    out = prepare_edit_arguments({"path": "a", "oldText": "a", "newText": "b"})
    assert out["edits"] == [{"oldText": "a", "newText": "b"}]
    assert "oldText" not in out


def test_prepare修不动就原样返回():
    raw = {"path": "a", "edits": "not json at all"}
    assert prepare_edit_arguments(raw)["edits"] == "not json at all"


# ---------------------------------------------------------------------------
# edit / write 的文件行为
# ---------------------------------------------------------------------------


async def test_edit保持CRLF行尾(tmp_path: Path):
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"one\r\ntwo\r\n")
    tool = create_edit_tool(tmp_path)

    await _run(tool, {"path": "crlf.txt", "edits": [{"oldText": "two", "newText": "three"}]})

    assert target.read_bytes() == b"one\r\nthree\r\n"


async def test_edit保持BOM(tmp_path: Path):
    target = tmp_path / "bom.txt"
    target.write_bytes("\ufeffhello\n".encode())
    tool = create_edit_tool(tmp_path)

    await _run(tool, {"path": "bom.txt", "edits": [{"oldText": "hello", "newText": "bye"}]})

    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"bye\n")


async def test_edit不存在的文件报错(tmp_path: Path):
    tool = create_edit_tool(tmp_path)
    with pytest.raises(RuntimeError, match="does not exist"):
        await _run(tool, {"path": "missing.txt", "edits": [{"oldText": "a", "newText": "b"}]})


async def test_write创建父目录(tmp_path: Path):
    tool = create_write_tool(tmp_path)
    await _run(tool, {"path": "deep/nested/f.txt", "content": "hi"})
    assert (tmp_path / "deep" / "nested" / "f.txt").read_text() == "hi"


# ---------------------------------------------------------------------------
# 路径护栏
# ---------------------------------------------------------------------------


async def test_越界路径被拒绝(tmp_path: Path):
    workdir = tmp_path / "project"
    workdir.mkdir()
    (tmp_path / "secret.txt").write_text("classified")

    tool = create_read_tool(workdir, root=workdir)
    with pytest.raises(PathOutsideRootError, match="escapes the allowed root"):
        await _run(tool, {"path": "../secret.txt"})


async def test_显式放行后可越界(tmp_path: Path):
    workdir = tmp_path / "project"
    workdir.mkdir()
    (tmp_path / "secret.txt").write_text("classified")

    tool = create_read_tool(workdir, root=workdir, allow_outside_root=True)
    text, _ = await _run(tool, {"path": "../secret.txt"})
    assert "classified" in text


async def test_不设root时不限制(tmp_path: Path):
    workdir = tmp_path / "project"
    workdir.mkdir()
    (tmp_path / "outside.txt").write_text("fine")

    tool = create_read_tool(workdir)
    text, _ = await _run(tool, {"path": "../outside.txt"})
    assert "fine" in text


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


async def test_read探测GBK编码(tmp_path: Path):
    target = tmp_path / "gbk.txt"
    target.write_bytes("中文内容测试".encode("gb18030"))

    tool = create_read_tool(tmp_path)
    text, _ = await _run(tool, {"path": "gbk.txt"})

    assert "中文内容测试" in text
    assert "gb18030" in text  # 标注实际编码，让模型知道读到的是什么


async def test_read列目录(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("x")

    tool = create_read_tool(tmp_path)
    text, details = await _run(tool, {"path": "."})

    assert details["kind"] == "directory"
    assert "sub/" in text
    assert "a.txt" in text


async def test_read按offset分页(tmp_path: Path):
    target = tmp_path / "many.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 11)))

    tool = create_read_tool(tmp_path)
    text, _ = await _run(tool, {"path": "many.txt", "offset": 5, "limit": 2})

    assert "line5" in text
    assert "line6" in text
    assert "line7" not in text


async def test_read缺失文件报错(tmp_path: Path):
    tool = create_read_tool(tmp_path)
    with pytest.raises(RuntimeError, match="File not found"):
        await _run(tool, {"path": "nope.txt"})


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


async def test_bash返回输出(tmp_path: Path):
    tool = create_bash_tool(tmp_path)
    text, details = await _run(tool, {"command": "echo hello-from-bash"})
    assert "hello-from-bash" in text
    assert details["exit_code"] == 0


async def test_bash报告非零退出码(tmp_path: Path):
    tool = create_bash_tool(tmp_path)
    text, details = await _run(tool, {"command": "exit 3"})
    assert details["exit_code"] == 3
    assert "exit code: 3" in text


async def test_bash超时会杀掉进程(tmp_path: Path):
    tool = create_bash_tool(tmp_path, default_timeout=0.5)
    text, details = await _run(tool, {"command": "sleep 10"})
    assert details["timed_out"] is True
    assert "timed out" in text


async def test_bash截断超长输出(tmp_path: Path):
    tool = create_bash_tool(tmp_path, max_output_bytes=200)
    text, _ = await _run(tool, {"command": "for i in $(seq 1 500); do echo padding-line-$i; done"})
    assert "omitted" in text
    assert len(text) < 1000


# ---------------------------------------------------------------------------
# 同文件写串行化
# ---------------------------------------------------------------------------


async def test_同文件写操作串行执行(tmp_path: Path):
    target = tmp_path / "shared.txt"
    order: list[str] = []

    async def slow(tag: str):
        async def _work():
            order.append(f"{tag}-start")
            await asyncio.sleep(0.05)
            order.append(f"{tag}-end")

        await with_file_mutation_lock(target, _work)

    await asyncio.gather(slow("a"), slow("b"))

    # 串行的话一定是 x-start, x-end, y-start, y-end，不会交错。
    assert order in (
        ["a-start", "a-end", "b-start", "b-end"],
        ["b-start", "b-end", "a-start", "a-end"],
    )


async def test_不同文件写操作可并行(tmp_path: Path):
    order: list[str] = []

    async def slow(name: str):
        async def _work():
            order.append(f"{name}-start")
            await asyncio.sleep(0.05)
            order.append(f"{name}-end")

        await with_file_mutation_lock(tmp_path / name, _work)

    await asyncio.gather(slow("a.txt"), slow("b.txt"))

    # 并行则两个 start 都排在两个 end 之前。
    assert order[0].endswith("-start")
    assert order[1].endswith("-start")

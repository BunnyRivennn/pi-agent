"""``grep`` / ``find`` / ``ls`` / ``powershell`` 四个工具的测试。

重点覆盖：

- ``ls`` 的排序、目录后缀、条目上限；
- ``find`` 的 glob 语义（裸模式匹配文件名、含斜杠匹配相对路径）与 gitignore；
- ``grep`` 的正则/字面量、忽略大小写、上下文行、匹配上限、二进制跳过、GBK 搜索；
- 忽略规则的取反、目录限定、剪枝；
- ``powershell`` 缺失时报工具错误而非创建失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pi_agent.tools import (
    ALL_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_powershell_tool,
    create_read_only_tools,
    tool_names,
)
from pi_agent.tools.find import matches_glob
from pi_agent.tools.ignore import IgnoreRules, load_ignore_rules, walk_files
from pi_agent.tools.truncate import truncate_head, truncate_line


async def _run(tool, params):
    result = await tool.execute("call-1", params, None, None)
    return result.content[0].text, result.details


@pytest.fixture
def 项目树(tmp_path: Path) -> Path:
    """搭一棵有代表性的目录树：含嵌套、dotfile、被忽略目录、非 UTF-8 文件。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\n\ndef main():\n    print('hello')\n", encoding="utf-8"
    )
    (tmp_path / "src" / "util.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "def test_main():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Demo\n\nhello world\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")

    # 应被 ALWAYS_IGNORED 剪掉
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("hello\n", encoding="utf-8")

    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n!keep.log\n", encoding="utf-8")
    (tmp_path / "debug.log").write_text("noise\n", encoding="utf-8")
    (tmp_path / "keep.log").write_text("hello kept\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_八个内置工具全部注册():
    assert set(tool_names()) == set(ALL_TOOL_NAMES)
    assert len(ALL_TOOL_NAMES) == 8


def test_只读套餐不含写工具(tmp_path: Path):
    tools = create_read_only_tools(tmp_path)
    names = [t.name for t in tools]
    assert names == list(READ_ONLY_TOOL_NAMES)
    assert not ({"write", "edit", "bash", "powershell"} & set(names))


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


async def test_ls按字母序列出且目录带斜杠(项目树: Path):
    text, details = await _run(create_ls_tool(项目树), {})
    lines = text.splitlines()
    assert "src/" in lines
    assert "README.md" in lines
    assert ".env" in lines, "dotfile 也要列出来"
    assert lines == sorted(lines, key=str.lower)
    assert details["count"] == len(lines)


async def test_ls空目录给出明确提示(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    text, _ = await _run(create_ls_tool(tmp_path), {"path": "empty"})
    assert text == "(empty directory)"


async def test_ls条目超限时提示如何拿更多(tmp_path: Path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    text, details = await _run(create_ls_tool(tmp_path), {"limit": 3})
    assert details["entry_limit_reached"] == 3
    assert "limit=6" in text, "要告诉模型下一步怎么办，而不是沉默截断"


async def test_ls对文件路径报错(项目树: Path):
    with pytest.raises(RuntimeError, match="Not a directory"):
        await _run(create_ls_tool(项目树), {"path": "README.md"})


async def test_ls对不存在路径报错(项目树: Path):
    with pytest.raises(RuntimeError, match="Path not found"):
        await _run(create_ls_tool(项目树), {"path": "nope"})


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


def test_glob裸模式匹配文件名而非整路径():
    # 模型写 '*.py' 时想要的是所有层级的 .py
    assert matches_glob("src/app.py", "*.py")
    assert matches_glob("app.py", "*.py")
    assert not matches_glob("src/app.txt", "*.py")


def test_glob含斜杠时匹配相对路径():
    assert matches_glob("src/app.py", "src/*.py")
    assert not matches_glob("tests/app.py", "src/*.py")


def test_glob双星号可匹配零层目录():
    assert matches_glob("src/app.py", "src/**/*.py")
    assert matches_glob("src/a/b.py", "src/**/*.py")


async def test_find找到所有python文件(项目树: Path):
    text, details = await _run(create_find_tool(项目树), {"pattern": "*.py"})
    found = set(text.splitlines())
    assert found == {"src/app.py", "src/util.py", "tests/test_app.py"}
    assert details["count"] == 3


async def test_find跳过node_modules(项目树: Path):
    text, _ = await _run(create_find_tool(项目树), {"pattern": "*.js"})
    assert "No files matching" in text, "node_modules 必须被剪枝"


async def test_find尊重gitignore(项目树: Path):
    text, _ = await _run(create_find_tool(项目树), {"pattern": "*.log"})
    assert "debug.log" not in text
    assert "keep.log" in text, "! 取反规则应把 keep.log 捞回来"


async def test_find可关闭gitignore(项目树: Path):
    tool = create_find_tool(项目树, use_gitignore=False)
    text, _ = await _run(tool, {"pattern": "*.log"})
    assert "debug.log" in text


async def test_find无结果时不报错(项目树: Path):
    text, details = await _run(create_find_tool(项目树), {"pattern": "*.rs"})
    assert "No files matching" in text
    assert details["count"] == 0


async def test_find超限时提示如何拿更多(项目树: Path):
    text, details = await _run(create_find_tool(项目树), {"pattern": "*.py", "limit": 2})
    assert details["result_limit_reached"] == 2
    assert "limit=4" in text


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


async def test_grep返回路径与行号(项目树: Path):
    text, details = await _run(create_grep_tool(项目树), {"pattern": "def main"})
    assert "src/app.py:4: def main():" in text
    assert details["count"] == 1


async def test_grep支持正则(项目树: Path):
    text, _ = await _run(create_grep_tool(项目树), {"pattern": r"def \w+\("})
    assert "app.py" in text and "util.py" in text


async def test_grep字面量模式不把点当通配(项目树: Path):
    (项目树 / "lit.txt").write_text("a.b\naxb\n", encoding="utf-8")
    text, details = await _run(
        create_grep_tool(项目树), {"pattern": "a.b", "literal": True, "glob": "lit.txt"}
    )
    assert details["count"] == 1
    assert "a.b" in text and "axb" not in text


async def test_grep非法正则给出可操作的报错(项目树: Path):
    with pytest.raises(RuntimeError, match="literal=true"):
        await _run(create_grep_tool(项目树), {"pattern": "a[b"})


async def test_grep忽略大小写(项目树: Path):
    text, _ = await _run(create_grep_tool(项目树), {"pattern": "HELLO", "ignoreCase": True})
    assert "README.md" in text


async def test_grep上下文行用短横线区分(项目树: Path):
    text, _ = await _run(create_grep_tool(项目树), {"pattern": "def main", "context": 1})
    # 匹配行用 ':'，上下文行用 '-'
    assert "src/app.py:4: def main():" in text
    assert "src/app.py-5- " in text


async def test_grep按glob过滤(项目树: Path):
    text, _ = await _run(create_grep_tool(项目树), {"pattern": "def", "glob": "*.py"})
    assert "README.md" not in text


async def test_grep尊重gitignore(项目树: Path):
    text, _ = await _run(create_grep_tool(项目树), {"pattern": "noise"})
    assert "No matches found" in text


async def test_grep跳过二进制文件(项目树: Path):
    (项目树 / "blob.bin").write_bytes(b"hello\x00\x00binary")
    text, _ = await _run(create_grep_tool(项目树), {"pattern": "hello"})
    assert "blob.bin" not in text


async def test_grep能搜到GBK中文文件(项目树: Path):
    # ripgrep 会把 GBK 当二进制跳过；我们复用 read 的编码探测，应该能搜到。
    (项目树 / "cn.txt").write_bytes("你好世界\n项目说明\n".encode("gb18030"))
    text, details = await _run(create_grep_tool(项目树), {"pattern": "项目"})
    assert "cn.txt" in text
    assert details["count"] == 1


async def test_grep超限时提示如何拿更多(项目树: Path):
    text, details = await _run(create_grep_tool(项目树), {"pattern": "e", "limit": 2})
    assert details["match_limit_reached"] == 2
    assert "limit=4" in text


async def test_grep无匹配时不报错(项目树: Path):
    text, details = await _run(create_grep_tool(项目树), {"pattern": "zzz_not_there"})
    assert text == "No matches found"
    assert details["count"] == 0


async def test_grep可搜索单个文件(项目树: Path):
    text, _ = await _run(
        create_grep_tool(项目树), {"pattern": "hello", "path": "README.md"}
    )
    assert "README.md" in text


async def test_grep截断超长行(tmp_path: Path):
    (tmp_path / "long.txt").write_text("x" * 5000 + "needle\n", encoding="utf-8")
    text, details = await _run(create_grep_tool(tmp_path), {"pattern": "needle"})
    assert details["lines_truncated"] is True
    assert "read tool" in text, "应引导模型用 read 看全行"


# ---------------------------------------------------------------------------
# 忽略规则
# ---------------------------------------------------------------------------


def test_忽略规则支持目录限定():
    from pi_agent.tools.ignore import _parse_line

    rules = IgnoreRules()
    pattern = _parse_line("build/")
    assert pattern is not None
    rules.patterns.append(pattern)

    # 'build/' 只忽略目录，同名文件应当保留
    assert rules.is_ignored("build", is_dir=True)
    assert not rules.is_ignored("build", is_dir=False)


def test_忽略规则后者覆盖前者():
    from pi_agent.tools.ignore import _parse_line

    rules = IgnoreRules()
    for line in ("*.log", "!keep.log"):
        pattern = _parse_line(line)
        assert pattern is not None
        rules.patterns.append(pattern)

    assert rules.is_ignored("debug.log", is_dir=False)
    assert not rules.is_ignored("keep.log", is_dir=False)


def test_忽略规则跳过注释与空行(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("# comment\n\n*.tmp\n", encoding="utf-8")
    rules = load_ignore_rules(tmp_path)
    assert len(rules.patterns) == 1
    assert rules.is_ignored("a.tmp", is_dir=False)


def test_忽略规则默认剪掉重目录(tmp_path: Path):
    rules = load_ignore_rules(tmp_path)
    assert rules.is_ignored("node_modules", is_dir=True)
    assert rules.is_ignored("a/b/__pycache__", is_dir=True)
    assert rules.is_ignored(".git/config", is_dir=False)


def test_遍历会剪枝而非逐个过滤(项目树: Path):
    rules = load_ignore_rules(项目树)
    relatives = [rel for _abs, rel in walk_files(项目树, rules)]
    assert not any(r.startswith("node_modules") for r in relatives)
    assert "src/app.py" in relatives


# ---------------------------------------------------------------------------
# 截断
# ---------------------------------------------------------------------------


def test_截断保留开头并标记():
    result = truncate_head("a" * 100, max_bytes=10)
    assert result.truncated is True
    assert len(result.content) == 10
    assert result.original_bytes == 100


def test_截断不切坏多字节字符():
    # 中文每字 3 字节，从 4 字节处切会切坏第二个字
    result = truncate_head("中文中文", max_bytes=4)
    assert result.content == "中", "残片必须被丢掉而不是变成乱码"


def test_未超限时不标记截断():
    result = truncate_head("short")
    assert result.truncated is False
    assert result.content == "short"


def test_行截断加省略号():
    text, was_cut = truncate_line("x" * 500, 100)
    assert was_cut is True
    assert text.endswith("…")
    assert len(text) == 101


# ---------------------------------------------------------------------------
# powershell
# ---------------------------------------------------------------------------


def test_powershell工具创建不因平台失败(tmp_path: Path):
    # 关键：即使系统没有 PowerShell，创建也必须成功——
    # 否则 create_all_tools() 会在 Linux 上整个炸掉。
    tool = create_powershell_tool(tmp_path)
    assert tool.name == "powershell"
    assert tool.parameters is not None


@pytest.mark.skipif(sys.platform != "win32", reason="仅 Windows 上有 PowerShell")
async def test_powershell能执行命令(tmp_path: Path):
    tool = create_powershell_tool(tmp_path)
    text, details = await _run(tool, {"command": "Write-Output 'pong'"})
    assert "pong" in text
    assert details["exit_code"] == 0

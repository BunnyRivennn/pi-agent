"""遍历时的忽略规则。

官方 ``find``/``grep`` 依赖 ``fd``/``rg`` 二进制，它们原生尊重 ``.gitignore``。
我们用纯 Python 实现（不额外要求用户装二进制），因此忽略规则得自己来。

这里实现的是 gitignore 的**常用子集**，不是完整规范：支持目录限定（``build/``）、
根锚定（``/dist``）、取反（``!keep.txt``）、``*`` / ``?`` / ``**`` 通配。
不支持字符类等冷门语法。对「别让 agent 把 node_modules 翻一遍」这个实际目的
足够了；边界情况下宁可多遍历，也不要漏掉用户真正想找的文件。
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["IgnoreRules", "load_ignore_rules", "walk_files", "ALWAYS_IGNORED"]

#: 无条件跳过的目录：体积大、几乎从不是搜索目标。
#: 即使没有 .gitignore 也生效，避免在无 git 的目录里把 venv 翻穿。
ALWAYS_IGNORED = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "target",
    }
)


@dataclass(slots=True)
class _Pattern:
    regex_source: str
    dir_only: bool
    negated: bool
    anchored: bool


@dataclass(slots=True)
class IgnoreRules:
    """一组忽略规则，按 .gitignore 的「后者覆盖前者」语义求值。"""

    patterns: list[_Pattern] = field(default_factory=list)
    use_always_ignored: bool = True

    def is_ignored(self, relative_path: str, *, is_dir: bool) -> bool:
        """判断相对于搜索根的路径是否应被忽略。"""
        normalized = relative_path.replace(os.sep, "/").strip("/")
        if not normalized:
            return False

        if self.use_always_ignored:
            segments = normalized.split("/")
            # 路径中间的段一定是目录；最后一段得看 is_dir。
            # 否则名叫 build/dist/target 的**文件**会被误杀。
            if any(part in ALWAYS_IGNORED for part in segments[:-1]):
                return True
            if is_dir and segments[-1] in ALWAYS_IGNORED:
                return True

        ignored = False
        for pattern in self.patterns:
            if pattern.dir_only and not is_dir:
                continue
            if _matches(pattern, normalized):
                # 取反规则把之前的忽略撤回；继续扫完，保持「最后匹配者胜出」。
                ignored = not pattern.negated
        return ignored


def _matches(pattern: _Pattern, path: str) -> bool:
    if pattern.anchored:
        return fnmatch.fnmatch(path, pattern.regex_source)
    # 未锚定的规则可以匹配任意层级：对每个后缀片段都试一次。
    if fnmatch.fnmatch(path, pattern.regex_source):
        return True
    segments = path.split("/")
    return any(
        fnmatch.fnmatch("/".join(segments[i:]), pattern.regex_source)
        for i in range(1, len(segments))
    )


def _parse_line(line: str) -> _Pattern | None:
    raw = line.rstrip("\n").rstrip()
    if not raw or raw.lstrip().startswith("#"):
        return None

    negated = raw.startswith("!")
    if negated:
        raw = raw[1:]

    dir_only = raw.endswith("/")
    if dir_only:
        raw = raw[:-1]

    anchored = raw.startswith("/")
    if anchored:
        raw = raw[1:]
    elif "/" in raw:
        # gitignore 规定：含斜杠（非结尾）的规则相对于 .gitignore 所在目录锚定。
        anchored = True

    if not raw:
        return None

    # fnmatch 没有 ** 概念，但它的 * 本身就跨 /，对我们的用途足够接近。
    return _Pattern(
        regex_source=raw.replace("**/", "*").replace("/**", "/*"),
        dir_only=dir_only,
        negated=negated,
        anchored=anchored,
    )


def load_ignore_rules(root: Path, *, use_gitignore: bool = True) -> IgnoreRules:
    """读取 ``root/.gitignore``（若存在）构建忽略规则。

    只读搜索根目录下的那一个 .gitignore——嵌套 .gitignore 的完整语义
    留到确有需要时再补。
    """
    rules = IgnoreRules()
    if not use_gitignore:
        return rules

    gitignore = root / ".gitignore"
    try:
        content = gitignore.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return rules

    for line in content.splitlines():
        parsed = _parse_line(line)
        if parsed is not None:
            rules.patterns.append(parsed)
    return rules


def walk_files(
    root: Path,
    rules: IgnoreRules,
    *,
    follow_symlinks: bool = False,
) -> Iterator[tuple[Path, str]]:
    """自顶向下遍历 ``root``，跳过被忽略的目录（剪枝而非过滤）。

    Yields:
        ``(绝对路径, 相对 root 的 posix 路径)``。
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        current = Path(dirpath)
        try:
            rel_dir = current.relative_to(root).as_posix()
        except ValueError:
            continue
        rel_dir = "" if rel_dir == "." else rel_dir

        # 原地修改 dirnames 让 os.walk 不再下探——这是剪枝的关键，
        # 否则 node_modules 里的几万个文件仍会被逐个 stat。
        kept: list[str] = []
        for name in dirnames:
            child_rel = f"{rel_dir}/{name}" if rel_dir else name
            if not rules.is_ignored(child_rel, is_dir=True):
                kept.append(name)
        dirnames[:] = sorted(kept, key=str.lower)

        for name in sorted(filenames, key=str.lower):
            child_rel = f"{rel_dir}/{name}" if rel_dir else name
            if rules.is_ignored(child_rel, is_dir=False):
                continue
            yield current / name, child_rel

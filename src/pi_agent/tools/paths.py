"""工具的路径解析与安全边界。

官方 TS 的 ``path-utils.ts`` 只做 macOS 文件名兼容 fallback，**不限制工具越出
cwd**。我们这里有意加了一道护栏 :func:`resolve_in_root`：可选地把读写限制在
项目根目录内，防止模型被诱导去读 ``~/.ssh/id_rsa`` 这类文件。

护栏默认**开启**但可显式关掉（``allow_outside_root=True``），因为有些正当场景
需要读项目外的文件（比如看系统日志）。这是取舍点，不是绝对安全边界——``bash``
工具天然能绕过它，真要隔离得靠容器。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["expand_path", "resolve_to_cwd", "resolve_in_root", "PathOutsideRootError"]


class PathOutsideRootError(ValueError):
    """请求的路径落在允许的根目录之外。"""


def expand_path(raw: str) -> Path:
    """展开 ``~`` 与环境变量，规范化分隔符。

    不解析符号链接、不要求文件存在——只做纯文本层面的整理。
    """
    text = raw.strip()
    # 模型有时会把路径包在引号里发过来。
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    # 有的前端会用 @path 前缀引用文件。
    if text.startswith("@"):
        text = text[1:]
    return Path(os.path.expandvars(os.path.expanduser(text)))


def resolve_to_cwd(raw: str, cwd: Path) -> Path:
    """把可能是相对路径的输入解析成绝对路径。

    相对路径以 ``cwd`` 为基准，而不是进程的当前目录——工具的 cwd 是显式传入的。
    """
    path = expand_path(raw)
    if not path.is_absolute():
        path = cwd / path
    # 用 os.path.normpath 而不是 Path.resolve()：后者会跟随符号链接，
    # 在“路径还不存在”的写入场景下行为也更难预测。
    return Path(os.path.normpath(path))


def resolve_in_root(
    raw: str,
    cwd: Path,
    root: Path | None,
    *,
    allow_outside_root: bool = False,
) -> Path:
    """解析路径并（可选地）校验它落在 ``root`` 内。

    Args:
        raw: 模型给的原始路径字符串。
        cwd: 相对路径的基准目录。
        root: 允许访问的根目录；``None`` 表示不设限。
        allow_outside_root: 显式放行根目录之外的路径。

    Raises:
        PathOutsideRootError: 路径越界且未放行。
    """
    resolved = resolve_to_cwd(raw, cwd)
    if root is None or allow_outside_root:
        return resolved

    root_norm = Path(os.path.normpath(root))
    try:
        # Python 3.9+ 的 is_relative_to 不碰文件系统，符合“路径可能不存在”的前提。
        if resolved.is_relative_to(root_norm):
            return resolved
    except ValueError:
        # Windows 上跨盘符比较会抛 ValueError，等同于“不在根目录内”。
        pass

    raise PathOutsideRootError(
        f"Path escapes the allowed root: {resolved}\n"
        f"Allowed root: {root_norm}\n"
        "Pass allow_outside_root=True when constructing the tool to permit this."
    )

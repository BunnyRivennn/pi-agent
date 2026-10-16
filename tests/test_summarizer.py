"""摘要器测试：调用形状 + 空摘要显式失败。"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from pi_agent.agent_core import AssistantMessage, Model, TextContent
from pi_agent.pi_ai import create_default_registry
from pi_agent.summarizer import LlmSummarizer, SummarizationError

MODEL = Model(id="mock-model", provider="mock-provider", api="mock-api")
TRANSCRIPT = "用户：你好\n助手：在"


def _stub(text: str) -> list[dict[str, Any]]:
    """替掉 LLM 调用，记下传进去的参数。"""
    calls: list[dict[str, Any]] = []

    async def fake_complete(prompt: str, **kwargs: Any) -> AssistantMessage:
        calls.append({"prompt": prompt, **kwargs})
        return AssistantMessage(
            content=[TextContent(text=text)] if text else [],
            api="mock-api",
            provider="mock-provider",
            model="mock-model",
        )

    # summarizer 模块级 import 了 complete_simple，所以打在模块属性上。
    import pi_agent.summarizer as mod

    mod.complete_simple = fake_complete  # type: ignore[assignment]
    return calls


@pytest.fixture(autouse=True)
def _restore_complete_simple() -> Any:
    import pi_agent.summarizer as mod

    original = mod.complete_simple
    yield
    mod.complete_simple = original  # type: ignore[assignment]


def _summarizer(**kwargs: Any) -> LlmSummarizer:
    return LlmSummarizer(model=MODEL, registry=create_default_registry(), **kwargs)


@pytest.mark.asyncio
async def test_return_summary_text() -> None:
    s = _summarizer()
    _stub("这是摘要")

    assert await s(TRANSCRIPT) == "这是摘要"
    assert s.summaries == 1


@pytest.mark.asyncio
async def test_directly_as_summarize_callback() -> None:
    """Summarize = Callable[[str], Awaitable[str]]：实例本身就满足。"""
    s = _summarizer()
    _stub("摘要")

    callback: Any = s
    assert await callback(TRANSCRIPT) == "摘要"


@pytest.mark.asyncio
async def test_empty_summary_raises_not_empty_string() -> None:
    """回空串会被当合法结果提交，那段上下文就真没了——必须炸出来。"""
    s = _summarizer()
    _stub("")

    with pytest.raises(SummarizationError, match="空文本"):
        await s(TRANSCRIPT)
    assert s.summaries == 0


@pytest.mark.asyncio
async def test_empty_input_does_not_call_model() -> None:
    s = _summarizer()
    calls = _stub("不该被调用")

    with pytest.raises(SummarizationError, match="为空"):
        await s("   \n ")
    assert calls == []


@pytest.mark.asyncio
async def test_custom_instructions_in_prompt() -> None:
    s = _summarizer(custom_instructions="重点保留文件路径")
    calls = _stub("摘要")

    await s(TRANSCRIPT)

    assert "重点保留文件路径" in calls[0]["prompt"]


@pytest.mark.asyncio
async def test_call_time_instructions_override_constructor() -> None:
    s = _summarizer(custom_instructions="默认指令")
    calls = _stub("摘要")

    await s(TRANSCRIPT, custom_instructions="本轮指令")

    assert "本轮指令" in calls[0]["prompt"]
    assert "默认指令" not in calls[0]["prompt"]


@pytest.mark.asyncio
async def test_no_tools_and_uses_summary_system_prompt() -> None:
    s = _summarizer()
    calls = _stub("摘要")

    await s(TRANSCRIPT)

    assert "tools" not in calls[0]
    assert calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_abort_signal_passed_through() -> None:
    s = _summarizer()
    calls = _stub("摘要")
    abort = asyncio.Event()

    await s(TRANSCRIPT, abort_event=abort)

    assert calls[0]["abort_event"] is abort


@pytest.mark.asyncio
async def test_no_abort_parameter_not_passed() -> None:
    s = _summarizer()
    calls = _stub("摘要")

    await s(TRANSCRIPT)

    assert "abort_event" not in calls[0]


@pytest.mark.asyncio
async def test_progress_report_error_does_not_break_compression() -> None:
    def bad(_stage: str) -> None:
        raise RuntimeError("打印炸了")

    s = _summarizer(on_progress=bad)
    _stub("摘要")

    assert await s(TRANSCRIPT) == "摘要"


@pytest.mark.asyncio
async def test_reports_two_stages_compressing_and_done() -> None:
    stages: list[str] = []
    s = _summarizer(on_progress=stages.append)
    _stub("摘要")

    await s(TRANSCRIPT)

    assert stages == ["压缩中", "压缩完成"]


# ---------------------------------------------------------------------------
# 真调 LLM：默认不跑（``addopts = -m 'not live'``），要跑用 ``pytest -m live``。
#
# 上面那些 stub 测试验的是调用形状——参数拼对了、空摘要会报错。它们验不了的是
# 摘要本身能不能用：真模型会不会把一段带公式的推导压成一句"讨论了数学问题"。
# 这个只能真调一次看。
# ---------------------------------------------------------------------------

# 一段故意有结构的材料：题目 + 四步推导 + 结论。好摘要应该保住"高斯曲率
# -1/a²"和"伪球面/常负曲率"这类关键结论，而不是只说"算了一个曲率"。
RIEMANN_TRANSCRIPT = """\
用户：帮我算一下伪球面（tractricoid）的高斯曲率。参数化用
r(u, v) = (a·sech(u)·cos v, a·sech(u)·sin v, a·(u − tanh(u)))，a > 0。

助手：分四步走。

第一步，第一基本形式。对 u 求偏导得 r_u = a(−sech u·tanh u·cos v,
−sech u·tanh u·sin v, tanh²u)，对 v 求偏导得 r_v = a·sech u(−sin v, cos v, 0)。
于是 E = r_u·r_u = a²·tanh²u，F = r_u·r_v = 0，G = r_v·r_v = a²·sech²u。
F = 0 说明这个参数化是正交的，后面公式能简化。

第二步，单位法向量。r_u × r_v 算出来方向是 (cos v, sin v, sinh u) 乘上
a²·sech²u·tanh u，模长给出单位法向 n = (cos v, sin v, sinh u)/cosh u。

第三步，第二基本形式。L = r_uu·n = −a·tanh u·sech u，M = r_uv·n = 0，
N = r_vv·n = a·sech u·tanh u。注意 L 和 N 反号——这是负曲率的来源。

第四步，高斯曲率。正交参数化下 K = (LN − M²)/(EG − F²)。
分子 LN − M² = −a²·sech²u·tanh²u，分母 EG − F² = a⁴·sech²u·tanh²u。
相除得 K = −1/a²。

结论：伪球面是常负曲率曲面，K ≡ −1/a²，与 u、v 都无关。这正是它能作为双曲
几何局部模型的原因——把 a 取 1 就得到曲率恒为 −1 的曲面。顺带一提，希尔伯特
定理说明没有完备的常负曲率曲面能等距嵌入 R³，所以伪球面只能是局部模型。
"""


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_真模型能压出保留关键结论的摘要() -> None:
    """真跑一次：确认摘要不是空壳，且关键结论没被压掉。

    不断言具体措辞——模型每次用词不同，那样的断言只会变成随机失败。断言的是
    「这段摘要还能不能用」：比原文短、提到了伪球面和那个曲率值。
    """
    pytest.importorskip("dotenv")
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("未配置 OPENAI_API_KEY")

    model = Model(
        id=os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini"),
        provider="openai-completions"
        if os.getenv("OPENAI_API_STYLE", "responses").lower() == "completions"
        else "openai",
        api="openai-completions"
        if os.getenv("OPENAI_API_STYLE", "responses").lower() == "completions"
        else "openai",
        base_url=os.getenv("OPENAI_API_BASE_URL", ""),
    )
    summarizer = LlmSummarizer(model=model, registry=create_default_registry())

    summary = await summarizer(RIEMANN_TRANSCRIPT)

    print(f"\n--- 真实摘要（{len(summary)} 字符）---\n{summary}\n---")

    assert summary.strip()
    assert len(summary) < len(RIEMANN_TRANSCRIPT)  # 没压缩就不叫摘要
    assert summarizer.summaries == 1
    # 伪球面和 −1/a² 是这段的全部要点，丢了等于这段上下文白存
    assert "伪球面" in summary or "pseudosphere" in summary.lower()
    assert "1/a" in summary or "1/a²" in summary or "负曲率" in summary

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any, TypeVar, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from .event_stream import EventStream
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    GetMessagesFn,
    LlmContext,
    StreamFn,
    TextContent,
    ToolCall,
    ToolResultMessage,
    assistant_tool_calls,
    message_role,
)


def agent_loop(
        prompts: list[AgentMessage],
        context: AgentContext,
        config: AgentLoopConfig,
        abort_event: asyncio.Event | None = None,
        stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = list(prompts)
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=[*context.messages, *prompts],
            tools=context.tools,
        )

        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})

        for prompt in prompts:
            stream.push({"type": "message_start", "message": prompt})
            stream.push({"type": "message_end", "message": prompt})

        await _run_loop(current_context, new_messages, config, abort_event, stream, stream_fn)

    asyncio.create_task(_run())
    return stream


def agent_loop_continue(
        context: AgentContext,
        config: AgentLoopConfig,
        abort_event: asyncio.Event | None = None,
        stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    if message_role(context.messages[-1]) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )

        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})

        await _run_loop(current_context, new_messages, config, abort_event, stream, stream_fn)

    asyncio.create_task(_run())
    return stream


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    return EventStream[AgentEvent, list[AgentMessage]](
        is_complete=lambda event: event["type"] == "agent_end",
        extract_result=lambda event: event["messages"] if event["type"] == "agent_end" else [],
    )


async def _run_loop(
        current_context: AgentContext,
        new_messages: list[AgentMessage],
        config: AgentLoopConfig,
        abort_event: asyncio.Event | None,
        stream: EventStream[AgentEvent, list[AgentMessage]],
        stream_fn: StreamFn | None,
) -> None:
    """事件流里的事件 生产者：作用：决定"要不要再调一次 LLM" """
    """
    current_context：当前的对话上下文（系统提示、消息列表、工具列表）

    new_messages：本轮新增的消息（最终会作为结果返回）
    
    config：配置（模型、API Key、转换函数等）
    
    abort_event：中止信号
    
    stream：事件流（往里面 push 事件）
    
    stream_fn：调用 LLM 的函数
    
    两层 while：
    外层：处理"follow-up"（追问）
    内层：处理"tool call"（工具调用链
    
    follow_up 是一个"延迟投递的收件箱"，决定消息在哪个 turn 进入上下文；
    消息一旦被消费、append 进 current_context.messages，它才融入那份裸的短期记忆。 
    它和 steering 共同构成了 Agent 的并发输入调度
    """

    first_turn = True
    # 第一步：获取 steering 消息：是人为插入的引导消息。
    # 每次 LLM 调用前，都会先塞这条消息进去。
    pending_messages = await _maybe_get_messages(config.get_steering_messages)

    while True:
        """
        外层循环只做一件事：检查有没有 follow-up 消息。
        有 → 继续循环
        没有 → break
        """
        has_more_tool_calls = True
        steering_after_tools: list[AgentMessage] | None = None

        # 一个 turn：模型说话 + 工具链
        while has_more_tool_calls or pending_messages:
            # steering 在这个内层循环的开头/工具后被消化（插队进当前 turn）

            # 1. 如果不是第一轮，push turn_start
            if not first_turn:
                stream.push({"type": "turn_start"})
            else:
                first_turn = False

            # 2. 如果有 pending 消息，先处理它们
            if pending_messages:
                for message in pending_messages:
                    stream.push({"type": "message_start", "message": message})
                    stream.push({"type": "message_end", "message": message})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # 3. 调 LLM → 得到 assistant_message
            message = await _stream_assistant_response(
                current_context,
                config,
                abort_event,
                stream,
                stream_fn,
            )
            new_messages.append(message)

            # 4. 检查 LLM 的返回状态
            if message.stop_reason in {"error", "aborted"}:
                stream.push({"type": "turn_end", "message": message, "tool_results": []})
                stream.push({"type": "agent_end", "messages": new_messages})
                stream.end(new_messages)
                return
            # 5. 检查是否有 tool call
            tool_calls = assistant_tool_calls(message)
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_execution = await _execute_tool_calls(
                    tools=current_context.tools,
                    assistant_message=message,
                    abort_event=abort_event,
                    stream=stream,
                    get_steering_messages=config.get_steering_messages,
                )
                tool_results.extend(tool_execution["tool_results"])
                steering_after_tools = tool_execution.get("steering_messages")

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            stream.push({"type": "turn_end", "message": message, "tool_results": tool_results})

            if steering_after_tools:
                pending_messages = steering_after_tools
                steering_after_tools = None
            else:
                pending_messages = await _maybe_get_messages(config.get_steering_messages)

        follow_up_messages = await _maybe_get_messages(config.get_follow_up_messages)
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue  # ← 模型以为下班了，发现队列里还有活，开新 turn

        break         # ← 真没活了，agent_end

    stream.push({"type": "agent_end", "messages": new_messages})
    stream.end(new_messages)


async def _stream_assistant_response(
        context: AgentContext,
        config: AgentLoopConfig,
        abort_event: asyncio.Event | None,
        stream: EventStream[AgentEvent, list[AgentMessage]],
        stream_fn: StreamFn | None,
) -> AssistantMessage: # LLM 的完整回复
    """调 LLM → 消费事件流 → 把 LLM 的回复放入上下文 → 返回最终的 AssistantMessage"""

    # 1. 准备消息（可能经过 transform_context 转换）
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(messages, abort_event)

    # 2. 把消息转换成 LLM 能理解的格式
    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    # 3. 构造 LLM 上下文
    llm_context = LlmContext(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    # 4. 获取 stream_fn（调 LLM 的函数）
    stream_function = stream_fn
    if stream_function is None:
        raise RuntimeError("stream_fn is required")

    if config.get_api_key:
        resolved_key = await _maybe_await(config.get_api_key(config.model.provider))
        if resolved_key is not None:
            config = replace(config, api_key=resolved_key)
    # 6. 调 LLM → 得到事件流
    response = await _maybe_await(stream_function(config.model, llm_context, config, abort_event))

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        event_type = event["type"]

        if event_type == "start":
            partial = event.get("partial")
            if isinstance(partial, AssistantMessage):
                partial_message = partial
                context.messages.append(partial_message)
                added_partial = True
                stream.push({"type": "message_start", "message": partial_message})
            continue

        if event_type in {
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        }:
            if partial_message is not None:
                partial = event.get("partial")
                if isinstance(partial, AssistantMessage):
                    partial_message = partial
                    context.messages[-1] = partial_message
                    stream.push(
                        {
                            "type": "message_update",
                            "message": partial_message,
                            "assistant_message_event": event,
                        }
                    )
            continue

        if event_type in {"done", "error"}:
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                stream.push({"type": "message_start", "message": final_message})

            stream.push({"type": "message_end", "message": final_message})
            return final_message

    return await response.result()


async def _execute_tool_calls(
        tools: list[AgentTool] | None,
        assistant_message: AssistantMessage,
        abort_event: asyncio.Event | None,
        stream: EventStream[AgentEvent, list[AgentMessage]],
        get_steering_messages: GetMessagesFn | None,
) -> dict[str, Any]:
    """遍历 tool call → 找工具 → 执行 → 推事件 → 处理插话"""
    tool_calls = assistant_tool_calls(assistant_message)
    results: list[ToolResultMessage] = []
    steering_messages: list[AgentMessage] | None = None

    for index, tool_call in enumerate(tool_calls):
        tool = next(
            (candidate for candidate in tools or [] if candidate.name == tool_call.name),
            None,
        )
        tool_call_id = tool_call.id
        tool_name = tool_call.name
        tool_args = tool_call.arguments

        stream.push(
            {
                "type": "tool_execution_start",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args": tool_args,
            }
        )

        result: AgentToolResult[Any]
        is_error = False

        try:
            if tool is None:
                raise RuntimeError(f"Tool {tool_name} not found")
            # 验证参数
            validated_args = _validate_tool_arguments(tool=tool, tool_call=tool_call)

            def on_update(
                    partial: AgentToolResult[Any],
                    *,
                    call_id: str = tool_call_id,
                    name: str = tool_name,
                    args: dict[str, Any] = tool_args,
            ) -> None:
                stream.push(
                    {
                        "type": "tool_execution_update",
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "args": args,
                        "partial_result": partial,
                    }
                )

            # 真的执行工具
            result = await tool.execute(
                tool_call_id,
                validated_args,
                abort_event,
                on_update,
            )
        except Exception as exc:  # noqa: BLE001
            result = AgentToolResult(content=[TextContent(text=str(exc))], details={})
            is_error = True

        stream.push(
            {
                "type": "tool_execution_end",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
                "is_error": is_error,
            }
        )

        tool_result_message = ToolResultMessage(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=result.content,
            details=result.details,
            is_error=is_error,
        )

        results.append(tool_result_message)
        stream.push({"type": "message_start", "message": tool_result_message})
        stream.push({"type": "message_end", "message": tool_result_message})

        if get_steering_messages is not None:
            steering = await _maybe_get_messages(get_steering_messages)
            if steering:
                steering_messages = steering
                for skipped_call in tool_calls[index + 1:]:
                    results.append(_skip_tool_call(skipped_call, stream))
                break

    return {"tool_results": results, "steering_messages": steering_messages}


def _skip_tool_call(
        tool_call: ToolCall,
        stream: EventStream[AgentEvent, list[AgentMessage]],
) -> ToolResultMessage:
    result: AgentToolResult[dict[str, Any]] = AgentToolResult(
        content=[TextContent(text="Skipped due to queued user message.")],
        details={},
    )

    stream.push(
        {
            "type": "tool_execution_start",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "args": tool_call.arguments,
        }
    )
    stream.push(
        {
            "type": "tool_execution_end",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "result": result,
            "is_error": True,
        }
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=True,
    )

    stream.push({"type": "message_start", "message": tool_result_message})
    stream.push({"type": "message_end", "message": tool_result_message})
    return tool_result_message


T = TypeVar("T")


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


async def _maybe_get_messages(
        getter: Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None,
) -> list[AgentMessage]:
    if getter is None:
        return []

    messages = await _maybe_await(getter())
    return list(messages or [])


def _validate_tool_arguments(
        *,
        tool: AgentTool,
        tool_call: ToolCall,
) -> dict[str, Any]:
    arguments: Any = tool_call.arguments
    if not isinstance(arguments, Mapping):
        raise RuntimeError(
            f'Tool "{tool_call.name}" produced non-object arguments. '
            f"Expected JSON object, got {type(arguments).__name__}."
        )

    schema = tool.parameters
    if schema is None:
        return {str(key): value for key, value in arguments.items()}

    if not isinstance(schema, Mapping):
        raise RuntimeError(
            f'Tool "{tool.name}" has an invalid parameter schema. '
            f"Expected mapping, got {type(schema).__name__}."
        )

    validator = Draft202012Validator(dict(schema))
    errors = sorted(
        validator.iter_errors(arguments),
        key=lambda error: _validation_error_path(error),
    )
    if errors:
        raise RuntimeError(
            _format_tool_validation_error(tool_name=tool_call.name, args=arguments, errors=errors)
        )

    return {str(key): value for key, value in arguments.items()}


def _validation_error_path(error: ValidationError) -> str:
    if error.path:
        return ".".join(str(part) for part in error.path)
    return "root"


def _format_tool_validation_error(
        *,
        tool_name: str,
        args: Mapping[str, Any],
        errors: list[ValidationError],
) -> str:
    lines = [f'Validation failed for tool "{tool_name}":']
    for error in errors:
        lines.append(f"  - {_validation_error_path(error)}: {error.message}")

    try:
        rendered_args = json.dumps(args, indent=2, sort_keys=True)
    except TypeError:
        rendered_args = str(args)

    lines.append("")
    lines.append("Received arguments:")
    lines.append(rendered_args)
    return "\n".join(lines)

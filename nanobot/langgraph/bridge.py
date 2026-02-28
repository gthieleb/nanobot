"""Bridge between nanobot's message bus and LangGraph state.

This module provides translation utilities to convert between:
- InboundMessage → LangGraph state (for agent invocation)
- LangGraph result → OutboundMessage (for channel delivery)

Also handles streaming updates and progress reporting.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

if TYPE_CHECKING:
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from nanobot.langgraph.checkpointer import SessionCheckpointer


def translate_inbound_to_state(
    msg: "InboundMessage",
    history: list[dict] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Convert InboundMessage to LangGraph state.

    Args:
        msg: The inbound message from nanobot's bus
        history: Optional conversation history
        system_prompt: Optional system prompt to prepend

    Returns:
        LangGraph state dict with 'messages' key
    """
    messages = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    if history:
        for entry in history:
            role = entry.get("role")
            content = entry.get("content", "")

            if role == "user":
                messages.append(HumanMessage(content=str(content)))
            elif role == "assistant":
                tool_calls = entry.get("tool_calls")
                if tool_calls:
                    messages.append(AIMessage(content=str(content) or "", tool_calls=tool_calls))
                else:
                    messages.append(AIMessage(content=str(content)))
            elif role == "tool":
                messages.append(
                    ToolMessage(
                        content=str(content),
                        tool_call_id=str(entry.get("tool_call_id", "")),
                    )
                )
            elif role == "system":
                messages.append(SystemMessage(content=str(content)))

    if msg.media:
        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": msg.content}]
        for media in msg.media:
            if media.get("type") == "image":
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": media.get("url", "")},
                    }
                )
        messages.append(HumanMessage(content=content_blocks))
    else:
        messages.append(HumanMessage(content=msg.content))

    return {"messages": messages}


def translate_result_to_outbound(
    result: dict[str, Any],
    msg: "InboundMessage",
    metadata: dict[str, Any] | None = None,
) -> "OutboundMessage":
    """Convert LangGraph result to OutboundMessage.

    Args:
        result: The result from LangGraph agent invocation
        msg: The original inbound message (for routing info)
        metadata: Optional metadata to include

    Returns:
        OutboundMessage ready for delivery
    """
    from nanobot.bus.events import OutboundMessage

    messages = result.get("messages", [])
    final_content = ""

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            final_content = str(message.content) or ""
            break

    if not final_content and messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            final_content = str(last_msg.content) or "Task completed."

    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=final_content,
        metadata=metadata or msg.metadata or {},
    )


class LangGraphBridge:
    """Bridge between nanobot's message bus and LangGraph agent.

    This class:
    1. Manages agent invocation with proper checkpointing
    2. Handles streaming/progress updates
    3. Translates messages between formats

    Args:
        agent: The compiled LangGraph agent
        checkpointer: Session checkpointer for persistence
        workspace: Workspace directory for file operations

    Example:
        >>> from deepagents import create_deep_agent
        >>> agent = create_deep_agent()
        >>> bridge = LangGraphBridge(agent, checkpointer)
        >>> response = await bridge.process(inbound_msg)
    """

    def __init__(
        self,
        agent: CompiledStateGraph,
        checkpointer: "SessionCheckpointer",
        workspace: Path | None = None,
    ):
        self.agent = agent
        self.checkpointer = checkpointer
        self.workspace = workspace or Path.home() / ".nanobot" / "workspace"
        self._running = False

    def _build_config(self, session_key: str) -> RunnableConfig:
        """Build LangGraph config for session."""
        return {
            "configurable": {
                "thread_id": session_key,
            },
        }

    async def process(
        self,
        msg: "InboundMessage",
        on_progress: Callable[[str, bool], Awaitable[None]] | None = None,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> "OutboundMessage":
        """Process an inbound message through the LangGraph agent.

        Args:
            msg: The inbound message
            on_progress: Optional callback for progress updates (content, is_tool_hint)
            system_prompt: Optional system prompt override
            history: Optional conversation history

        Returns:
            OutboundMessage with the response
        """
        from nanobot.bus.events import OutboundMessage

        state = translate_inbound_to_state(msg, history, system_prompt)
        config = self._build_config(msg.session_key)

        try:
            if on_progress:
                result = await self._invoke_with_progress(state, config, on_progress)
            else:
                result = await self.agent.ainvoke(state, config)

            return translate_result_to_outbound(result, msg)

        except Exception as e:
            logger.exception("Error processing message in LangGraph agent")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Sorry, I encountered an error: {str(e)}",
                metadata=msg.metadata or {},
            )

    async def _invoke_with_progress(
        self,
        state: dict[str, Any],
        config: RunnableConfig,
        on_progress: Callable[[str, bool], Awaitable[None]],
    ) -> dict[str, Any]:
        """Invoke agent with streaming and progress updates."""
        final_result: dict[str, Any] = {}

        async for event in self.agent.astream_events(state, config, version="v2"):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                content = event.get("data", {}).get("chunk")
                if content and hasattr(content, "content") and content.content:
                    await on_progress(str(content.content), False)

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                hint = self._format_tool_hint(tool_name, tool_input)
                if hint:
                    await on_progress(hint, True)

            elif kind == "on_chain_end":
                if event.get("name") == "LangGraph":
                    final_result = event.get("data", {}).get("output", {})

        return final_result or {"messages": []}

    @staticmethod
    def _format_tool_hint(name: str, args: dict) -> str:
        """Format tool call as concise hint."""
        if not args:
            return f"{name}()"

        first_val = next(iter(args.values()), None)
        if isinstance(first_val, str):
            if len(first_val) > 40:
                return f'{name}("{first_val[:40]}…")'
            return f'{name}("{first_val}")'
        return f"{name}()"

    async def run(self) -> None:
        """Run the bridge as a background task (for gateway mode)."""
        self._running = True
        logger.info("LangGraph bridge started")

    def stop(self) -> None:
        """Stop the bridge."""
        self._running = False
        logger.info("LangGraph bridge stopping")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage).

        Args:
            content: The message content
            session_key: Session identifier
            channel: Channel name
            chat_id: Chat identifier
            on_progress: Optional progress callback

        Returns:
            Response content
        """
        from nanobot.bus.events import InboundMessage

        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
        )

        async def _progress(text: str, is_tool_hint: bool = False) -> None:
            if on_progress:
                await on_progress(text)

        response = await self.process(msg, on_progress=_progress)
        return response.content

    def clear_session(self, session_key: str) -> bool:
        """Clear a session's checkpoint history.

        Args:
            session_key: The session to clear

        Returns:
            True if session was deleted
        """
        return self.checkpointer.delete_session(session_key)

    def get_history(self, session_key: str, limit: int = 100) -> list[dict]:
        """Get message history for a session.

        Args:
            session_key: The session identifier
            limit: Maximum messages to return

        Returns:
            List of message dicts
        """
        return self.checkpointer.get_session_history(session_key, limit)

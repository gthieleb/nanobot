"""Deep Agent wrapper for nanobot.

This module provides the main integration between nanobot and deepagents,
creating a LangGraph-based agent with nanobot-specific tools and features.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from loguru import logger

if TYPE_CHECKING:
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from nanobot.config.schema import Config

try:
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend, StateBackend
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware import (
        FilesystemMiddleware,
        SkillsMiddleware,
        SubAgentMiddleware,
        SummarizationMiddleware,
    )

    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False
    create_deep_agent = None


class DeepAgent:
    """LangGraph-based agent using deepagents framework.

    This class wraps deepagents' create_deep_agent() to provide:
    - File operations via FilesystemBackend
    - Subagent spawning via task tool
    - Session checkpointing
    - Multi-channel message routing

    Args:
        workspace: Workspace directory for file operations
        config: nanobot configuration
        checkpointer: Session checkpointer instance

    Example:
        >>> from nanobot.langgraph import SessionCheckpointer
        >>> checkpointer = SessionCheckpointer()
        >>> agent = DeepAgent(workspace, config, checkpointer)
        >>> response = await agent.process(msg)
    """

    def __init__(
        self,
        workspace: Path,
        config: "Config",
        checkpointer: Any | None = None,
    ):
        if not DEEPAGENTS_AVAILABLE:
            raise ImportError("deepagents is not installed. Install with: pip install deepagents")

        self.workspace = workspace
        self.config = config
        self.checkpointer = checkpointer

        self._agent = None
        self._backend: BackendProtocol | None = None
        self._tools: list[Any] = []
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_servers = config.tools.mcp_servers if hasattr(config, "tools") else {}

    def _get_model_spec(self) -> str:
        """Get model specification from config."""
        model = self.config.agents.defaults.model if hasattr(self.config, "agents") else None
        if not model:
            model = "anthropic:claude-sonnet-4-5"

        provider = self.config.agents.defaults.provider if hasattr(self.config, "agents") else None
        if provider and ":" not in model:
            model = f"{provider}:{model}"

        return model

    def _get_backend(self) -> BackendProtocol:
        """Get or create the backend for file operations."""
        if self._backend is None:
            self._backend = FilesystemBackend(root_dir=self.workspace)
        return self._backend

    def _build_custom_tools(self) -> list[Any]:
        """Build nanobot-specific custom tools."""
        tools = []

        return tools

    def _create_agent(self) -> Any:
        """Create the deep agent with configured middleware."""
        model = self._get_model_spec()
        backend = self._get_backend()
        custom_tools = self._build_custom_tools()

        system_prompt = self._build_system_prompt()

        agent = create_deep_agent(
            model=model,
            tools=custom_tools,
            system_prompt=system_prompt,
            backend=backend,
            checkpointer=self.checkpointer,
        )

        return agent

    def _build_system_prompt(self) -> str:
        """Build system prompt with nanobot context."""
        from datetime import datetime
        import time

        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"

        return f"""# nanobot Agent

## Current Time
{now} ({tz})

## Workspace
Your workspace is at: {self.workspace}

## Skills
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

## Behavior
- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- When doing tasks: understand first, act, then verify.
- Keep working until the task is fully complete.
- Use the task tool to delegate complex, multi-step tasks to subagents.
"""

    @property
    def agent(self) -> Any:
        """Get or create the compiled agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers."""
        if self._mcp_connected or not self._mcp_servers:
            return

        try:
            from langchain_mcp_adapters import load_mcp_tools

            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()

            for name, server_config in self._mcp_servers.items():
                try:
                    tools = await load_mcp_tools(server_config, self._mcp_stack)
                    self._tools.extend(tools)
                    logger.info(f"Loaded {len(tools)} tools from MCP server: {name}")
                except Exception as e:
                    logger.error(f"Failed to load MCP tools from {name}: {e}")

            self._mcp_connected = True
            if self._tools:
                self._agent = None

        except ImportError:
            logger.warning("langchain-mcp-adapters not installed, skipping MCP")
        except Exception as e:
            logger.error(f"Failed to connect MCP servers: {e}")

    async def process(
        self,
        msg: "InboundMessage",
        on_progress: Callable[[str, bool], Awaitable[None]] | None = None,
        history: list[dict] | None = None,
    ) -> "OutboundMessage":
        """Process an inbound message.

        Args:
            msg: The inbound message
            on_progress: Optional callback for progress updates
            history: Optional conversation history

        Returns:
            OutboundMessage with the response
        """
        from nanobot.bus.events import OutboundMessage
        from nanobot.langgraph.bridge import (
            translate_inbound_to_state,
            translate_result_to_outbound,
        )

        await self._connect_mcp()

        state = translate_inbound_to_state(msg, history)
        config = {
            "configurable": {
                "thread_id": msg.session_key,
            },
        }

        try:
            if on_progress:
                result = await self._invoke_with_progress(state, config, on_progress)
            else:
                result = await self.agent.ainvoke(state, config)

            return translate_result_to_outbound(result, msg)

        except Exception as e:
            logger.exception("Error in deep agent")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Sorry, I encountered an error: {str(e)}",
                metadata=msg.metadata or {},
            )

    async def _invoke_with_progress(
        self,
        state: dict[str, Any],
        config: dict[str, Any],
        on_progress: Callable[[str, bool], Awaitable[None]],
    ) -> dict[str, Any]:
        """Invoke agent with streaming."""
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

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI usage).

        Args:
            content: Message content
            session_key: Session identifier
            channel: Channel name
            chat_id: Chat identifier
            on_progress: Progress callback

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
        """Clear a session's history."""
        if self.checkpointer:
            return self.checkpointer.delete_session(session_key)
        return False

    def get_history(self, session_key: str, limit: int = 100) -> list[dict]:
        """Get message history for a session."""
        if self.checkpointer:
            return self.checkpointer.get_session_history(session_key, limit)
        return []

    async def close(self) -> None:
        """Close connections and cleanup."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except Exception:
                pass
            self._mcp_stack = None
        self._mcp_connected = False


def is_deepagents_available() -> bool:
    """Check if deepagents is available."""
    return DEEPAGENTS_AVAILABLE

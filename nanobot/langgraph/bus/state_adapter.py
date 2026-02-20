"""Message Bus Adapter for LangGraph integration."""

from typing import Any

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.langgraph.graph.state import AgentState
from loguru import logger


class StateMessageBusAdapter:
    """
    Verbindet LangGraph-State mit existierendem Nanobot Message Bus.
    Ermöglicht parallele Coexistenz.
    """

    def __init__(self, graph_app, nanobot_bus, config):
        self.graph_app = graph_app
        self.nanobot_bus = nanobot_bus
        self.config = config
        self._active_threads: dict[str, Any] = {}

    async def consume_from_bus(self) -> InboundMessage:
        """
        Wartet auf Messages vom Nanobot Bus.
        """
        return await self.nanobot_bus.consume_inbound()

    async def process_message(self, msg: InboundMessage):
        """
        Verarbeitet Message über LangGraph.
        """
        # Thread ID für Session (wie Nanobot Sessions)
        thread_id = f"{msg.channel}:{msg.chat_id}"

        # Initial State
        initial_state = {
            "messages": [],
            "current_tools": [],
            "subagent_tasks": [],
            "current_context": {},
        }

        # User Message hinzufügen
        initial_state["messages"].append({"role": "user", "content": msg.content})

        # Config für den Graph-Aufruf
        graph_config = {
            "configurable": {
                "thread_id": thread_id,
                "provider": self._get_provider(),
                "tool_registry": self._get_tool_registry(),
                "workspace": self.config.workspace_path,
                "subagent_manager": self._get_subagent_manager(),
                "model": self.config.agents.defaults.model,
                "temperature": self.config.agents.defaults.temperature,
                "max_tokens": self.config.agents.defaults.max_tokens,
            }
        }

        try:
            # Graph ausführen
            result = await self.graph_app.ainvoke(initial_state, config=graph_config)

            # Antwort an Bus senden
            if result and result.get("messages"):
                last_message = result["messages"][-1]
                content = (
                    last_message.get("content")
                    if isinstance(last_message, dict)
                    else last_message.content
                )

                await self.nanobot_bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=content,
                        metadata=msg.metadata or {},
                    )
                )

        except Exception as e:
            logger.error("Error processing message: {}", e)
            # Error-Response an Bus
            await self.nanobot_bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Error: {str(e)}",
                    metadata=msg.metadata or {},
                )
            )

    def _get_provider(self):
        """Lädt Provider aus Nanobot Config."""
        from nanobot.cli.commands import _make_provider

        return _make_provider(self.config)

    def _get_tool_registry(self):
        """Lädt Tool Registry."""
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.tools.filesystem import (
            ReadFileTool,
            WriteFileTool,
            EditFileTool,
            ListDirTool,
        )
        from nanobot.agent.tools.shell import ExecTool
        from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
        from nanobot.agent.tools.message import MessageTool
        from nanobot.agent.tools.spawn import SpawnTool
        from nanobot.langgraph.tools.spawn_tool import SpawnTool as LangGraphSpawnTool

        registry = ToolRegistry()

        # Tools registrieren (wie in AgentLoop._register_default_tools)
        workspace = self.config.workspace_path

        registry.register(ReadFileTool(workspace=workspace))
        registry.register(WriteFileTool(workspace=workspace))
        registry.register(EditFileTool(workspace=workspace))
        registry.register(ListDirTool(workspace=workspace))
        registry.register(
            ExecTool(
                working_dir=str(workspace),
                timeout=self.config.tools.exec.timeout,
                restrict_to_workspace=self.config.tools.restrict_to_workspace,
            )
        )
        registry.register(WebSearchTool(api_key=self.config.tools.web.search.api_key or None))
        registry.register(WebFetchTool())

        # Spawn Tool (LangGraph-spezifisch)
        # Da SubagentManager noch nicht existiert, Placeholder
        spawn_tool = SpawnTool(None)
        registry.register(spawn_tool)

        return registry

    def _get_subagent_manager(self):
        """Lädt oder erstellt Subagent Manager."""
        from nanobot.langgraph.subagent.manager import SubagentManager

        provider = self._get_provider()
        workspace = self.config.workspace_path

        # Manager erstellen
        manager = SubagentManager(
            provider=provider, workspace=workspace, bus=self.nanobot_bus, main_state_ref={}
        )

        return manager

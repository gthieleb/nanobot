"""Subagent Tool Adapter (without spawn/message tools)."""

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


class SubagentToolAdapter:
    """
    Adapter für Subagent-Tools (ohne spawn/message Tool).
    """

    def __init__(self, workspace=None, brave_api_key=None, exec_config=None):
        self.registry = ToolRegistry()

        # Nur erlaubte Tools registrieren
        self.registry.register(ReadFileTool(workspace=workspace))
        self.registry.register(WriteFileTool(workspace=workspace))
        self.registry.register(EditFileTool(workspace=workspace))
        self.registry.register(ListDirTool(workspace=workspace))

        if exec_config:
            self.registry.register(ExecTool(working_dir=workspace, **exec_config))

        if brave_api_key:
            self.registry.register(WebSearchTool(api_key=brave_api_key))

        self.registry.register(WebFetchTool())

    def get_definitions(self) -> list[dict]:
        """Gibt Tool-Definitionen für LLM zurück."""
        return self.registry.get_definitions()

    async def execute(self, tool_name: str, args: dict) -> str:
        """Führt Tool aus."""
        return await self.registry.execute(tool_name, args)

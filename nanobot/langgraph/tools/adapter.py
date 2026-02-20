"""Nanobot Registry to LangChain tool adapter."""

from nanobot.agent.tools.registry import ToolRegistry


def get_tool_definitions(tool_registry: ToolRegistry) -> list[dict]:
    """
    Konvertiert Nanobot Tool-Registry in OpenAI-Format für LLM.
    """
    return tool_registry.get_definitions()


async def execute_nanobot_tool(tool_registry: ToolRegistry, tool_name: str, args: dict) -> str:
    """
    Führt Tool über Nanobot Registry aus.
    """
    return await tool_registry.execute(tool_name, args)

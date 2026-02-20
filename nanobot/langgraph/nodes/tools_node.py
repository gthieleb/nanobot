"""Tools node for executing tools."""

from typing import Any

from nanobot.langgraph.graph.state import AgentState
from nanobot.langgraph.tools.adapter import execute_nanobot_tool


async def execute_tools(state: AgentState, config) -> dict[str, Any]:
    """
    Führt Tools aus - Nanobot Registry über Adapter.
    """
    last_message = state["messages"][-1]

    if not last_message:
        return state

    # Prüfe, ob tool_calls vorhanden sind
    tool_calls = last_message.get("tool_calls") if isinstance(last_message, dict) else []
    if not tool_calls and hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls

    if not tool_calls:
        return state

    # Tool Registry aus config
    tool_registry = config["configurable"]["tool_registry"]

    # Alle Tool-Aufrufe ausführen
    for tool_call in tool_calls:
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call.get("args") if isinstance(tool_call, dict) else tool_call.args
        tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id

        try:
            # Über Adapter mit Nanobot Registry ausführen
            result = await execute_nanobot_tool(tool_registry, tool_name, tool_args)

            # Tool-Result als Message anhängen
            tool_message = {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_id,
                "name": tool_name,
            }
            state["messages"].append(tool_message)

        except Exception as e:
            from loguru import logger

            logger.error("Error executing tool {}: {}", tool_name, e)

            error_message = {
                "role": "tool",
                "content": f"Error executing {tool_name}: {str(e)}",
                "tool_call_id": tool_id,
                "name": tool_name,
            }
            state["messages"].append(error_message)

    return state

"""Agent node for LLM calls."""

from typing import Any

from nanobot.langgraph.graph.state import AgentState
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.langgraph.tools.adapter import get_tool_definitions


async def call_model(state: AgentState, config) -> dict[str, Any]:
    """
    LLM-Aufruf - äquivalent zu AgentLoop._run_agent_loop().
    """
    # LLM Provider aus config holen (wiederverwendet Nanobot's Provider)
    provider = config["configurable"]["provider"]

    if not state["messages"]:
        return state

    # State in Message-Liste konvertieren
    messages = []
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = msg.type
            content = msg.content if hasattr(msg, "content") else str(msg)
        elif isinstance(msg, str):
            role = "user"
            content = msg
        elif isinstance(msg, dict) and "role" in msg:
            role = msg["role"]
            content = msg.get("content", "")
        else:
            role = "user"
            content = str(msg)

        messages.append({"role": role, "content": content})

    # Tool-Definitionen über Adapter holen
    tools = get_tool_definitions(config["configurable"]["tool_registry"])

    # LLM aufrufen
    try:
        response: LLMResponse = await provider.chat(
            messages=messages,
            tools=tools,
            model=config["configurable"]["model"],
            temperature=config["configurable"]["temperature"],
            max_tokens=config["configurable"]["max_tokens"],
        )
    except Exception as e:
        from loguru import logger

        logger.error("LLM call failed: {}", e)
        # Error als Message zurückgeben
        state["messages"].append({"role": "assistant", "content": f"Error: {str(e)}"})
        state["current_tools"] = []
        return state

    # Response in Message konvertieren
    if response.tool_calls:
        # Tool-Aufrufe
        tool_calls_data = [
            {"id": tc.id, "name": tc.name, "args": tc.arguments} for tc in response.tool_calls
        ]

        ai_message = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": tool_calls_data,
        }

        state["messages"].append(ai_message)
        state["current_tools"] = [tc.name for tc in response.tool_calls]
    else:
        # Normale Antwort
        ai_message = {"role": "assistant", "content": response.content or ""}

        state["messages"].append(ai_message)
        state["current_tools"] = []

    return state

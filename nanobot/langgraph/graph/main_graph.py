"""Main graph definition for the agent loop."""

from langgraph.graph import StateGraph, END

from nanobot.langgraph.graph.state import AgentState


def should_continue(state: AgentState) -> str:
    """
    Entscheidet, ob der Loop weiterläuft (Tool-Aufrufe) oder endet.
    """
    last_message = state["messages"][-1]

    if not last_message:
        return "end"

    # Wenn letzte Message Tool-Aufrufe enthält → weiter zu Tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # Keine Tool-Aufrufe → Ende
    return "end"


def should_spawn_subagent(state: AgentState) -> str:
    """
    Prüft, ob spawn tool aufgerufen wurde.
    """
    last_message = state["messages"][-1]

    if not last_message:
        return should_continue(state)

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            if tool_call.name == "spawn":
                return "spawn"

    # Sonst: normaler Tool-Loop oder Ende
    return should_continue(state)


def create_main_graph(config) -> StateGraph:
    """Erstellt den Haupt-Graph für den Agent Loop."""

    graph = StateGraph(AgentState)

    # Import nodes lazily to avoid circular imports
    from nanobot.langgraph.nodes.agent_node import call_model
    from nanobot.langgraph.nodes.tools_node import execute_tools
    from nanobot.langgraph.nodes.state_update_node import update_state
    from nanobot.langgraph.nodes.subagent_node import spawn_subagent_node

    # Nodes registrieren
    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("update_state", update_state)
    graph.add_node("spawn_subagent", spawn_subagent_node)

    # Kanten definieren
    graph.set_entry_point("agent")

    # Agent → Tools oder Spawn?
    graph.add_conditional_edges(
        "agent",
        should_spawn_subagent,
        {"spawn": "spawn_subagent", "continue": "tools", "end": "update_state"},
    )

    # Spawn Subagent → zurück zu Agent
    graph.add_edge("spawn_subagent", "update_state")

    # Tools → State Update → zurück zu Agent
    graph.add_edge("tools", "update_state")
    graph.add_edge("update_state", "agent")

    # Update State → Prüfen, ob fertig
    graph.add_conditional_edges("update_state", should_continue, {"continue": "agent", "end": END})

    return graph

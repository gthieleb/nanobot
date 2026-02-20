"""State update node."""

from typing import Any

from nanobot.langgraph.graph.state import AgentState


def update_state(state: AgentState, config) -> dict[str, Any]:
    """
    Aktualisiert State für nächsten Loop.
    """
    # Current Context aktualisieren
    # (wird von Subagent initial gefüllt)

    # Subagent-Tasks bereinigen (abgeschlossene entfernen)
    state["subagent_tasks"] = [
        task
        for task in state["subagent_tasks"]
        if task.get("status") not in ["completed", "failed", "cancelled"]
    ]

    return state

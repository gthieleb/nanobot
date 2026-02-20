"""State definitions for LangGraph graphs."""

from typing import Annotated, Sequence, TypedDict, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State für den Haupt-Agenten (persistiert durch MemorySaver).
    """

    messages: Annotated[Sequence, add_messages]
    current_tools: list[str]
    subagent_tasks: list[dict[str, Any]]
    current_context: dict[str, Any]


class SubagentState(TypedDict):
    """
    State für Subagents (isoliert, aber initialer Kontext von Main).
    """

    task_id: str
    task: str
    initial_context: dict[str, Any]
    messages: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    result: str | None
    status: str  # "running", "completed", "failed", "awaiting_adjustment"

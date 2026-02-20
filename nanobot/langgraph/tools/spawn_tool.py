"""Spawn tool for LangGraph main graph."""

from pydantic import BaseModel, Field


class SpawnToolInput(BaseModel):
    task: str = Field(description="The task for subagent to complete")
    label: str | None = Field(default=None, description="Optional short label for display")


class SpawnTool:
    """
    Spawn Tool für Main Graph.
    """

    name = "spawn"
    description = (
        "Spawn a subagent to handle a task in background. "
        "Use this for complex or time-consuming tasks that can run independently. "
        "The subagent will complete the task and report back when done."
    )
    args_schema = SpawnToolInput

    def __init__(self, manager):
        self._manager = manager

    def _run(self, task: str, label: str | None = None) -> str:
        """
        Wird vom Spawn Subagent Node aufgerufen (nicht direkt hier).
        """
        # Die eigentliche Ausführung passiert im spawn_subagent_node
        # Dieses Tool ist nur für die LLM-Definition
        return f"Spawning subagent for task: {task}"

"""LangGraph-specific configuration."""

from pydantic import BaseModel, Field


class LangGraphSettings(BaseModel):
    """Konfiguration für LangGraph-Integration."""

    enabled: bool = Field(default=False, description="Flag für Parallel-Migration")
    checkpoint_type: str = Field(default="memory", description="memory, postgres, redis")
    subagent_max_iterations: int = Field(default=15, description="Max iterations for subagents")
    adjustment_interval: int = Field(
        default=3, description="Alle N Iterationen Adjustierung anfragen"
    )

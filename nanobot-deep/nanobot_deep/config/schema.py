"""DeepAgents-specific configuration schema.

This config controls the LangGraph-based agent backend.
It is separate from nanobot's main config and contains only
deepagents-specific settings.

Config file: ~/.nanobot/deepagents.json

Usage:
    from nanobot.config.deepagents_loader import load_deepagents_config

    config = load_deepagents_config()
    agent = create_deep_agent(
        recursion_limit=config.recursion_limit,
        skills=config.skills,
        ...
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base model with camelCase support."""

    class Config:
        populate_by_name = True


class DeepAgentsModelConfig(BaseConfig):
    """Model configuration for deepagents.

    For standalone use (ralph mode), specify model and api_key here.
    When used with nanobot, these are overridden by nanobot's config.json.
    """

    name: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    max_tokens: int = 20000
    temperature: float = 0.1


class DeepAgentsSubagentConfig(BaseConfig):
    """Subagent configuration."""

    name: str
    description: str
    system_prompt: str | None = None
    model: str | None = None
    tools: list[str] | None = None


class DeepAgentsInterruptConfig(BaseConfig):
    """Human-in-the-loop interrupt configuration.

    Set to true to pause agent execution before that tool runs.
    """

    edit_file: bool = False
    write_file: bool = False
    execute: bool = False
    all_tools: bool = False


class DeepAgentsBackendConfig(BaseConfig):
    """Backend/sandbox configuration overrides.

    Note: The sandbox type comes from nanobot config (agents.defaults.sandbox).
    These are additional backend-specific settings.
    """

    sandbox_id: str | None = None
    setup_script: str | None = None
    restrict_to_workspace: bool = False
    exec_timeout: int = 60
    path_append: str = ""


class DeepAgentsCheckpointerConfig(BaseConfig):
    """Checkpointer (session persistence) configuration."""

    type: Literal["sqlite", "memory", "none"] = "sqlite"
    path: str = "~/.nanobot/sessions.db"


class DeepAgentsSummarizationConfig(BaseConfig):
    """Summarization middleware configuration."""

    enabled: bool = True
    trigger_tokens: int = 100000
    keep_messages: int = 20


class DeepAgentsMiddlewareConfig(BaseConfig):
    """Middleware toggle configuration."""

    enable_todolist: bool = True
    enable_summarization: bool = True
    enable_prompt_caching: bool = True
    enable_skills: bool = True
    enable_memory: bool = True
    enable_subagents: bool = True
    enable_filesystem: bool = True


class DeepAgentsTaskRoutingConfig(BaseConfig):
    """Task routing configuration for control vs delegate distinction."""

    auto_delegate_reply_to: bool = True
    control_commands: list[str] = Field(
        default_factory=lambda: ["/help", "/new", "/stop", "/tasks", "/start"]
    )
    delegate_threshold_tokens: int = 1000


class DeepAgentsConfig(BaseConfig):
    """DeepAgents-specific configuration.

    This config controls only the deepagents/LangGraph-specific behavior.
    All LLM, provider, tool, and channel settings come from nanobot's
    main config.json.

    File: ~/.nanobot/deepagents.json
    """

    model: DeepAgentsModelConfig = Field(default_factory=DeepAgentsModelConfig)
    recursion_limit: int = 500
    debug: bool = False
    name: str = "nanobot-deep-agent"

    backend: DeepAgentsBackendConfig = Field(default_factory=DeepAgentsBackendConfig)
    checkpointer: DeepAgentsCheckpointerConfig = Field(default_factory=DeepAgentsCheckpointerConfig)

    skills: list[str] = Field(default_factory=lambda: ["~/.nanobot/workspace/skills"])
    memory: list[str] = Field(default_factory=list)

    subagents: list[DeepAgentsSubagentConfig] = Field(default_factory=list)

    interrupt_on: DeepAgentsInterruptConfig = Field(default_factory=DeepAgentsInterruptConfig)

    summarization: DeepAgentsSummarizationConfig = Field(
        default_factory=DeepAgentsSummarizationConfig
    )

    middleware: DeepAgentsMiddlewareConfig = Field(default_factory=DeepAgentsMiddlewareConfig)

    task_routing: DeepAgentsTaskRoutingConfig = Field(default_factory=DeepAgentsTaskRoutingConfig)

    def get_skills_paths(self, workspace: Path | None = None) -> list[str]:
        """Get expanded skills paths."""
        paths = []
        for skill_path in self.skills:
            expanded = Path(skill_path).expanduser()
            if not expanded.is_absolute() and workspace:
                expanded = workspace / skill_path
            paths.append(str(expanded))
        return paths

    def get_memory_paths(self, workspace: Path | None = None) -> list[str]:
        """Get expanded memory paths."""
        paths = []
        for mem_path in self.memory:
            expanded = Path(mem_path).expanduser()
            if not expanded.is_absolute() and workspace:
                expanded = workspace / mem_path
            paths.append(str(expanded))
        return paths

    def get_checkpointer_path(self) -> Path:
        """Get expanded checkpointer path."""
        return Path(self.checkpointer.path).expanduser()

    def get_interrupt_on_config(self) -> dict[str, bool]:
        """Convert interrupt_on to deepagents format."""
        if self.interrupt_on.all_tools:
            return {
                "edit_file": True,
                "write_file": True,
                "execute": True,
            }
        return {
            "edit_file": self.interrupt_on.edit_file,
            "write_file": self.interrupt_on.write_file,
            "execute": self.interrupt_on.execute,
        }

"""DeepAgents configuration loader.

Handles loading deepagents.json and merging with nanobot config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot_deep.config.schema import DeepAgentsConfig

if TYPE_CHECKING:
    from nanobot.config.schema import Config


def get_deepagents_config_path() -> Path:
    """Get path to deepagents config file."""
    return Path.home() / ".nanobot" / "deepagents.json"


def load_deepagents_config(
    config_path: Path | None = None,
    override: dict[str, Any] | None = None,
) -> DeepAgentsConfig:
    """Load deepagents config from file.

    Args:
        config_path: Path to deepagents.json (default: ~/.nanobot/deepagents.json)
        override: Dict to override loaded values

    Returns:
        DeepAgentsConfig instance
    """
    if config_path is None:
        config_path = get_deepagents_config_path()

    data = {}

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded deepagents config from {config_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {config_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load {config_path}: {e}")

    if override:
        data = _deep_merge(data, override)

    return DeepAgentsConfig(**data)


def save_deepagents_config(
    config: DeepAgentsConfig,
    config_path: Path | None = None,
) -> None:
    """Save deepagents config to file.

    Args:
        config: DeepAgentsConfig instance
        config_path: Path to save to (default: ~/.nanobot/deepagents.json)
    """
    if config_path is None:
        config_path = get_deepagents_config_path()

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(exclude_none=True), f, indent=2)

    logger.info(f"Saved deepagents config to {config_path}")


def merge_with_nanobot_config(
    nanobot_config: "Config",
    deepagents_config: DeepAgentsConfig | None = None,
) -> DeepAgentsConfig:
    """Merge nanobot config into deepagents config.

    nanobot config provides:
    - Model name, provider, api_key
    - max_tokens, temperature
    - sandbox type
    - MCP servers
    - tools.exec settings
    - workspace path

    deepagents.json provides:
    - recursion_limit
    - skills, memory paths
    - subagents
    - interrupt_on
    - summarization settings
    - middleware toggles

    Args:
        nanobot_config: nanobot Config instance
        deepagents_config: DeepAgentsConfig instance (loaded if None)

    Returns:
        Merged DeepAgentsConfig
    """
    if deepagents_config is None:
        deepagents_config = load_deepagents_config()

    workspace = nanobot_config.workspace_path

    merged = deepagents_config.model_copy(deep=True)

    defaults = nanobot_config.agents.defaults

    merged.recursion_limit = defaults.max_tool_iterations + 10

    sandbox = defaults.sandbox
    if sandbox and sandbox != "none":
        pass

    merged.backend.exec_timeout = nanobot_config.tools.exec.timeout
    merged.backend.path_append = nanobot_config.tools.exec.path_append
    merged.backend.restrict_to_workspace = nanobot_config.tools.restrict_to_workspace

    if not deepagents_config.skills or len(deepagents_config.skills) == 0:
        merged.skills = [str(workspace / "skills")]
    elif any("~/.nanobot" in s for s in deepagents_config.skills):
        expanded_skills = []
        for skill_path in deepagents_config.skills:
            if "~/.nanobot/workspace" in skill_path:
                expanded_skills.append(str(workspace / "skills"))
            else:
                expanded_skills.append(skill_path)
        merged.skills = expanded_skills

    merged.middleware.enable_subagents = True

    return merged


def create_default_deepagents_config() -> DeepAgentsConfig:
    """Create default deepagents config."""
    return DeepAgentsConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts, override takes precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result

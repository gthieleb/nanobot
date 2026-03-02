"""Configuration module for nanobot."""

from nanobot.config.loader import load_config, get_config_path
from nanobot.config.schema import Config
from nanobot.config.deepagents_schema import DeepAgentsConfig
from nanobot.config.deepagents_loader import (
    load_deepagents_config,
    save_deepagents_config,
    merge_with_nanobot_config,
    get_deepagents_config_path,
)

__all__ = [
    "Config",
    "load_config",
    "get_config_path",
    "DeepAgentsConfig",
    "load_deepagents_config",
    "save_deepagents_config",
    "merge_with_nanobot_config",
    "get_deepagents_config_path",
]

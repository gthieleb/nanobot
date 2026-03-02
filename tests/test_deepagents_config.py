"""Unit tests for DeepAgents config schema and loader.

Tests:
- Schema validation and defaults
- Config loading and saving
- Merging with nanobot config
- Config precedence rules
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.config.deepagents_schema import (
    DeepAgentsConfig,
    DeepAgentsInterruptConfig,
    DeepAgentsBackendConfig,
    DeepAgentsCheckpointerConfig,
    DeepAgentsSummarizationConfig,
    DeepAgentsMiddlewareConfig,
    DeepAgentsSubagentConfig,
)
from nanobot.config.deepagents_loader import (
    load_deepagents_config,
    save_deepagents_config,
    merge_with_nanobot_config,
    get_deepagents_config_path,
    _deep_merge,
)


class TestDeepAgentsSchema:
    """Tests for DeepAgentsConfig schema."""

    def test_default_config_values(self):
        """Test default values are set correctly."""
        config = DeepAgentsConfig()

        assert config.recursion_limit == 500
        assert config.debug is False
        assert config.name == "nanobot-deep-agent"
        assert config.checkpointer.type == "sqlite"
        assert config.middleware.enable_todolist is True
        assert config.middleware.enable_summarization is True

    def test_interrupt_config_defaults(self):
        """Test interrupt_on default values."""
        config = DeepAgentsInterruptConfig()

        assert config.edit_file is False
        assert config.write_file is False
        assert config.execute is False
        assert config.all_tools is False

    def test_interrupt_config_custom(self):
        """Test interrupt_on custom values."""
        config = DeepAgentsInterruptConfig(edit_file=True, execute=True)

        assert config.edit_file is True
        assert config.write_file is False
        assert config.execute is True

    def test_interrupt_all_tools_enables_all(self):
        """Test all_tools=true enables all interrupts in get_interrupt_on_config."""
        config = DeepAgentsConfig(interrupt_on=DeepAgentsInterruptConfig(all_tools=True))

        result = config.get_interrupt_on_config()

        assert result["edit_file"] is True
        assert result["write_file"] is True
        assert result["execute"] is True

    def test_interrupt_partial(self):
        """Test partial interrupt config."""
        config = DeepAgentsConfig(interrupt_on=DeepAgentsInterruptConfig(edit_file=True))

        result = config.get_interrupt_on_config()

        assert result["edit_file"] is True
        assert result["write_file"] is False
        assert result["execute"] is False

    def test_skills_path_expansion(self):
        """Test skills paths are expanded correctly."""
        config = DeepAgentsConfig(skills=["~/skills", "./relative"])

        workspace = Path("/workspace")
        paths = config.get_skills_paths(workspace)

        home = str(Path.home())
        assert paths[0] == f"{home}/skills"
        assert paths[1] == "/workspace/relative"

    def test_skills_path_absolute(self):
        """Test absolute skills paths are preserved."""
        config = DeepAgentsConfig(skills=["/absolute/path/skills"])

        paths = config.get_skills_paths()

        assert paths[0] == "/absolute/path/skills"

    def test_memory_path_expansion(self):
        """Test memory paths are expanded correctly."""
        config = DeepAgentsConfig(memory=["~/AGENTS.md"])

        paths = config.get_memory_paths()

        home = str(Path.home())
        assert paths[0] == f"{home}/AGENTS.md"

    def test_checkpointer_path_expansion(self):
        """Test checkpointer path expansion."""
        config = DeepAgentsConfig(checkpointer=DeepAgentsCheckpointerConfig(path="~/sessions.db"))

        path = config.get_checkpointer_path()
        home = str(Path.home())
        assert str(path) == f"{home}/sessions.db"

    def test_checkpointer_types(self):
        """Test valid checkpointer types."""
        for ctype in ["sqlite", "memory", "none"]:
            config = DeepAgentsConfig(checkpointer=DeepAgentsCheckpointerConfig(type=ctype))
            assert config.checkpointer.type == ctype

    def test_backend_config_defaults(self):
        """Test backend config defaults."""
        config = DeepAgentsBackendConfig()

        assert config.sandbox_id is None
        assert config.setup_script is None
        assert config.restrict_to_workspace is False
        assert config.exec_timeout == 60
        assert config.path_append == ""

    def test_summarization_config(self):
        """Test summarization config."""
        config = DeepAgentsSummarizationConfig(
            enabled=True,
            trigger_tokens=50000,
            keep_messages=10,
        )

        assert config.enabled is True
        assert config.trigger_tokens == 50000
        assert config.keep_messages == 10

    def test_middleware_config(self):
        """Test middleware toggles."""
        config = DeepAgentsMiddlewareConfig(
            enable_todolist=False,
            enable_prompt_caching=False,
        )

        assert config.enable_todolist is False
        assert config.enable_prompt_caching is False
        assert config.enable_summarization is True  # Default

    def test_subagent_config(self):
        """Test subagent configuration."""
        config = DeepAgentsConfig(
            subagents=[
                DeepAgentsSubagentConfig(
                    name="researcher",
                    description="Researches topics",
                    model="anthropic:claude-sonnet-4-5",
                )
            ]
        )

        assert len(config.subagents) == 1
        assert config.subagents[0].name == "researcher"
        assert config.subagents[0].description == "Researches topics"

    def test_full_config_from_dict(self):
        """Test creating full config from dict."""
        data = {
            "recursion_limit": 1000,
            "debug": True,
            "name": "test-agent",
            "interrupt_on": {"edit_file": True},
            "summarization": {"trigger_tokens": 50000},
        }

        config = DeepAgentsConfig(**data)

        assert config.recursion_limit == 1000
        assert config.debug is True
        assert config.name == "test-agent"
        assert config.interrupt_on.edit_file is True
        assert config.summarization.trigger_tokens == 50000


class TestDeepAgentsLoader:
    """Tests for config loader functions."""

    def test_get_config_path(self):
        """Test default config path."""
        path = get_deepagents_config_path()

        assert ".nanobot" in str(path)
        assert path.name == "deepagents.json"

    def test_load_missing_file_returns_defaults(self, tmp_path):
        """Test loading when file doesn't exist returns defaults."""
        config_path = tmp_path / "deepagents.json"
        config = load_deepagents_config(config_path)

        assert config.recursion_limit == 500  # Default
        assert config.debug is False

    def test_load_valid_file(self, tmp_path):
        """Test loading valid config file."""
        config_path = tmp_path / "deepagents.json"
        config_path.write_text(
            json.dumps(
                {
                    "recursion_limit": 1000,
                    "debug": True,
                    "name": "custom-agent",
                }
            )
        )

        config = load_deepagents_config(config_path)

        assert config.recursion_limit == 1000
        assert config.debug is True
        assert config.name == "custom-agent"

    def test_load_partial_file(self, tmp_path):
        """Test loading partial config uses defaults for missing fields."""
        config_path = tmp_path / "deepagents.json"
        config_path.write_text(
            json.dumps(
                {
                    "recursion_limit": 750,
                }
            )
        )

        config = load_deepagents_config(config_path)

        assert config.recursion_limit == 750
        assert config.debug is False  # Default
        assert config.middleware.enable_todolist is True  # Default

    def test_load_invalid_json_falls_back(self, tmp_path):
        """Test handling of invalid JSON."""
        config_path = tmp_path / "deepagents.json"
        config_path.write_text("{ invalid json }")

        config = load_deepagents_config(config_path)

        assert config.recursion_limit == 500  # Falls back to default

    def test_load_empty_file(self, tmp_path):
        """Test handling of empty file."""
        config_path = tmp_path / "deepagents.json"
        config_path.write_text("")

        config = load_deepagents_config(config_path)

        assert config.recursion_limit == 500  # Default

    def test_save_config(self, tmp_path):
        """Test saving config to file."""
        config_path = tmp_path / "deepagents.json"
        config = DeepAgentsConfig(recursion_limit=750, debug=True)

        save_deepagents_config(config, config_path)

        assert config_path.exists()
        loaded = json.loads(config_path.read_text())
        assert loaded["recursion_limit"] == 750
        assert loaded["debug"] is True

    def test_save_creates_directory(self, tmp_path):
        """Test save creates parent directory if needed."""
        config_path = tmp_path / "subdir" / "deepagents.json"
        config = DeepAgentsConfig()

        save_deepagents_config(config, config_path)

        assert config_path.exists()

    def test_override_dict(self, tmp_path):
        """Test override parameter takes precedence."""
        config_path = tmp_path / "deepagents.json"
        config_path.write_text(json.dumps({"recursion_limit": 500}))

        config = load_deepagents_config(
            config_path, override={"recursion_limit": 999, "debug": True}
        )

        assert config.recursion_limit == 999
        assert config.debug is True

    def test_deep_merge_nested(self):
        """Test deep merge of nested dicts."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 4}

        result = _deep_merge(base, override)

        assert result == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}


class TestMergeWithNanobotConfig:
    """Tests for merging nanobot config with deepagents config."""

    def _make_nanobot_config(self, **kwargs):
        """Create mock nanobot config."""
        mock = MagicMock()
        mock.agents.defaults.max_tool_iterations = kwargs.get("max_tool_iterations", 40)
        mock.agents.defaults.sandbox = kwargs.get("sandbox", "none")
        mock.tools.exec.timeout = kwargs.get("exec_timeout", 60)
        mock.tools.exec.path_append = kwargs.get("path_append", "")
        mock.tools.restrict_to_workspace = kwargs.get("restrict_to_workspace", False)
        mock.workspace_path = kwargs.get("workspace", Path("/workspace"))
        return mock

    def test_recursion_limit_from_max_iterations(self):
        """Test recursion_limit is derived from max_tool_iterations."""
        nanobot = self._make_nanobot_config(max_tool_iterations=50)

        merged = merge_with_nanobot_config(nanobot)

        assert merged.recursion_limit == 60  # 50 + 10

    def test_recursion_limit_default(self):
        """Test recursion_limit with default max_tool_iterations."""
        nanobot = self._make_nanobot_config(max_tool_iterations=40)

        merged = merge_with_nanobot_config(nanobot)

        assert merged.recursion_limit == 50  # 40 + 10

    def test_backend_exec_timeout_merged(self):
        """Test exec_timeout from nanobot is merged."""
        nanobot = self._make_nanobot_config(exec_timeout=120)

        merged = merge_with_nanobot_config(nanobot)

        assert merged.backend.exec_timeout == 120

    def test_backend_path_append_merged(self):
        """Test path_append from nanobot is merged."""
        nanobot = self._make_nanobot_config(path_append="/usr/local/bin")

        merged = merge_with_nanobot_config(nanobot)

        assert merged.backend.path_append == "/usr/local/bin"

    def test_backend_restrict_to_workspace_merged(self):
        """Test restrict_to_workspace from nanobot is merged."""
        nanobot = self._make_nanobot_config(restrict_to_workspace=True)

        merged = merge_with_nanobot_config(nanobot)

        assert merged.backend.restrict_to_workspace is True

    def test_skills_default_to_workspace(self):
        """Test skills path defaults to workspace/skills."""
        nanobot = self._make_nanobot_config(workspace=Path("/my/workspace"))

        merged = merge_with_nanobot_config(nanobot)

        assert len(merged.skills) >= 1
        assert "/my/workspace/skills" in merged.skills

    def test_deepagents_config_preserved(self, tmp_path):
        """Test deepagents.json values are preserved when merging."""
        dg_path = tmp_path / "deepagents.json"
        dg_path.write_text(
            json.dumps(
                {
                    "debug": True,
                    "interrupt_on": {"edit_file": True},
                    "summarization": {"trigger_tokens": 50000},
                }
            )
        )

        nanobot = self._make_nanobot_config()

        with patch(
            "nanobot.config.deepagents_loader.get_deepagents_config_path", return_value=dg_path
        ):
            merged = merge_with_nanobot_config(nanobot)

        # deepagents.json settings preserved
        assert merged.debug is True
        assert merged.interrupt_on.edit_file is True
        assert merged.summarization.trigger_tokens == 50000


class TestConfigPrecedence:
    """Tests for config precedence rules."""

    def test_nanobot_overrides_recursion_limit(self, tmp_path):
        """Test nanobot takes precedence for recursion_limit (shared field)."""
        dg_path = tmp_path / "deepagents.json"
        dg_path.write_text(
            json.dumps(
                {
                    "recursion_limit": 2000,  # deepagents.json says 2000
                }
            )
        )

        # nanobot says 40 iterations -> 50 recursion_limit
        mock = MagicMock()
        mock.agents.defaults.max_tool_iterations = 40
        mock.agents.defaults.sandbox = "none"
        mock.tools.exec.timeout = 60
        mock.tools.exec.path_append = ""
        mock.tools.restrict_to_workspace = False
        mock.workspace_path = Path("/workspace")

        with patch(
            "nanobot.config.deepagents_loader.get_deepagents_config_path", return_value=dg_path
        ):
            merged = merge_with_nanobot_config(mock)

        # nanobot wins for shared field
        assert merged.recursion_limit == 50

    def test_deepagents_specific_settings_not_overridden(self, tmp_path):
        """Test deepagents-only settings are preserved."""
        dg_path = tmp_path / "deepagents.json"
        dg_path.write_text(
            json.dumps(
                {
                    "interrupt_on": {"all_tools": True},
                    "summarization": {"trigger_tokens": 50000, "keep_messages": 5},
                    "middleware": {"enable_todolist": False, "enable_prompt_caching": False},
                }
            )
        )

        mock = MagicMock()
        mock.agents.defaults.max_tool_iterations = 40
        mock.agents.defaults.sandbox = "none"
        mock.tools.exec.timeout = 60
        mock.tools.exec.path_append = ""
        mock.tools.restrict_to_workspace = False
        mock.workspace_path = Path("/workspace")

        with patch(
            "nanobot.config.deepagents_loader.get_deepagents_config_path", return_value=dg_path
        ):
            merged = merge_with_nanobot_config(mock)

        # deepagents-specific preserved
        assert merged.interrupt_on.all_tools is True
        assert merged.summarization.trigger_tokens == 50000
        assert merged.summarization.keep_messages == 5
        assert merged.middleware.enable_todolist is False
        assert merged.middleware.enable_prompt_caching is False

    def test_no_deepagents_json_uses_defaults(self):
        """Test missing deepagents.json falls back to defaults."""
        mock = MagicMock()
        mock.agents.defaults.max_tool_iterations = 30
        mock.agents.defaults.sandbox = "none"
        mock.tools.exec.timeout = 60
        mock.tools.exec.path_append = ""
        mock.tools.restrict_to_workspace = False
        mock.workspace_path = Path("/workspace")

        with patch(
            "nanobot.config.deepagents_loader.get_deepagents_config_path",
            return_value=Path("/nonexistent/deepagents.json"),
        ):
            merged = merge_with_nanobot_config(mock)

        assert merged.recursion_limit == 40  # 30 + 10
        assert merged.interrupt_on.edit_file is False  # Default
        assert merged.middleware.enable_summarization is True  # Default
        assert merged.middleware.enable_todolist is True  # Default

    def test_backend_settings_from_nanobot(self, tmp_path):
        """Test backend settings come from nanobot config."""
        dg_path = tmp_path / "deepagents.json"
        dg_path.write_text(
            json.dumps(
                {
                    "backend": {
                        "exec_timeout": 30,  # This should be overridden
                        "path_append": "/should/be/overridden",
                    }
                }
            )
        )

        mock = MagicMock()
        mock.agents.defaults.max_tool_iterations = 40
        mock.agents.defaults.sandbox = "none"
        mock.tools.exec.timeout = 120
        mock.tools.exec.path_append = "/usr/bin"
        mock.tools.restrict_to_workspace = True
        mock.workspace_path = Path("/workspace")

        with patch(
            "nanobot.config.deepagents_loader.get_deepagents_config_path", return_value=dg_path
        ):
            merged = merge_with_nanobot_config(mock)

        # nanobot wins for backend settings
        assert merged.backend.exec_timeout == 120
        assert merged.backend.path_append == "/usr/bin"
        assert merged.backend.restrict_to_workspace is True

    def test_subagents_from_deepagents_json(self, tmp_path):
        """Test subagents are loaded from deepagents.json."""
        dg_path = tmp_path / "deepagents.json"
        dg_path.write_text(
            json.dumps(
                {
                    "subagents": [
                        {
                            "name": "researcher",
                            "description": "Research assistant",
                            "model": "anthropic:claude-sonnet-4-5",
                        }
                    ]
                }
            )
        )

        mock = MagicMock()
        mock.agents.defaults.max_tool_iterations = 40
        mock.agents.defaults.sandbox = "none"
        mock.tools.exec.timeout = 60
        mock.tools.exec.path_append = ""
        mock.tools.restrict_to_workspace = False
        mock.workspace_path = Path("/workspace")

        with patch(
            "nanobot.config.deepagents_loader.get_deepagents_config_path", return_value=dg_path
        ):
            merged = merge_with_nanobot_config(mock)

        assert len(merged.subagents) == 1
        assert merged.subagents[0].name == "researcher"

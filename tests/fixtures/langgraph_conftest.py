"""Pytest fixtures for LangGraph tests."""

import pytest
import tempfile
from pathlib import Path
import shutil


@pytest.fixture
def tmp_path():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def test_config_path(tmp_path):
    """Path to test config directory."""
    configs_dir = tmp_path / "langgraph_configs"
    configs_dir.mkdir(exist_ok=True)

    # Copy test configs from fixtures
    source_configs = Path(__file__).parent / "langgraph_configs"
    if source_configs.exists():
        shutil.copytree(source_configs, configs_dir, dirs_exist_ok=True)

    return configs_dir


@pytest.fixture
def mock_provider():
    """Create mock LLM provider."""
    from tests.fixtures.mocks.mock_provider import MockLLMProvider

    return MockLLMProvider()


@pytest.fixture
def mock_bus():
    """Create mock message bus."""
    from tests.fixtures.mocks.mock_bus import MockMessageBus

    return MockMessageBus()

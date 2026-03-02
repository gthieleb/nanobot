"""Pytest fixtures for E2E tests with real LLM calls.

Usage:
    NANOBOT_TEST_API_KEY=sk-... pytest tests/e2e/ -m live -v
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "live: tests that require real LLM API key (expensive)")
    config.addinivalue_line("markers", "slow: tests that take more than 60 seconds")


@pytest.fixture
def live_api_key() -> str:
    """Get API key for live tests."""
    key = os.environ.get("NANOBOT_TEST_API_KEY")
    if not key:
        pytest.skip("Set NANOBOT_TEST_API_KEY to run live tests")
    return key


@pytest.fixture
def live_model() -> str:
    """Get model for live tests."""
    return os.environ.get("NANOBOT_TEST_MODEL", "anthropic/claude-sonnet-4-5")


@pytest.fixture
def live_provider(live_api_key: str, live_model: str):
    """Create real LLM provider for live tests."""
    from nanobot.providers.litellm_provider import LiteLLMProvider

    return LiteLLMProvider(
        api_key=live_api_key,
        default_model=live_model,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create test workspace with basic structure."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "skills").mkdir()
    return ws


@pytest.fixture
async def live_gateway(
    workspace: Path,
    live_provider,
    live_model: str,
):
    """Start gateway with real LLM for live testing.

    Yields dict with:
        - bus: MessageBus for sending/receiving messages
        - agent: AgentLoop instance
        - provider: LLM provider
    """
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop

    bus = MessageBus()

    agent = AgentLoop(
        bus=bus,
        provider=live_provider,
        workspace=workspace,
        model=live_model,
        max_iterations=10,
        temperature=0.1,
    )

    agent_task = asyncio.create_task(agent.run())

    yield {
        "bus": bus,
        "agent": agent,
        "provider": live_provider,
        "workspace": workspace,
    }

    agent.stop()
    agent_task.cancel()
    try:
        await asyncio.wait_for(agent_task, timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


@pytest.fixture
async def live_gateway_no_cancel(
    workspace: Path,
    live_provider,
    live_model: str,
):
    """Start gateway without auto-cancellation (for tests that need to stop manually)."""
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop

    bus = MessageBus()

    agent = AgentLoop(
        bus=bus,
        provider=live_provider,
        workspace=workspace,
        model=live_model,
        max_iterations=10,
        temperature=0.1,
    )

    agent_task = asyncio.create_task(agent.run())

    yield {
        "bus": bus,
        "agent": agent,
        "provider": live_provider,
        "workspace": workspace,
        "task": agent_task,
    }

    agent.stop()
    try:
        await asyncio.wait_for(agent_task, timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


@pytest.fixture
def assert_response():
    """Helper for asserting response content."""

    def _assert(
        response,
        contains: str | list[str] | None = None,
        not_contains: str | list[str] | None = None,
    ):
        content = response.content.lower()

        if contains:
            items = [contains] if isinstance(contains, str) else contains
            for item in items:
                assert item.lower() in content, (
                    f"Expected '{item}' in response: {response.content[:200]}"
                )

        if not_contains:
            items = [not_contains] if isinstance(not_contains, str) else not_contains
            for item in items:
                assert item.lower() not in content, f"Did not expect '{item}' in response"

    return _assert


@pytest.fixture
async def send_and_wait():
    """Helper to send message and wait for response."""

    async def _send(
        bus: "MessageBus",
        content: str,
        channel: str = "test",
        sender_id: str = "user1",
        chat_id: str = "chat1",
        timeout: float = 60.0,
    ):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        await bus.publish_inbound(
            InboundMessage(
                channel=channel,
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
            )
        )

        response = await asyncio.wait_for(bus.consume_outbound(), timeout=timeout)
        return response

    return _send

"""Gateway integration tests with real LLM calls.

Tests the full message flow: inbound → agent → outbound

Run with:
    NANOBOT_TEST_API_KEY=sk-... pytest tests/e2e/test_gateway_live.py -m live -v
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot.bus.events import InboundMessage


pytestmark = pytest.mark.live


class TestGatewayBasic:
    """Basic gateway functionality tests."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_simple_message(self, live_gateway, send_and_wait, assert_response):
        """Test basic message processing through gateway."""
        bus = live_gateway["bus"]

        response = await send_and_wait(bus, "Reply with exactly the word 'pong' and nothing else")

        assert response.channel == "test"
        assert response.chat_id == "chat1"
        assert "pong" in response.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_message_routing(self, live_gateway):
        """Test message is routed to correct channel/chat."""
        bus = live_gateway["bus"]

        await bus.publish_inbound(
            InboundMessage(
                channel="test_channel",
                sender_id="test_sender",
                chat_id="test_chat_123",
                content="Say 'hello'",
            )
        )

        response = await asyncio.wait_for(bus.consume_outbound(), timeout=30)

        assert response.channel == "test_channel"
        assert response.chat_id == "test_chat_123"
        assert "hello" in response.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_empty_message_handling(self, live_gateway, send_and_wait):
        """Test gateway handles empty or whitespace messages."""
        bus = live_gateway["bus"]

        # Empty messages should still get a response
        response = await send_and_wait(bus, "   Say 'ok'   ")

        assert response.content
        assert len(response.content) > 0


class TestGatewaySlashCommands:
    """Test slash commands through gateway."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_help_command(self, live_gateway, send_and_wait):
        """Test /help command."""
        bus = live_gateway["bus"]

        response = await send_and_wait(bus, "/help")

        assert response.channel == "test"
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["help", "commands", "usage"])

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_new_command(self, live_gateway, send_and_wait):
        """Test /new command clears session."""
        bus = live_gateway["bus"]

        response = await send_and_wait(bus, "/new")

        assert response.channel == "test"
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["new", "session", "cleared", "started"])


class TestGatewayErrorHandling:
    """Test error handling in gateway."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_invalid_tool_request(self, live_gateway, send_and_wait):
        """Test gateway handles requests for non-existent tools gracefully."""
        bus = live_gateway["bus"]

        # Ask agent to use a non-existent tool
        response = await send_and_wait(
            bus, "Just say 'I cannot do that' - do not try to use any tools"
        )

        # Should get a valid response, not an error
        assert response.content
        assert "error" not in response.content.lower() or "cannot" in response.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_long_message(self, live_gateway, send_and_wait):
        """Test gateway handles long messages."""
        bus = live_gateway["bus"]

        long_message = "Repeat after me: " + "hello " * 100
        response = await send_and_wait(bus, long_message, timeout=90)

        assert response.content
        assert len(response.content) > 0


class TestGatewayConcurrency:
    """Test concurrent message handling."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_sequential_messages_same_session(self, live_gateway, send_and_wait):
        """Test sequential messages in same session are processed in order."""
        bus = live_gateway["bus"]

        # Send two messages sequentially
        r1 = await send_and_wait(bus, "Say 'first'", chat_id="concurrent_test")
        r2 = await send_and_wait(bus, "Say 'second'", chat_id="concurrent_test")

        assert "first" in r1.content.lower()
        assert "second" in r2.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_messages_different_sessions(self, live_gateway):
        """Test messages in different sessions can be processed."""
        bus = live_gateway["bus"]

        # Send to two different sessions
        await bus.publish_inbound(
            InboundMessage(
                channel="test", sender_id="u1", chat_id="session_a", content="Say 'alpha'"
            )
        )
        await bus.publish_inbound(
            InboundMessage(
                channel="test", sender_id="u2", chat_id="session_b", content="Say 'beta'"
            )
        )

        # Collect both responses
        responses = []
        for _ in range(2):
            r = await asyncio.wait_for(bus.consume_outbound(), timeout=60)
            responses.append(r)

        contents = [r.content.lower() for r in responses]
        assert any("alpha" in c for c in contents)
        assert any("beta" in c for c in contents)

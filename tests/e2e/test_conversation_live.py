"""Multi-turn conversation tests with real LLM calls.

Tests conversation history and session persistence.

Run with:
    NANOBOT_TEST_API_KEY=sk-... pytest tests/e2e/test_conversation_live.py -m live -v
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot.bus.events import InboundMessage


pytestmark = pytest.mark.live


class TestConversationHistory:
    """Test conversation history is maintained."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_remember_name(self, live_gateway, send_and_wait):
        """Test agent remembers name across turns."""
        bus = live_gateway["bus"]

        # Turn 1: Introduce
        r1 = await send_and_wait(
            bus, "My name is Alice. Just acknowledge briefly.", chat_id="memory_test"
        )
        assert r1.content

        # Turn 2: Recall
        r2 = await send_and_wait(bus, "What is my name?", chat_id="memory_test")

        assert "alice" in r2.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_remember_preference(self, live_gateway, send_and_wait):
        """Test agent remembers preferences across turns."""
        bus = live_gateway["bus"]

        # Turn 1: Set preference
        await send_and_wait(bus, "I prefer Python. Remember this.", chat_id="pref_test")

        # Turn 2: Ask about preference
        r2 = await send_and_wait(bus, "What programming language do I prefer?", chat_id="pref_test")

        assert "python" in r2.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_context_accumulation(self, live_gateway, send_and_wait):
        """Test context accumulates across multiple turns."""
        bus = live_gateway["bus"]

        # Build up context
        await send_and_wait(bus, "Remember: color=blue", chat_id="context_test")
        await send_and_wait(bus, "Remember: animal=cat", chat_id="context_test")
        await send_and_wait(bus, "Remember: food=pizza", chat_id="context_test")

        # Query accumulated context
        r = await send_and_wait(
            bus,
            "Tell me my favorite color, animal, and food in one sentence",
            chat_id="context_test",
        )

        content_lower = r.content.lower()
        assert "blue" in content_lower
        assert "cat" in content_lower
        assert "pizza" in content_lower


class TestSessionIsolation:
    """Test sessions are isolated from each other."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_different_sessions_different_context(self, live_gateway, send_and_wait):
        """Test different sessions don't share context."""
        bus = live_gateway["bus"]

        # Session A: Set name to Alice
        await send_and_wait(bus, "My name is Alice.", chat_id="session_a")

        # Session B: Set name to Bob
        await send_and_wait(bus, "My name is Bob.", chat_id="session_b")

        # Session A: Ask name - should be Alice
        r_a = await send_and_wait(bus, "What is my name?", chat_id="session_a")

        # Session B: Ask name - should be Bob
        r_b = await send_and_wait(bus, "What is my name?", chat_id="session_b")

        assert "alice" in r_a.content.lower()
        assert "bob" in r_b.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_session_isolation_with_tools(self, live_gateway, workspace, send_and_wait):
        """Test sessions don't see each other's files."""
        bus = live_gateway["bus"]

        # Session A: Create file
        await send_and_wait(
            bus,
            "Create a file called session_a_file.txt with content 'from session A'",
            chat_id="session_x",
        )

        # Session B: Try to find session A's file (shouldn't know about it from context)
        r = await send_and_wait(bus, "What files exist? List them.", chat_id="session_y")

        # Session B might see the file on disk but shouldn't have it in context
        # This is a weak test - mainly checking no crash occurs
        assert r.content


class TestConversationReset:
    """Test conversation reset functionality."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_new_command_resets_context(self, live_gateway, send_and_wait):
        """Test /new command clears conversation history."""
        bus = live_gateway["bus"]

        # Set context
        await send_and_wait(bus, "My secret code is XYZ123", chat_id="reset_test")

        # Reset
        await send_and_wait(bus, "/new", chat_id="reset_test")

        # Ask about secret - shouldn't remember
        r = await send_and_wait(bus, "What is my secret code?", chat_id="reset_test")

        # Should not contain the secret
        content_lower = r.content.lower()
        assert (
            "xyz123" not in content_lower
            or "don't know" in content_lower
            or "not sure" in content_lower
        )


class TestConversationEdgeCases:
    """Test edge cases in conversations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_very_short_turns(self, live_gateway, send_and_wait):
        """Test handling of very short messages."""
        bus = live_gateway["bus"]

        r = await send_and_wait(bus, "Hi")
        assert r.content

        r = await send_and_wait(bus, "Ok")
        assert r.content

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_repeated_message(self, live_gateway, send_and_wait):
        """Test handling of repeated identical messages."""
        bus = live_gateway["bus"]

        r1 = await send_and_wait(bus, "Say 'response'", chat_id="repeat_test")
        r2 = await send_and_wait(bus, "Say 'response'", chat_id="repeat_test")

        # Both should have valid responses
        assert r1.content
        assert r2.content

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_long_conversation(self, live_gateway, send_and_wait):
        """Test handling of longer conversations."""
        bus = live_gateway["bus"]

        # Have a multi-turn conversation
        for i in range(5):
            r = await send_and_wait(bus, f"Remember: item{i} = value{i}", chat_id="long_conv_test")
            assert r.content

        # Final question
        r = await send_and_wait(
            bus, "List all items I told you to remember", chat_id="long_conv_test"
        )

        # Should remember most items
        content_lower = r.content.lower()
        items_found = sum(
            1 for i in range(5) if f"item{i}" in content_lower or f"value{i}" in content_lower
        )
        assert items_found >= 3  # Should remember at least 3 of 5

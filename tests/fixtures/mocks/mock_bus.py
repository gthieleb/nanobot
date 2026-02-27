"""Mock message bus for testing."""

from unittest.mock import AsyncMock
from nanobot.bus.events import InboundMessage, OutboundMessage
import asyncio


class MockMessageBus:
    """Mock bus for testing gateway mode."""

    def __init__(self):
        self.inbox = []
        self.outbox = []
        self.consume_method = AsyncMock()
        self.publish_method = AsyncMock()

    async def consume_inbound(self):
        """Wait for inbound message with timeout."""
        if self.inbox:
            return self.inbox.pop(0)
        # Simulate waiting with timeout
        await asyncio.sleep(0.1)
        return None

    async def publish_outbound(self, msg: OutboundMessage):
        """Store outbound message."""
        self.outbox.append(msg)

    async def publish_inbound(self, msg: InboundMessage):
        """Add to inbound queue."""
        self.inbox.append(msg)

"""Messenger for asynchronous adjustments between subagents and main agent."""

import asyncio
from typing import Callable

from nanobot.bus.events import InboundMessage
from loguru import logger


class SubagentMessenger:
    """
    Ermöglicht asynchrone Adjustierungen zwischen Subagent und Main Agent.
    """

    def __init__(self, bus, on_adjustment: Callable):
        self.bus = bus
        self.on_adjustment = on_adjustment
        self._pending_adjustments: dict[str, asyncio.Queue] = {}

    async def request_adjustment(self, task_id: str, current_context: list[dict]) -> dict | None:
        """
        Requestet Adjustierung vom Main Agent.
        """
        # Queue für diese Request erstellen
        self._pending_adjustments[task_id] = asyncio.Queue()

        # Request über Message Bus an Main Agent
        try:
            await self.bus.publish_inbound(
                InboundMessage(
                    channel="system",
                    sender_id=f"subagent:{task_id}",
                    chat_id="adjustment_request",
                    content=f"""[Adjustment Request from Subagent {task_id}]

Current Context:
{str(current_context[-5:] if len(current_context) > 5 else current_context)}

Should I adjust my approach? Please provide feedback.
""",
                )
            )
        except Exception as e:
            logger.error("Failed to publish adjustment request: {}", e)
            return None

        # Warte auf Response mit Timeout
        try:
            adjustment = await asyncio.wait_for(
                self._pending_adjustments[task_id].get(), timeout=30.0
            )
            return adjustment
        except asyncio.TimeoutError:
            logger.warning("Subagent [{}] adjustment request timed out", task_id)
            return None
        finally:
            self._pending_adjustments.pop(task_id, None)

    def deliver_adjustment(self, task_id: str, adjustment: dict):
        """
        Liefert Adjustierung an wartenden Subagent.
        """
        queue = self._pending_adjustments.get(task_id)
        if queue:
            queue.put_nowait(adjustment)
        else:
            logger.warning("No pending adjustment queue for subagent [{}]", task_id)

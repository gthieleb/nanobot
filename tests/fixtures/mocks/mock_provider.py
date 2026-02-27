"""Mock LLM provider for testing."""

from unittest.mock import AsyncMock
from nanobot.providers.base import LLMResponse, ToolCallRequest


class MockLLMProvider:
    """Mock provider that returns predefined responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.chat_method = AsyncMock()

    async def chat(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
        """Return predefined response or generate based on tools."""
        self.call_count += 1

        if tools and len(tools) > 0:
            # Simulate tool call request
            return LLMResponse(
                content="I'll use a tool",
                tool_calls=[
                    ToolCallRequest(
                        id=f"test_{self.call_count}", name="test_tool", arguments={"test": "value"}
                    )
                ],
            )
        else:
            # Simulate normal response
            return LLMResponse(content=f"Mock response #{self.call_count}")

    def get_default_model(self):
        return "mock-model"

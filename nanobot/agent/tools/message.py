"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError, field_validator

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class InlineButton(BaseModel):
    """Single inline button with validation."""
    label: str
    callback_data: str

    @field_validator('label')
    @classmethod
    def label_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Button label cannot be empty')
        return v.strip()

    @field_validator('callback_data')
    @classmethod
    def callback_data_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('callback_data cannot be empty')
        return v.strip()


class ButtonRow(BaseModel):
    """A row of inline buttons."""
    root: list[InlineButton]

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                },
                "buttons": {
                    "type": "array",
                    "description": "Optional: inline keyboard buttons for Telegram. List of rows, each row is list of {label, callback_data} objects.",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Button text"},
                                "callback_data": {"type": "string", "description": "Data sent when button is clicked"}
                            },
                            "required": ["label", "callback_data"]
                        }
                    }
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        buttons: list[list[dict[str, str]]] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        # Validate buttons with Pydantic
        buttons_tuples: list[list[tuple[str, str]]] = []
        if buttons:
            try:
                for row_idx, row in enumerate(buttons):
                    validated_row = []
                    for btn_idx, btn in enumerate(row):
                        try:
                            validated_btn = InlineButton(**btn)
                            validated_row.append((validated_btn.label, validated_btn.callback_data))
                        except ValidationError as e:
                            return f"Button validation error (row {row_idx}, button {btn_idx}): {e}"
                    buttons_tuples.append(validated_row)
            except Exception as e:
                return f"Button structure error: {str(e)}"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            },
            buttons=buttons_tuples,
        )

        try:
            await self._send_callback(msg)
            if channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            buttons_info = f" with {sum(len(r) for r in buttons_tuples)} buttons" if buttons_tuples else ""
            return f"Message sent to {channel}:{chat_id}{media_info}{buttons_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"

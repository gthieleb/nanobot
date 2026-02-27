"""Unit tests for Telegram keyboard functionality."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from nanobot.config.schema import TelegramConfig
from nanobot.channels.telegram import TelegramChannel
from nanobot.bus.events import OutboundMessage


class TestBuildKeyboard:
    """Tests for _build_keyboard method."""
    
    def test_no_buttons_returns_none(self):
        """Empty buttons should return None."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=False)
        channel = TelegramChannel(config, MagicMock())
        
        result = channel._build_keyboard([])
        
        assert result is None
    
    def test_no_buttons_none_input(self):
        """None input should return None."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=False)
        channel = TelegramChannel(config, MagicMock())
        
        result = channel._build_keyboard(None)
        
        assert result is None
    
    def test_disabled_returns_none(self):
        """Should return None when keyboard_buttons_enabled=False."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=False)
        channel = TelegramChannel(config, MagicMock())
        
        # buttons format: [[(label, callback_data), ...], ...]
        buttons = [
            [("Yes", "yes"), ("No", "no")],
            [("Maybe", "maybe")],
        ]
        
        result = channel._build_keyboard(buttons)
        
        # Should be None when disabled
        assert result is None
    
    def test_inline_keyboard_when_enabled(self):
        """Should build InlineKeyboardMarkup when keyboard_buttons_enabled=True."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=True)
        channel = TelegramChannel(config, MagicMock())
        
        buttons = [
            [("Yes", "yes"), ("No", "no")],
            [("Maybe", "maybe")],
        ]
        
        result = channel._build_keyboard(buttons)
        
        # Should be InlineKeyboardMarkup
        assert result is not None
        assert hasattr(result, 'inline_keyboard')
    
    def test_inline_keyboard_button_callback_data_is_label(self):
        """Inline keyboard buttons should use label as callback_data."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=True)
        channel = TelegramChannel(config, MagicMock())
        
        buttons = [[("My Button", "ignored_callback")]]
        
        result = channel._build_keyboard(buttons)
        
        # Check that callback_data equals the label
        button = result.inline_keyboard[0][0]
        assert button.callback_data == "My Button"
    
    def test_single_row_buttons(self):
        """Should handle single row of buttons."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=True)
        channel = TelegramChannel(config, MagicMock())
        
        buttons = [[("A", "a"), ("B", "b"), ("C", "c")]]
        
        result = channel._build_keyboard(buttons)
        
        assert len(result.inline_keyboard) == 1
        assert len(result.inline_keyboard[0]) == 3
    
    def test_multiple_rows_buttons(self):
        """Should handle multiple rows of buttons."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=True)
        channel = TelegramChannel(config, MagicMock())
        
        buttons = [
            [("Row1_A", "r1a"), ("Row1_B", "r1b")],
            [("Row2_A", "r2a")],
            [("Row3_A", "r3a"), ("Row3_B", "r3b"), ("Row3_C", "r3c")],
        ]
        
        result = channel._build_keyboard(buttons)
        
        assert len(result.inline_keyboard) == 3
        assert len(result.inline_keyboard[0]) == 2
        assert len(result.inline_keyboard[1]) == 1
        assert len(result.inline_keyboard[2]) == 3


class TestConfigKey:
    """Tests for config key naming."""
    
    def test_config_default_false(self):
        """Default should be False (disabled)."""
        config = TelegramConfig(token="test")
        
        assert config.keyboard_buttons_enabled is False
    
    def test_config_can_enable(self):
        """Should be able to enable via config."""
        config = TelegramConfig(token="test", keyboard_buttons_enabled=True)
        
        assert config.keyboard_buttons_enabled is True


class TestReplyTo:
    """Tests for reply_to parameter in message sending."""
    
    @pytest.mark.asyncio
    async def test_send_uses_explicit_reply_to(self):
        """Should use explicit reply_to parameter for ReplyParameters."""
        config = TelegramConfig(token="test", reply_to_message=False)
        channel = TelegramChannel(config, MagicMock())
        
        # Mock the bot
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=123))
        channel._app = MagicMock()
        channel._app.bot = mock_bot
        
        # Message with explicit reply_to
        msg = OutboundMessage(
            channel="telegram",
            chat_id="12345",
            content="Test reply",
            reply_to="999"
        )
        
        result = await channel.send(msg)
        
        # Should have called send_message with reply_parameters
        call_args = mock_bot.send_message.call_args
        assert call_args is not None
        reply_params = call_args.kwargs.get('reply_parameters')
        assert reply_params is not None
        assert reply_params.message_id == 999
    
    @pytest.mark.asyncio
    async def test_send_returns_message_id(self):
        """Should return the sent message_id."""
        config = TelegramConfig(token="test")
        channel = TelegramChannel(config, MagicMock())
        
        # Mock the bot
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=456))
        channel._app = MagicMock()
        channel._app.bot = mock_bot
        
        msg = OutboundMessage(
            channel="telegram",
            chat_id="12345",
            content="Test message"
        )
        
        result = await channel.send(msg)
        
        assert result == "456"

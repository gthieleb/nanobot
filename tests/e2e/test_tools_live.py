"""Tool execution tests with real LLM calls.

Tests file operations through the agent loop.

Run with:
    NANOBOT_TEST_API_KEY=sk-... pytest tests/e2e/test_tools_live.py -m live -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nanobot.bus.events import InboundMessage


pytestmark = pytest.mark.live


class TestFileReadTool:
    """Test file read operations through gateway."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_read_file(self, live_gateway, workspace):
        """Test reading a file through the agent."""
        bus = live_gateway["bus"]

        # Create test file
        test_file = workspace / "test_read.txt"
        test_file.write_text("Hello from the test file!")

        await bus.publish_inbound(
            InboundMessage(
                channel="test",
                sender_id="user1",
                chat_id="chat1",
                content="Read the file test_read.txt and tell me what it says",
            )
        )

        response = await asyncio.wait_for(bus.consume_outbound(), timeout=60)

        assert "Hello from the test file" in response.content

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_read_nonexistent_file(self, live_gateway, send_and_wait):
        """Test handling of non-existent file."""
        bus = live_gateway["bus"]

        response = await send_and_wait(
            bus, "Try to read a file called nonexistent_file_12345.txt and report what happens"
        )

        # Should handle error gracefully
        content_lower = response.content.lower()
        assert any(
            word in content_lower
            for word in ["not found", "does not exist", "error", "cannot", "unable"]
        )


class TestFileWriteTool:
    """Test file write operations through gateway."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_write_file(self, live_gateway, workspace, send_and_wait):
        """Test writing a file through the agent."""
        bus = live_gateway["bus"]

        await send_and_wait(
            bus, "Create a file called test_write.txt with the content 'Written by agent'"
        )

        # Verify file was created
        written_file = workspace / "test_write.txt"
        assert written_file.exists(), f"File not created in {workspace}"
        assert "Written by agent" in written_file.read_text()

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_write_nested_path(self, live_gateway, workspace, send_and_wait):
        """Test writing to nested directory."""
        bus = live_gateway["bus"]

        await send_and_wait(
            bus, "Create a file at nested/deep/path.txt with content 'nested content'"
        )

        nested_file = workspace / "nested" / "deep" / "path.txt"
        assert nested_file.exists()
        assert "nested content" in nested_file.read_text()


class TestFileEditTool:
    """Test file edit operations through gateway."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_edit_file(self, live_gateway, workspace, send_and_wait):
        """Test editing a file through the agent."""
        bus = live_gateway["bus"]

        # Create initial file
        test_file = workspace / "test_edit.txt"
        test_file.write_text("Original content here")

        await send_and_wait(
            bus, "Edit the file test_edit.txt and replace 'Original' with 'Modified'"
        )

        # Verify edit
        content = test_file.read_text()
        assert "Modified" in content
        assert "Original" not in content

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_edit_preserves_rest(self, live_gateway, workspace, send_and_wait):
        """Test that edit preserves content outside the edit."""
        bus = live_gateway["bus"]

        # Create file with multiple parts
        test_file = workspace / "test_edit_multi.txt"
        test_file.write_text("Keep this. REPLACE_ME. Keep this too.")

        await send_and_wait(bus, "Edit test_edit_multi.txt: replace REPLACE_ME with DONE")

        content = test_file.read_text()
        assert "Keep this" in content
        assert "Keep this too" in content
        assert "DONE" in content
        assert "REPLACE_ME" not in content


class TestListDirTool:
    """Test directory listing through gateway."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_list_directory(self, live_gateway, workspace, send_and_wait):
        """Test listing directory contents."""
        bus = live_gateway["bus"]

        # Create some files
        (workspace / "file1.txt").write_text("content1")
        (workspace / "file2.txt").write_text("content2")
        (workspace / "subdir").mkdir()

        response = await send_and_wait(
            bus, "List the files in the current directory and tell me what you see"
        )

        content_lower = response.content.lower()
        assert "file1" in content_lower or "file" in content_lower
        assert "file2" in content_lower or "file" in content_lower


class TestToolChaining:
    """Test multiple tool operations in sequence."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_read_write_chain(self, live_gateway, workspace, send_and_wait):
        """Test reading one file and writing to another."""
        bus = live_gateway["bus"]

        # Create source file
        source = workspace / "source.txt"
        source.write_text("Source content to copy")

        await send_and_wait(bus, "Read source.txt and write its content to destination.txt")

        dest = workspace / "destination.txt"
        assert dest.exists()
        assert "Source content to copy" in dest.read_text()

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_multi_step_task(self, live_gateway, workspace, send_and_wait):
        """Test a multi-step task with multiple tool calls."""
        bus = live_gateway["bus"]

        response = await send_and_wait(
            bus,
            "Create a file called todo.txt with the content 'Task 1: Done', then read it back and confirm",
        )

        # Verify file exists
        todo_file = workspace / "todo.txt"
        assert todo_file.exists()
        assert "Task 1: Done" in todo_file.read_text()

        # Response should confirm
        assert "done" in response.content.lower() or "created" in response.content.lower()

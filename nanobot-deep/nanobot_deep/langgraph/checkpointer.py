"""SQLite-based session checkpointer for LangGraph integration.

Wraps LangGraph's SqliteSaver with:
1. Automatic migration from nanobot's JSON sessions
2. Helper methods for session management
"""

from __future__ import annotations

import json
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger


class SessionCheckpointer(SqliteSaver):
    """SQLite-based checkpointer with nanobot session migration.

    Extends LangGraph's SqliteSaver with:
    - Automatic migration from JSON session files
    - Helper methods for session management

    Args:
        db_path: Path to SQLite database file (default: ~/.nanobot/sessions.db)
        migrate_from_json: Whether to auto-migrate from JSON sessions

    Example:
        >>> checkpointer = SessionCheckpointer()
        >>> config = {"configurable": {"thread_id": "telegram:12345"}}
        >>> result = await agent.ainvoke(state, config)
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        migrate_from_json: bool = True,
    ):
        if db_path is None:
            db_path = Path.home() / ".nanobot" / "sessions.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        import sqlite3

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        super().__init__(conn)

        self._migrate_from_json = migrate_from_json
        self._migrated = False

    def setup(self) -> None:
        """Set up the database and optionally migrate sessions."""
        super().setup()
        if self._migrate_from_json and not self._migrated:
            self._migrate_json_sessions()
            self._migrated = True

    def _migrate_json_sessions(self) -> None:
        """Migrate existing JSON session files to SQLite."""
        sessions_dir = Path.home() / ".nanobot" / "sessions"
        if not sessions_dir.exists():
            return

        migrated = 0
        with self.cursor() as cur:
            for session_file in sessions_dir.glob("*.json"):
                try:
                    with open(session_file) as f:
                        data = json.load(f)

                    thread_id = session_file.stem
                    messages = data.get("messages", [])

                    if not messages:
                        continue

                    checkpoint_id = f"migrated-{datetime.now().isoformat()}"
                    checkpoint: Checkpoint = {
                        "id": checkpoint_id,
                        "ts": datetime.now().isoformat(),
                        "channel_values": {"messages": messages},
                        "channel_versions": {"messages": "1"},
                        "versions_seen": {},
                        "pending_sends": [],
                    }

                    metadata: CheckpointMetadata = {
                        "source": "migrated_from_json",
                        "step": -1,
                        "writes": {},
                        "parents": {},
                    }

                    type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
                    serialized_metadata = json.dumps(metadata, ensure_ascii=False).encode("utf-8")

                    cur.execute(
                        """
                        INSERT OR REPLACE INTO checkpoints 
                        (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata)
                        VALUES (?, '', ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            checkpoint_id,
                            type_,
                            serialized_checkpoint,
                            serialized_metadata,
                        ),
                    )
                    migrated += 1

                except Exception as e:
                    logger.warning(f"Failed to migrate session {session_file}: {e}")

        if migrated > 0:
            logger.info(f"Migrated {migrated} JSON sessions to SQLite")

    def delete_session(self, thread_id: str) -> bool:
        """Delete all checkpoints for a session.

        Args:
            thread_id: The session/thread identifier

        Returns:
            True if any data was deleted
        """
        with self.cursor() as cur:
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            return cur.rowcount > 0

    def list_sessions(self) -> list[str]:
        """List all session/thread IDs."""
        with self.cursor(transaction=False) as cur:
            cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
            return [row[0] for row in cur.fetchall()]

    def get_session_history(self, thread_id: str, limit: int = 100) -> list[dict]:
        """Get message history for a session.

        Args:
            thread_id: The session identifier
            limit: Maximum messages to return

        Returns:
            List of message dicts in nanobot format
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = self.get_tuple(config)

        if not checkpoint_tuple:
            return []

        checkpoint = checkpoint_tuple.checkpoint
        messages = checkpoint.get("channel_values", {}).get("messages", [])

        history = []
        for msg in messages[-limit:]:
            msg_dict: dict[str, Any] = {}

            if hasattr(msg, "content"):
                if hasattr(msg, "type"):
                    if msg.type == "human":
                        msg_dict["role"] = "user"
                    elif msg.type == "ai":
                        msg_dict["role"] = "assistant"
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            msg_dict["tool_calls"] = [
                                {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", ""),
                                        "arguments": json.dumps(tc.get("args", {})),
                                    },
                                }
                                for tc in msg.tool_calls
                            ]
                    elif msg.type == "tool":
                        msg_dict["role"] = "tool"
                        msg_dict["tool_call_id"] = getattr(msg, "tool_call_id", "")
                    elif msg.type == "system":
                        msg_dict["role"] = "system"

                msg_dict["content"] = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )

            if msg_dict.get("role"):
                history.append(msg_dict)

        return history

"""Small persistence boundary for conversations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4


class ConversationStore:
    def __init__(self, path: str = "data/federalscope.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    response_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_conversation(self, title: str = "New research") -> dict[str, Any]:
        conversation_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conversation_id, title),
            )
        return {"id": conversation_id, "title": title, "messages": []}

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not conversation:
                return None
            messages = connection.execute(
                "SELECT id, role, content, response_json, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
        return {
            **dict(conversation),
            "messages": [
                {
                    **dict(message),
                    "response": json.loads(message["response_json"])
                    if message["response_json"]
                    else None,
                }
                for message in messages
            ],
        }

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        response: dict[str, Any] | None = None,
    ) -> str:
        message_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages (id, conversation_id, role, content, response_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, json.dumps(response) if response else None),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,),
            )
        return message_id

"""
memory_store.py — Pure SQLite session/memory logic. No FastAPI or Streamlit imports.
"""

import sqlite3
import uuid
from datetime import datetime, timezone


class MemoryStore:
    """Manages chat sessions and message history in a SQLite database."""

    def __init__(self, db_path: str = "sessions.sqlite3"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new connection (SQLite connections are not thread-safe,
        so FastAPI handlers each get their own)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def create_session(self) -> str:
        """Create a new session and return a short readable ID
        (first 8 chars of a uuid4)."""
        session_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
                (session_id, now),
            )
            conn.commit()
        finally:
            conn.close()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_history(self, session_id: str, limit: int = 50) -> list[tuple[str, str]]:
        """Return message history oldest-first as list of (role, content) tuples.
        Returns the most recent *limit* messages, ordered chronologically."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, id FROM messages "
                "  WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                ") sub ORDER BY id ASC",
                (session_id, limit),
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        finally:
            conn.close()

    def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session but keep the session alive."""
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
        finally:
            conn.close()


def format_history_text(history: list[tuple[str, str]], max_turns: int = 4) -> str:
    """Format the last *max_turns* exchanges into a plain-text transcript for
    prompting, or return ``'(no previous turns)'`` if empty."""
    if not history:
        return "(no previous turns)"

    # Each "turn" is a user+assistant pair.  We take the last max_turns*2
    # individual messages to capture up to max_turns exchanges.
    recent = history[-(max_turns * 2):]
    lines = []
    for role, content in recent:
        label = "Customer" if role == "user" else "Agent"
        lines.append(f"{label}: {content}")

    return "\n".join(lines) if lines else "(no previous turns)"

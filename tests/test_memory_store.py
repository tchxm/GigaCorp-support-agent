"""
tests/test_memory_store.py — Tests for SQLite session/memory persistence.
"""

import os
import tempfile
import pytest
from memory_store import MemoryStore, format_history_text


@pytest.fixture
def db_path(tmp_path):
    """Provide a unique temporary database path."""
    return str(tmp_path / "test_sessions.sqlite3")


@pytest.fixture
def store(db_path):
    """Provide a fresh MemoryStore."""
    return MemoryStore(db_path)


class TestSessionCreation:
    def test_create_session_returns_string(self, store):
        sid = store.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 8

    def test_session_ids_are_unique(self, store):
        ids = {store.create_session() for _ in range(20)}
        assert len(ids) == 20

    def test_session_exists_true(self, store):
        sid = store.create_session()
        assert store.session_exists(sid) is True

    def test_session_exists_false(self, store):
        assert store.session_exists("nonexistent") is False


class TestHistory:
    def test_empty_history(self, store):
        sid = store.create_session()
        history = store.get_history(sid)
        assert history == []

    def test_append_and_retrieve(self, store):
        sid = store.create_session()
        store.append_message(sid, "user", "Hello")
        store.append_message(sid, "assistant", "Hi there!")
        history = store.get_history(sid)
        assert len(history) == 2
        assert history[0] == ("user", "Hello")
        assert history[1] == ("assistant", "Hi there!")

    def test_history_ordering_oldest_first(self, store):
        sid = store.create_session()
        for i in range(5):
            store.append_message(sid, "user", f"msg-{i}")
        history = store.get_history(sid)
        contents = [c for _, c in history]
        assert contents == [f"msg-{i}" for i in range(5)]

    def test_history_limit(self, store):
        sid = store.create_session()
        for i in range(10):
            store.append_message(sid, "user", f"msg-{i}")
        history = store.get_history(sid, limit=3)
        assert len(history) == 3
        # Should be the MOST RECENT 3, ordered oldest-first
        contents = [c for _, c in history]
        assert contents == ["msg-7", "msg-8", "msg-9"]

    def test_sessions_are_isolated(self, store):
        sid1 = store.create_session()
        sid2 = store.create_session()
        store.append_message(sid1, "user", "for session 1")
        store.append_message(sid2, "user", "for session 2")
        h1 = store.get_history(sid1)
        h2 = store.get_history(sid2)
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0][1] == "for session 1"
        assert h2[0][1] == "for session 2"


class TestPersistenceAcrossInstances:
    """Simulates an app restart by creating a new MemoryStore
    over the same database file."""

    def test_data_survives_new_instance(self, db_path):
        store1 = MemoryStore(db_path)
        sid = store1.create_session()
        store1.append_message(sid, "user", "persist this")
        store1.append_message(sid, "assistant", "persisted!")

        # Create a brand new instance on the same file
        store2 = MemoryStore(db_path)
        assert store2.session_exists(sid) is True
        history = store2.get_history(sid)
        assert len(history) == 2
        assert history[0] == ("user", "persist this")
        assert history[1] == ("assistant", "persisted!")


class TestClearSession:
    def test_clear_removes_messages(self, store):
        sid = store.create_session()
        store.append_message(sid, "user", "hello")
        store.append_message(sid, "assistant", "hi")
        store.clear_session(sid)
        assert store.get_history(sid) == []

    def test_session_still_exists_after_clear(self, store):
        sid = store.create_session()
        store.append_message(sid, "user", "hello")
        store.clear_session(sid)
        assert store.session_exists(sid) is True


class TestFormatHistoryText:
    def test_empty_history(self):
        result = format_history_text([])
        assert result == "(no previous turns)"

    def test_single_turn(self):
        history = [("user", "What are your hours?"), ("assistant", "We're open 8-8.")]
        result = format_history_text(history)
        assert "Customer: What are your hours?" in result
        assert "Agent: We're open 8-8." in result

    def test_truncation(self):
        history = []
        for i in range(10):
            history.append(("user", f"q{i}"))
            history.append(("assistant", f"a{i}"))
        result = format_history_text(history, max_turns=2)
        # Should only contain the last 2 turns (4 messages)
        assert "q8" in result
        assert "a8" in result
        assert "q9" in result
        assert "a9" in result
        assert "q0" not in result
        assert "q7" not in result

"""
tests/test_api.py — Integration tests for FastAPI endpoints.

Uses FastAPI's TestClient with fake LLM/embeddings injected via dependency
overrides — never hits real Anthropic/HuggingFace APIs.

The grounding threshold is also overridable so we can deterministically test
both grounded and ungrounded paths despite FakeEmbeddings producing
non-semantic hash-based vectors.
"""

import os
import uuid
import pytest
from fastapi.testclient import TestClient

from rag_core import build_index, REFUSAL_MESSAGE
from tests.fakes import FakeEmbeddings, FakeLLM


FAQ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gigacorp_faq.txt")


@pytest.fixture
def test_app(tmp_path):
    """Create a FastAPI test app with fake dependencies injected.

    Yields (client, fake_llm_holder, threshold_holder).
    - Set fake_llm_holder[0] = FakeLLM([...]) before each request.
    - Set threshold_holder[0] to control grounding (0.0 = always ground, 1.0 = never).
    """
    persist_dir = str(tmp_path / f"chroma_{uuid.uuid4().hex[:8]}")
    embeddings = FakeEmbeddings()
    vectordb = build_index(FAQ_PATH, persist_dir, embeddings)

    import main
    from memory_store import MemoryStore

    db_path = str(tmp_path / "test_sessions.sqlite3")
    memory = MemoryStore(db_path)

    fake_llm_holder = [None]
    threshold_holder = [0.0]  # Default to always-grounded for most tests

    def get_vectordb_override():
        return vectordb

    def get_llm_override():
        return fake_llm_holder[0]

    def get_memory_override():
        return memory

    def get_threshold_override():
        return threshold_holder[0]

    main.app.dependency_overrides[main.get_vectordb] = get_vectordb_override
    main.app.dependency_overrides[main.get_llm] = get_llm_override
    main.app.dependency_overrides[main.get_memory] = get_memory_override
    main.app.dependency_overrides[main.get_threshold] = get_threshold_override

    client = TestClient(main.app, raise_server_exceptions=False)

    yield client, fake_llm_holder, threshold_holder

    main.app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_ok(self, test_app):
        client, _, _ = test_app
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["vector_store_loaded"] is True


class TestSessionEndpoints:
    def test_create_session(self, test_app):
        client, _, _ = test_app
        res = client.post("/sessions")
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 8

    def test_get_history_empty(self, test_app):
        client, _, _ = test_app
        sid = client.post("/sessions").json()["session_id"]
        res = client.get(f"/sessions/{sid}/history")
        assert res.status_code == 200
        assert res.json() == []

    def test_unknown_session_404(self, test_app):
        client, _, _ = test_app
        res = client.get("/sessions/nonexistent/history")
        assert res.status_code == 404

    def test_unknown_session_post_404(self, test_app):
        client, _, _ = test_app
        res = client.post(
            "/sessions/nonexistent/messages",
            json={"message": "hello"},
        )
        assert res.status_code == 404


class TestMessageEndpoint:
    def test_grounded_answer_with_sources(self, test_app):
        """Post a relevant question and get a grounded answer with sources.

        threshold=0.0 (default in fixture) so FakeEmbeddings always produce
        a grounded result.
        """
        client, llm_holder, _ = test_app

        sid = client.post("/sessions").json()["session_id"]

        # No history → condense_question skips LLM → only generate_answer calls LLM
        llm_holder[0] = FakeLLM(["We offer three shipping tiers for domestic orders."])

        res = client.post(
            f"/sessions/{sid}/messages",
            json={"message": "What is your shipping policy?"},
        )
        assert res.status_code == 200
        data = res.json()

        assert "answer" in data
        assert data["grounded"] is True
        assert "standalone_question" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

        # Verify source structure
        for src in data["sources"]:
            assert "source" in src
            assert "section" in src
            assert "start_line" in src
            assert "end_line" in src
            assert "similarity" in src
            assert "evidence" in src
            assert src["evidence"]

    def test_refusal_for_off_topic(self, test_app):
        """With threshold=1.0, any query is ungrounded → fixed refusal."""
        client, llm_holder, threshold_holder = test_app

        sid = client.post("/sessions").json()["session_id"]

        # Set threshold to 1.0 so nothing is grounded
        threshold_holder[0] = 1.0

        # Ungrounded: condense skips (no history), generate skips → 0 LLM calls
        llm_holder[0] = FakeLLM([])

        res = client.post(
            f"/sessions/{sid}/messages",
            json={"message": "What is the meaning of life?"},
        )
        assert res.status_code == 200
        data = res.json()

        assert data["grounded"] is False
        assert data["answer"] == REFUSAL_MESSAGE

    def test_history_persists_across_requests(self, test_app):
        """Two sequential requests to the same session should both be in history."""
        client, llm_holder, _ = test_app

        sid = client.post("/sessions").json()["session_id"]

        # First message — no history, condense skips, 1 LLM call for generate
        llm_holder[0] = FakeLLM(["Answer to first question."])
        client.post(
            f"/sessions/{sid}/messages",
            json={"message": "First question about shipping"},
        )

        # Second message — now there IS history.
        # condense_question will call LLM (1 call) + generate_answer (1 call) = 2 calls
        llm_holder[0] = FakeLLM([
            "What are the express shipping rates?",  # condensed
            "Express shipping is $12.99.",            # answer
        ])
        client.post(
            f"/sessions/{sid}/messages",
            json={"message": "How much for express?"},
        )

        # Now check history
        res = client.get(f"/sessions/{sid}/history")
        assert res.status_code == 200
        history = res.json()

        # Should have 4 messages: user, assistant, user, assistant
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[3]["role"] == "assistant"

    def test_clear_messages(self, test_app):
        """DELETE /sessions/{id}/messages clears history but session remains."""
        client, llm_holder, _ = test_app

        sid = client.post("/sessions").json()["session_id"]

        llm_holder[0] = FakeLLM(["Some answer."])
        client.post(
            f"/sessions/{sid}/messages",
            json={"message": "Hello"},
        )

        # Clear
        res = client.delete(f"/sessions/{sid}/messages")
        assert res.status_code == 200

        # History should be empty
        res = client.get(f"/sessions/{sid}/history")
        assert res.status_code == 200
        assert res.json() == []

        # Session still exists (can post again)
        llm_holder[0] = FakeLLM(["Another answer."])
        res = client.post(
            f"/sessions/{sid}/messages",
            json={"message": "New message after clear"},
        )
        assert res.status_code == 200

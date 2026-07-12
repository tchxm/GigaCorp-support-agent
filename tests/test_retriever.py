"""
tests/test_retriever.py — Tests for vector store, retrieval, citation, grounding,
and the full answer_query pipeline.

Uses FakeEmbeddings and FakeLLM so nothing hits the network.
"""

import os
import shutil
import tempfile
import uuid
import pytest
from langchain_core.documents import Document

from rag_core import (
    build_index,
    retrieve_with_scores,
    is_grounded,
    format_citations,
    small_talk_response,
    condense_question,
    generate_answer,
    answer_query,
    parse_sections,
    sections_to_documents,
    REFUSAL_MESSAGE,
    GREETING_MESSAGE,
)
from tests.fakes import FakeEmbeddings, FakeLLM


FAQ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gigacorp_faq.txt")


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def persist_dir(tmp_path):
    """Unique persist directory per test — avoids Chroma connection caching issues."""
    d = tmp_path / f"chroma_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return str(d)


@pytest.fixture
def vectordb(embeddings, persist_dir):
    """Build a Chroma index from the real FAQ using FakeEmbeddings."""
    return build_index(FAQ_PATH, persist_dir, embeddings)


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_builds_from_scratch(self, embeddings, persist_dir):
        db = build_index(FAQ_PATH, persist_dir, embeddings)
        # Should have loaded/created documents
        assert db is not None

    def test_idempotent_rebuild(self, embeddings, persist_dir):
        """Building twice on the same persist_dir should not error or duplicate docs."""
        db1 = build_index(FAQ_PATH, persist_dir, embeddings)
        count1 = len(db1.get()["ids"])
        db2 = build_index(FAQ_PATH, persist_dir, embeddings)
        count2 = len(db2.get()["ids"])
        assert count1 == count2

    def test_index_contains_expected_sections(self, vectordb):
        """All FAQ sections should be in the index."""
        data = vectordb.get()
        sections_found = set()
        for meta in data["metadatas"]:
            sections_found.add(meta["section"])
        assert "SHIPPING POLICY" in sections_found
        assert "RETURN AND REFUND PROCESS" in sections_found
        assert "BUSINESS HOURS" in sections_found
        assert "SERVICE TIERS" in sections_found


# ---------------------------------------------------------------------------
# retrieve_with_scores
# ---------------------------------------------------------------------------

class TestRetrieveWithScores:
    def test_returns_k_results(self, vectordb):
        results = retrieve_with_scores(vectordb, "shipping", k=3)
        assert len(results) == 3

    def test_scores_between_zero_and_one(self, vectordb):
        results = retrieve_with_scores(vectordb, "return policy", k=3)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_results_are_documents_with_metadata(self, vectordb):
        results = retrieve_with_scores(vectordb, "hours", k=2)
        for doc, _ in results:
            assert isinstance(doc, Document)
            assert "source" in doc.metadata
            assert "section" in doc.metadata
            assert "start_line" in doc.metadata
            assert "end_line" in doc.metadata


# ---------------------------------------------------------------------------
# is_grounded
# ---------------------------------------------------------------------------

class TestIsGrounded:
    def test_grounded_above_threshold(self):
        docs = [
            (Document(page_content="x"), 0.5),
            (Document(page_content="y"), 0.3),
        ]
        assert is_grounded(docs, threshold=0.35) is True

    def test_ungrounded_below_threshold(self):
        docs = [
            (Document(page_content="x"), 0.2),
            (Document(page_content="y"), 0.1),
        ]
        assert is_grounded(docs, threshold=0.35) is False

    def test_ungrounded_empty(self):
        assert is_grounded([], threshold=0.35) is False

    def test_boundary_at_threshold(self):
        docs = [(Document(page_content="x"), 0.35)]
        assert is_grounded(docs, threshold=0.35) is True

    def test_boundary_just_below_threshold(self):
        docs = [(Document(page_content="x"), 0.349)]
        assert is_grounded(docs, threshold=0.35) is False


# ---------------------------------------------------------------------------
# format_citations
# ---------------------------------------------------------------------------

class TestFormatCitations:
    def _make_doc(self, source, section, start, end):
        return Document(
            page_content="dummy",
            metadata={
                "source": source,
                "section": section,
                "start_line": start,
                "end_line": end,
            },
        )

    def test_basic_formatting(self):
        scored = [(self._make_doc("faq.txt", "SHIPPING", 1, 10), 0.8)]
        result = format_citations(scored)
        assert "Sources:" in result
        assert "faq.txt" in result
        assert "SHIPPING" in result
        assert "lines 1-10" in result
        assert "0.80" in result

    def test_deduplication(self):
        """Same (source, start_line, end_line) should appear only once."""
        doc = self._make_doc("faq.txt", "SHIPPING", 1, 10)
        scored = [(doc, 0.8), (doc, 0.7)]
        result = format_citations(scored)
        # Count lines starting with '-'
        citation_lines = [l for l in result.split("\n") if l.startswith("-")]
        assert len(citation_lines) == 1

    def test_multiple_sources(self):
        scored = [
            (self._make_doc("faq.txt", "SHIPPING", 1, 10), 0.8),
            (self._make_doc("faq.txt", "RETURNS", 11, 20), 0.6),
        ]
        result = format_citations(scored)
        citation_lines = [l for l in result.split("\n") if l.startswith("-")]
        assert len(citation_lines) == 2

    def test_empty_docs(self):
        result = format_citations([])
        assert result == ""


# ---------------------------------------------------------------------------
# condense_question
# ---------------------------------------------------------------------------

class TestCondenseQuestion:
    def test_no_history_returns_question_unchanged(self):
        llm = FakeLLM(["should not be called"])
        result = condense_question(llm, "(no previous turns)", "What is shipping?")
        assert result == "What is shipping?"
        # Verify LLM was NOT called (index should still be 0)
        assert llm._call_index == 0

    def test_empty_history_returns_question_unchanged(self):
        llm = FakeLLM(["should not be called"])
        result = condense_question(llm, "", "What is shipping?")
        assert result == "What is shipping?"
        assert llm._call_index == 0

    def test_with_history_calls_llm(self):
        llm = FakeLLM(["What is the shipping cost to Canada?"])
        result = condense_question(
            llm,
            "Customer: How do you ship?\nAgent: We use various carriers.",
            "How much to ship there?",
        )
        assert result == "What is the shipping cost to Canada?"
        assert llm._call_index == 1


# ---------------------------------------------------------------------------
# small_talk_response
# ---------------------------------------------------------------------------

class TestSmallTalkResponse:
    def test_greeting_returns_canned_response(self):
        assert small_talk_response("hey") == GREETING_MESSAGE
        assert small_talk_response("hello there") == GREETING_MESSAGE

    def test_real_question_does_not_match(self):
        assert small_talk_response("hi, what is your return policy?") is None
        assert small_talk_response("what are your business hours?") is None


# ---------------------------------------------------------------------------
# answer_query — full pipeline
# ---------------------------------------------------------------------------

class TestAnswerQuery:
    def test_greeting_skips_retrieval_and_sources(self, vectordb):
        """Greetings should not get irrelevant FAQ citations."""
        llm = FakeLLM([])
        result = answer_query(
            llm, vectordb, "(no previous turns)", "hey", threshold=0.0,
        )

        assert result["grounded"] is False
        assert result["answer"] == GREETING_MESSAGE
        assert result["scored_docs"] == []
        assert llm._call_index == 0

    def test_grounded_answer(self, vectordb):
        """With a relevant question, the pipeline should return a grounded answer.

        When history is empty condense_question skips the LLM call, so only ONE
        fake response is needed (for generate_answer).

        threshold=0.0 because FakeEmbeddings produce non-semantic hash-based
        vectors (~0.004 similarity), so we disable grounding to test pipeline.
        """
        llm = FakeLLM(["GigaCorp offers three shipping tiers."])
        result = answer_query(
            llm, vectordb, "(no previous turns)",
            "What is your shipping policy?", threshold=0.0,
        )

        assert result["grounded"] is True
        assert "GigaCorp offers three shipping tiers." in result["answer"]
        assert result["standalone_question"] == "What is your shipping policy?"
        assert len(result["scored_docs"]) > 0

    def test_ungrounded_answer(self, vectordb):
        """With threshold=1.0, ANY query is ungrounded → fixed refusal.

        When history is empty, condense_question skips the LLM call,
        and when ungrounded, generate_answer is also skipped.
        So ZERO LLM calls are needed.
        """
        llm = FakeLLM([])  # No responses needed
        result = answer_query(
            llm, vectordb, "(no previous turns)",
            "What is the meaning of life and the universe and everything?",
            k=3, threshold=1.0,
        )

        assert result["grounded"] is False
        assert result["answer"] == REFUSAL_MESSAGE
        assert "standalone_question" in result
        assert "scored_docs" in result
        assert result["scored_docs"] == []

    def test_answer_with_history(self, vectordb):
        """With conversation history, condense_question DOES call the LLM
        (1 call for condensing + 1 call for generating = 2 fake responses needed).

        threshold=0.0 so FakeEmbeddings always produce a grounded result.
        """
        llm = FakeLLM([
            "What are GigaCorp's shipping costs?",  # condensed question
            "Standard shipping is free over $50.",   # answer
        ])
        history_text = "Customer: Tell me about shipping.\nAgent: We have three tiers."
        result = answer_query(
            llm, vectordb, history_text, "How much does it cost?",
            threshold=0.0,
        )

        assert result["standalone_question"] == "What are GigaCorp's shipping costs?"
        assert "Standard shipping is free over $50." in result["answer"]

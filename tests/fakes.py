"""
tests/fakes.py — Fake LLM and Fake Embeddings for offline testing.

These allow the entire test suite to run with zero API keys and zero network calls.
"""

import hashlib
from typing import Any


class FakeEmbeddings:
    """Deterministic hash-based embeddings.  Produces the same vector
    for the same input text, no model download required.

    Implements the LangChain Embeddings interface (embed_documents / embed_query).
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic float vector via SHA-256 hash."""
        # Repeat the hash to fill the dimension
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Extend by hashing the hash repeatedly
        raw_bytes = h
        while len(raw_bytes) < self.dimension * 4:
            h = hashlib.sha256(h).digest()
            raw_bytes += h
        # Convert to floats in [-1, 1]
        vector = []
        for i in range(self.dimension):
            byte_val = raw_bytes[i]
            vector.append((byte_val / 127.5) - 1.0)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._hash_to_vector(text)


class _FakeMessage:
    """Minimal message object with a .content attribute,
    matching the interface of LangChain's AIMessage."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Constructed with a list of canned string responses.
    Each call to .invoke() returns the next response in order.

    Implements enough of the LangChain LLM interface to work with
    rag_core's condense_question and generate_answer functions.
    """

    def __init__(self, responses: list[str]):
        self.responses = list(responses)  # copy to avoid mutation
        self._call_index = 0

    def invoke(self, prompt: Any) -> _FakeMessage:
        """Return the next canned response as an object with a .content attribute."""
        if self._call_index >= len(self.responses):
            raise RuntimeError(
                f"FakeLLM exhausted: {len(self.responses)} responses provided "
                f"but call #{self._call_index + 1} was made. "
                f"Check your test setup — you may need more canned responses."
            )
        response = self.responses[self._call_index]
        self._call_index += 1
        return _FakeMessage(response)

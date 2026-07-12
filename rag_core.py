"""
rag_core.py — Pure RAG logic. No FastAPI or Streamlit imports.

All functions accept llm/embeddings as arguments — never constructed internally.
This makes the entire module testable without a network call.
"""

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A parsed section from the FAQ file."""
    header: str
    content: str
    start_line: int   # 1-indexed, inclusive
    end_line: int      # 1-indexed, inclusive


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_sections(path: str) -> list[Section]:
    """Split a FAQ file on ``## HEADER`` lines, tracking exact 1-indexed
    start/end line numbers per section."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections: list[Section] = []
    current_header: str | None = None
    current_lines: list[str] = []
    start_line: int = 0

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("## "):
            # Close previous section if any
            if current_header is not None:
                sections.append(Section(
                    header=current_header,
                    content="".join(current_lines).strip(),
                    start_line=start_line,
                    end_line=idx - 1,
                ))
            current_header = stripped[3:].strip()
            current_lines = []
            start_line = idx
        else:
            current_lines.append(line)

    # Close final section
    if current_header is not None:
        sections.append(Section(
            header=current_header,
            content="".join(current_lines).strip(),
            start_line=start_line,
            end_line=len(lines),
        ))

    return sections


def sections_to_documents(sections: list[Section], source_name: str) -> list[Document]:
    """Wrap each section as a LangChain Document with metadata."""
    docs = []
    for sec in sections:
        doc = Document(
            page_content=f"{sec.header}\n{sec.content}",
            metadata={
                "source": source_name,
                "section": sec.header,
                "start_line": sec.start_line,
                "end_line": sec.end_line,
            },
        )
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

COLLECTION_NAME = "gigacorp_faq"
MANIFEST_NAME = "index_manifest.json"


class LocalKeywordEmbeddings:
    """Small deterministic embeddings for the mock FAQ.

    This keeps the demo lightweight: Chroma still provides the local vector
    store, but the app does not need Torch or external embedding downloads.
    """

    model_name = "local-keyword-embeddings-v1"

    _canonical_terms = {
        "ship": "shipping",
        "ships": "shipping",
        "shipped": "shipping",
        "shipment": "shipping",
        "shipping": "shipping",
        "delivery": "shipping",
        "deliver": "shipping",
        "international": "international",
        "india": "international",
        "standard": "shipping_rate",
        "express": "shipping_rate",
        "overnight": "shipping_rate",
        "cost": "shipping_rate",
        "rate": "shipping_rate",
        "rates": "shipping_rate",
        "return": "returns",
        "returns": "returns",
        "refund": "returns",
        "refunds": "returns",
        "defective": "returns",
        "exchange": "returns",
        "hours": "hours",
        "business": "hours",
        "support": "hours",
        "sunday": "hours",
        "holiday": "hours",
        "tier": "tiers",
        "tiers": "tiers",
        "bronze": "tiers",
        "silver": "tiers",
        "gold": "tiers",
    }

    _domain_slots = {
        "shipping": 0,
        "international": 1,
        "shipping_rate": 2,
        "returns": 3,
        "hours": 4,
        "tiers": 5,
    }

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokens(text):
            canonical = self._canonical_terms.get(token)
            if canonical:
                vector[self._domain_slots[canonical]] += 2.0

            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = 8 + (int.from_bytes(digest[:4], "big") % (self.dimension - 8))
            vector[slot] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedding_signature(embeddings) -> str:
    model_name = getattr(embeddings, "model_name", None)
    return f"{type(embeddings).__module__}.{type(embeddings).__name__}:{model_name}"


def _index_manifest(faq_path: str, embeddings) -> dict:
    return {
        "source_name": os.path.basename(faq_path),
        "source_sha256": _file_sha256(faq_path),
        "embedding_signature": _embedding_signature(embeddings),
        "collection_name": COLLECTION_NAME,
    }


def _manifest_matches(persist_dir: str, expected: dict) -> bool:
    manifest_path = os.path.join(persist_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _write_manifest(persist_dir: str, manifest: dict) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    manifest_path = os.path.join(persist_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def build_index(faq_path: str, persist_dir: str, embeddings) -> "Chroma":
    """Load a matching persisted Chroma index or rebuild from the FAQ.

    The manifest prevents stale local vector stores from being reused after the
    FAQ file or embedding configuration changes.
    """
    from langchain_chroma import Chroma

    manifest = _index_manifest(faq_path, embeddings)

    if (
        os.path.isdir(persist_dir)
        and os.listdir(persist_dir)
        and _manifest_matches(persist_dir, manifest)
    ):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        shutil.rmtree(persist_dir)

    sections = parse_sections(faq_path)
    source_name = os.path.basename(faq_path)
    docs = sections_to_documents(sections, source_name)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )
    _write_manifest(persist_dir, manifest)
    return vectordb


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def distance_to_similarity(distance: float) -> float:
    """Convert Chroma L2 distance for normalized vectors into 0-1 similarity.

    Chroma's default L2 distance for normalized vectors behaves like squared
    L2 distance, where 0 is identical and unrelated vectors are often near 2.
    ``1 / (1 + distance)`` compresses that range too much; ``1 - distance/2``
    gives the grounded-refusal threshold useful separation.
    """
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


def retrieve_with_scores(vectordb, query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Similarity search, converting Chroma's raw distance to a useful
    0-1 similarity score."""
    raw_results = vectordb.similarity_search_with_score(query, k=k)
    scored = []
    for doc, distance in raw_results:
        similarity = distance_to_similarity(distance)
        scored.append((doc, similarity))
    return scored


def is_grounded(scored_docs: list[tuple[Document, float]], threshold: float = 0.35) -> bool:
    """Grounded-refusal gate.  Returns ``False`` if no docs or if the best
    score is below *threshold*."""
    if not scored_docs:
        return False
    best_score = max(score for _, score in scored_docs)
    return best_score >= threshold


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def format_citations(scored_docs: list[tuple[Document, float]]) -> str:
    """Build a 'Sources:' block purely from doc.metadata, deduplicated by
    (source, start_line, end_line).  Never derived from LLM output."""
    seen: set[tuple[str, int, int]] = set()
    lines: list[str] = []
    for doc, score in scored_docs:
        m = doc.metadata
        key = (m["source"], m["start_line"], m["end_line"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- {m['source']}: {m['section']} "
            f"(lines {m['start_line']}-{m['end_line']}) "
            f"— similarity {score:.2f}"
        )
    if not lines:
        return ""
    return "Sources:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Question condensing
# ---------------------------------------------------------------------------

CONDENSE_PROMPT = (
    "Given the following conversation history and a follow-up question, "
    "rewrite the follow-up question as a standalone question that can be "
    "understood without the conversation history.\n\n"
    "Conversation history:\n{history}\n\n"
    "Follow-up question: {question}\n\n"
    "Standalone question:"
)


def condense_question(llm, history_text: str, question: str) -> str:
    """Rewrite a follow-up into a standalone query using chat history.
    Skips the LLM call entirely if there's no history — saves a round-trip."""
    if not history_text or history_text.strip() == "(no previous turns)":
        return question
    prompt = CONDENSE_PROMPT.format(history=history_text, question=question)
    response = llm.invoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_PROMPT = (
    "You are a helpful customer support assistant for GigaCorp. "
    "Answer the customer's question using ONLY the provided context. "
    "If the context doesn't contain enough information, say so honestly. "
    "Do NOT make up information. Do NOT include a 'Sources' section in "
    "your response — sources are added separately.\n\n"
    "Context:\n{context}\n\n"
    "Conversation history:\n{history}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

REFUSAL_MESSAGE = (
    "I'm sorry, but I don't have enough information in our knowledge base "
    "to answer that question. Could you try rephrasing, or ask about our "
    "shipping, returns, business hours, or service tiers?"
)

GREETING_MESSAGE = (
    "Hello! I can help with GigaCorp shipping, returns, business hours, "
    "or service tiers. What would you like to know?"
)

THANKS_MESSAGE = (
    "You're welcome! Ask me anytime about GigaCorp shipping, returns, "
    "business hours, or service tiers."
)


def _normalized_tokens(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def small_talk_response(question: str) -> str | None:
    """Return a canned non-RAG response for greetings and brief courtesies."""
    tokens = _normalized_tokens(question)
    if not tokens or len(tokens) > 4:
        return None

    greeting_tokens = {
        "hey", "hi", "hello", "yo", "hiya", "sup", "morning", "afternoon",
        "evening",
    }
    thanks_tokens = {"thanks", "thank", "you", "thx", "ty"}
    filler_tokens = {"there", "good"}

    token_set = set(tokens)
    if token_set <= greeting_tokens | filler_tokens and token_set & greeting_tokens:
        return GREETING_MESSAGE
    if token_set <= thanks_tokens and token_set & thanks_tokens:
        return THANKS_MESSAGE
    return None


def generate_answer(llm, standalone_question: str, scored_docs: list[tuple[Document, float]], history_text: str) -> str:
    """Generate ONLY the answer body.  The model is explicitly instructed
    not to write its own 'Sources' line."""
    context = "\n\n".join(doc.page_content for doc, _ in scored_docs)
    prompt = ANSWER_PROMPT.format(
        context=context,
        history=history_text,
        question=standalone_question,
    )
    response = llm.invoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def answer_query(llm, vectordb, history_text: str, question: str, k: int = 3, threshold: float = 0.35) -> dict:
    """Full pipeline for one turn.  Returns dict with keys:
    ``standalone_question``, ``scored_docs``, ``grounded``, ``answer``.

    When ungrounded, ``answer`` is the fixed refusal string — not something
    the LLM wrote.

    Args:
        threshold: similarity threshold for grounded-refusal gate (default 0.35).
                   Can be overridden in tests using non-semantic fake embeddings.
    """
    canned_response = small_talk_response(question)
    if canned_response:
        return {
            "standalone_question": question,
            "scored_docs": [],
            "grounded": False,
            "answer": canned_response,
        }

    standalone = condense_question(llm, history_text, question)
    scored_docs = retrieve_with_scores(vectordb, standalone, k=k)
    grounded = is_grounded(scored_docs, threshold=threshold)

    if grounded:
        answer_body = generate_answer(llm, standalone, scored_docs, history_text)
        citations = format_citations(scored_docs)
        full_answer = f"{answer_body}\n\n{citations}" if citations else answer_body
    else:
        scored_docs = []
        full_answer = REFUSAL_MESSAGE

    return {
        "standalone_question": standalone,
        "scored_docs": scored_docs,
        "grounded": grounded,
        "answer": full_answer,
    }

"""
main.py — FastAPI app wiring rag_core + memory_store together.

Endpoints:
  POST   /sessions                         → create session
  GET    /sessions/{session_id}/history     → get message history
  POST   /sessions/{session_id}/messages    → send message, get RAG answer
  DELETE /sessions/{session_id}/messages    → clear history, keep session
  GET    /health                            → health check
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()  # Load .env before anything reads env vars
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_core import LocalKeywordEmbeddings, build_index, answer_query
from memory_store import MemoryStore, format_history_text


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FAQ_PATH = os.path.join(os.path.dirname(__file__), "data", "gigacorp_faq.txt")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.sqlite3")
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.35"))
PROJECT_VENV_PYTHON = os.path.normcase(
    os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
)


# ---------------------------------------------------------------------------
# Singleton state (populated at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Holds singleton instances created once at startup."""
    vectordb: Any = None
    llm: Any = None
    memory: MemoryStore | None = None


state = AppState()


def _select_llm_provider() -> str:
    """Select a configured provider, preferring an explicit LLM_PROVIDER."""
    configured = os.getenv("LLM_PROVIDER")
    if configured:
        return configured.lower()

    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "groq"


def _build_llm():
    provider = _select_llm_provider()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)
    if provider == "groq":
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(model=model, temperature=0, api_key=api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        return ChatAnthropic(model=model, temperature=0)

    raise RuntimeError(
        "Unsupported LLM_PROVIDER. Use one of: groq, gemini, anthropic, openai."
    )


def _build_embeddings():
    """Build lightweight local embeddings; no API key or model download needed."""
    return LocalKeywordEmbeddings()


def _ensure_correct_local_python() -> None:
    """Give Windows users a clear error when bare uvicorn resolves to Anaconda."""
    if os.name != "nt" or not os.path.exists(PROJECT_VENV_PYTHON):
        return

    current_python = os.path.normcase(sys.executable)
    if current_python == PROJECT_VENV_PYTHON:
        return

    if "anaconda" in current_python or "conda" in current_python:
        raise RuntimeError(
            "This project is being run with Anaconda Python instead of the "
            "project virtual environment. Stop this server and run: .\\run.ps1"
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize singletons at startup, release at shutdown."""
    _ensure_correct_local_python()
    embeddings = _build_embeddings()
    state.vectordb = build_index(FAQ_PATH, CHROMA_DIR, embeddings)
    state.llm = _build_llm()
    state.memory = MemoryStore(DB_PATH)
    yield
    # Cleanup (nothing to do for SQLite/Chroma)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GigaCorp Support Agent",
    description="RAG-based customer support with session memory",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for this assignment.
# NOTE: In production, restrict allowed_origins to the actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def get_vectordb():
    return state.vectordb

def get_llm():
    return state.llm

def get_memory() -> MemoryStore:
    return state.memory

def get_threshold() -> float:
    return GROUNDING_THRESHOLD


def _source_evidence(doc, max_chars: int = 240) -> str:
    """Return a compact evidence preview for the UI source panel."""
    text = " ".join(doc.page_content.split())
    section = doc.metadata.get("section", "")
    if section and text.startswith(section):
        text = text[len(section):].strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class MessageRequest(BaseModel):
    message: str

class SessionResponse(BaseModel):
    session_id: str

class SourceItem(BaseModel):
    source: str
    section: str
    start_line: int
    end_line: int
    similarity: float
    evidence: str

class AnswerResponse(BaseModel):
    answer: str
    grounded: bool
    standalone_question: str
    sources: list[SourceItem]

class HistoryMessage(BaseModel):
    role: str
    content: str

class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=SessionResponse)
def create_session(memory: MemoryStore = Depends(get_memory)):
    """Create a new chat session."""
    session_id = memory.create_session()
    return SessionResponse(session_id=session_id)


@app.get("/sessions/{session_id}/history", response_model=list[HistoryMessage])
def get_history(session_id: str, memory: MemoryStore = Depends(get_memory)):
    """Retrieve full message history for a session."""
    if not memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    history = memory.get_history(session_id)
    return [HistoryMessage(role=role, content=content) for role, content in history]


@app.post("/sessions/{session_id}/messages", response_model=AnswerResponse)
def post_message(
    session_id: str,
    body: MessageRequest,
    memory: MemoryStore = Depends(get_memory),
    vectordb=Depends(get_vectordb),
    llm=Depends(get_llm),
    threshold: float = Depends(get_threshold),
):
    """Send a message and get a RAG-powered answer."""
    if not memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Get conversation history for context
    history = memory.get_history(session_id)
    history_text = format_history_text(history)

    try:
        result = answer_query(llm, vectordb, history_text, body.message, threshold=threshold)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM provider request failed. Check LLM_PROVIDER, API key, "
                "model name, and provider quota before retrying."
            ),
        ) from exc

    # Persist both messages
    memory.append_message(session_id, "user", body.message)
    memory.append_message(session_id, "assistant", result["answer"])

    # Build sources list from scored_docs metadata
    sources = []
    seen_keys = set()
    for doc, score in result["scored_docs"]:
        m = doc.metadata
        key = (m["source"], m["start_line"], m["end_line"])
        if key not in seen_keys:
            seen_keys.add(key)
            sources.append(SourceItem(
                source=m["source"],
                section=m["section"],
                start_line=m["start_line"],
                end_line=m["end_line"],
                similarity=round(score, 4),
                evidence=_source_evidence(doc),
            ))

    return AnswerResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        standalone_question=result["standalone_question"],
        sources=sources,
    )


@app.delete("/sessions/{session_id}/messages")
def clear_messages(session_id: str, memory: MemoryStore = Depends(get_memory)):
    """Clear message history but keep the session alive."""
    if not memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    memory.clear_session(session_id)
    return {"status": "cleared"}


@app.get("/health", response_model=HealthResponse)
def health_check(vectordb=Depends(get_vectordb)):
    """Health check — confirms vector store is loaded."""
    return HealthResponse(
        status="ok",
        vector_store_loaded=vectordb is not None,
    )


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------

# Serve the frontend directory for static assets
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    def serve_frontend():
        """Serve the frontend index.html at root."""
        return FileResponse(os.path.join(frontend_dir, "index.html"))

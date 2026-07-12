# GigaCorp Support Agent — RAG + Memory

AI-powered customer support agent with Retrieval-Augmented Generation (RAG) and persistent session memory, built as a **decoupled FastAPI backend + vanilla JS frontend**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (index.html)                        │
│         Vanilla HTML/CSS/JS • Session management via localStorage   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ REST API (fetch)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (main.py)                        │
│                                                                     │
│  POST /sessions              → Create session                       │
│  GET  /sessions/{id}/history → Get message history                  │
│  POST /sessions/{id}/messages→ RAG pipeline + persist               │
│  DELETE /sessions/{id}/messages → Clear history                     │
│  GET  /health                → Health check                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   RAG Pipeline (rag_core.py)                  │   │
│  │                                                               │   │
│  │  1. condense_question(history, query) → standalone query      │   │
│  │  2. retrieve_with_scores(vectordb, query) → scored docs       │   │
│  │  3. is_grounded(scored_docs, threshold) → bool                │   │
│  │  4. generate_answer(llm, query, docs) → answer body           │   │
│  │  5. format_citations(scored_docs) → metadata-only sources     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                       │                   │
│              ┌───────────┘                       └──────────┐       │
│              ▼                                              ▼       │
│  ┌──────────────────┐                          ┌─────────────────┐  │
│  │   Chroma (disk)   │                          │  SQLite (disk)   │  │
│  │  Vector Store     │                          │  Session Memory  │  │
│  │  (auto-built on   │                          │  (memory_store.py│  │
│  │   first run)      │                          │   sessions +     │  │
│  └──────────────────┘                          │   messages)      │  │
│                                                 └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Separate Backend and Frontend over REST API
Unlike monolithic Streamlit apps, this architecture has a **genuine network boundary** between frontend and backend. Open Chrome DevTools → Network tab and you'll see the actual `POST /sessions/{id}/messages` call with JSON request/response — clear visual proof of separation.

### 2. Citations Computed from Retrieval Metadata, Never LLM-Generated
The model is **never trusted** to state a source or line number. All citations are constructed from `doc.metadata` (source file, section header, exact start/end line numbers) returned by the vector store. The UI source panel also shows a short evidence snippet from the retrieved document so reviewers can inspect the grounding directly.

### 3. Grounded Refusal
If the best retrieval similarity score is below the threshold (default 0.35), the API returns a **fixed refusal message** instead of letting the LLM improvise. This prevents hallucinated answers for off-topic queries — the LLM never gets called when the system can't find grounding.

### 4. Fully Offline Test Suite
All tests run with **zero API keys and zero network calls** using dependency-injected `FakeEmbeddings` (deterministic hash-based vectors) and `FakeLLM` (canned response queue). The grading team can run `pytest` immediately.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Orchestration | LangChain (LCEL-style calls) |
| Vector Store | Chroma (persisted to disk, auto-built) |
| Embeddings | Lightweight local keyword embeddings (no download, no API key) |
| LLM | Groq `llama-3.3-70b-versatile` (default) / Gemini, Anthropic, or OpenAI via env vars |
| Session Persistence | SQLite (stdlib `sqlite3`) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Testing | pytest with FakeEmbeddings + FakeLLM |

---

## Quick Start

### Prerequisites
- Python 3.11+
- A Groq API key by default, or a Gemini/Anthropic/OpenAI key if you change `LLM_PROVIDER`

### Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd gigacorp-support-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run Tests (no API key needed!)

```bash
pytest tests/ -v
```

All 59 tests should pass with zero network calls.

### Run the Server

```powershell
.\run.ps1
```

Then open http://localhost:8000 in your browser.

The Chroma vector store is built automatically on first startup from `data/gigacorp_faq.txt`. Subsequent starts load the persisted index from `chroma_db/`.

### Using Another LLM Provider

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-your-key-here
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

```bash
export LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your-google-ai-studio-key
export GEMINI_MODEL=gemini-2.5-flash
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

---

## Project Structure

```
├── main.py                 # FastAPI app — wires rag_core + memory_store
├── rag_core.py             # Pure RAG logic (parsing, indexing, retrieval, generation)
├── memory_store.py         # Pure SQLite session/memory logic
├── data/
│   └── gigacorp_faq.txt    # Mock FAQ knowledge base (4 sections, ~33 lines)
├── frontend/
│   └── index.html          # Single-file vanilla HTML/CSS/JS chat UI
├── tests/
│   ├── fakes.py            # FakeEmbeddings + FakeLLM for offline testing
│   ├── test_chunker.py     # FAQ parsing and Document creation tests
│   ├── test_retriever.py   # Vector store, retrieval, grounding, pipeline tests
│   ├── test_memory_store.py# SQLite session/memory persistence tests
│   └── test_api.py         # FastAPI endpoint integration tests
├── requirements.txt        # Pinned dependencies
├── render.yaml             # Render free-tier deployment config
├── run.ps1                 # Windows launcher that uses the project .venv
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create new session → `{"session_id": "..."}` |
| `GET` | `/sessions/{id}/history` | Get full message history (404 if unknown) |
| `POST` | `/sessions/{id}/messages` | Send message → RAG answer with sources |
| `DELETE` | `/sessions/{id}/messages` | Clear history, keep session alive |
| `GET` | `/health` | Health check + vector store status |

### POST /sessions/{id}/messages

**Request:**
```json
{"message": "What is your shipping policy?"}
```

**Response:**
```json
{
  "answer": "GigaCorp offers three shipping tiers...\n\nSources:\n- gigacorp_faq.txt: SHIPPING POLICY (lines 1-9) — similarity 0.82",
  "grounded": true,
  "standalone_question": "What is your shipping policy?",
  "sources": [
    {
      "source": "gigacorp_faq.txt",
      "section": "SHIPPING POLICY",
      "start_line": 1,
      "end_line": 9,
      "similarity": 0.82,
      "evidence": "GigaCorp offers three shipping tiers for all domestic orders..."
    }
  ]
}
```

---

## Deployment (Render)

### Backend on Render

1. Push to a public GitHub repo.
2. Render → **New** → **Web Service** → connect repo.
3. Render will detect `render.yaml`; otherwise set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add `GROQ_API_KEY` under **Environment**.

### Frontend

The FastAPI app serves `frontend/index.html` at `/`, and the frontend uses the current hosted origin automatically. No separate frontend deployment is required.

### ⚠️ Known Limitation
Render's free tier spins down on inactivity and the SQLite file lives on **ephemeral disk**, so session history won't survive a cold start after a long idle period. This is an acceptable trade-off for a free-tier demo deployment.

---

## Demo Video Script (2-3 min)

### Beat 1: UX Flow (~45s)
- Show the frontend, create a new session
- Ask: "What is your shipping policy?" → Show the answer with sources
- Expand the sources panel → Point out section name, line numbers, similarity scores, and evidence snippets
- Emphasize: citations come from retrieval metadata, not the LLM

### Beat 2: Memory / Follow-up (~30s)
- Ask: "How much does express cost?" (pronoun + context from previous turn)
- Show the standalone_question in the response → Demonstrates question condensing
- Point out the session ID in the header → Show it persists on refresh

### Beat 3: Grounded Refusal (~20s)
- Ask: "What is the capital of France?"
- Show the fixed refusal message → "I don't have that information"
- Explain: below the similarity threshold, the LLM is never called

### Beat 4: Live pytest (~20s)
- Run `pytest tests/ -v` in terminal
- Show all 59 tests passing with zero API key required

### Beat 5: Network Tab (~20s)
- Open Chrome DevTools → Network tab
- Send a message → Show the `POST /sessions/{id}/messages` request
- Show the JSON response with answer + sources
- This is the single clearest proof that frontend and backend are separate

---

## License

MIT

# Submission Checklist

## What This Project Includes

- FastAPI backend with LangChain orchestration.
- Chroma local vector store built automatically from `data/gigacorp_faq.txt`.
- SQLite-backed conversational memory.
- Vanilla HTML/CSS/JS chat UI served by FastAPI.
- Source citations with file name, section, line range, similarity score, and evidence snippet.
- Grounded refusal for off-topic questions.
- Small-talk handling so greetings do not get fake citations.
- Optional LLM providers: Groq, Gemini, Anthropic, and OpenAI.
- Render free-tier deployment config in `render.yaml`.
- Offline tests with fake LLM and fake embeddings.

## Reviewer Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
notepad .env
.\run.ps1
```

Open:

```text
http://127.0.0.1:8000
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected result:

```text
59 passed
```

## Do Not Submit Local Secrets Or Runtime Files

These are intentionally excluded from the clean submission package:

- `.env`
- `.venv/`
- `sessions.sqlite3`
- `chroma_db/`
- `__pycache__/`
- `.pytest_cache/`
- local tool folders such as `.qodo/` and `.sixth/`


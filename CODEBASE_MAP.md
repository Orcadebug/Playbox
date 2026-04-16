# Waver Codebase Map

## Overview
Full-stack MVP: instant search over messy data sources. Upload documents → parse → BM25 rank + rerank → optional LLM answers.

---

## Backend (FastAPI + Python)

```
backend/
├── app/
│   ├── main.py                 # FastAPI app, startup/shutdown
│   ├── config.py               # Pydantic Settings (env vars)
│   │
│   ├── api/
│   │   ├── router.py           # API router setup
│   │   └── v1/
│   │       ├── health.py       # GET /healthz
│   │       ├── upload.py       # POST /upload (FormData)
│   │       ├── search.py       # POST /search, POST /search/stream
│   │       └── sources.py      # GET /sources, DELETE /sources/{id}
│   │
│   ├── services/
│   │   ├── search.py           # Search logic (orchestrates retrieval + answers)
│   │   └── sources.py          # Source/document CRUD
│   │
│   ├── retrieval/
│   │   ├── pipeline.py         # Orchestrates chunking → BM25 → reranking
│   │   ├── chunker.py          # Text chunking with overlap
│   │   ├── tokenizer.py        # Tokenization logic
│   │   ├── bm25.py             # BM25 scoring wrapper
│   │   ├── bm25_cache.py       # BM25 index caching
│   │   └── reranker.py         # ONNX cross-encoder inference
│   │
│   ├── answer/
│   │   ├── generator.py        # LLM answer generation (OpenRouter/Anthropic)
│   │   ├── citations.py        # Maps answer spans → source documents
│   │   └── prompts.py          # System prompts for LLM
│   │
│   ├── parsers/
│   │   ├── base.py             # BaseParser interface
│   │   ├── detector.py         # MIME type → parser mapping
│   │   ├── registry.py         # Parser registry
│   │   ├── pdf.py              # PDF parsing (pypdf)
│   │   ├── markdown.py         # Markdown parsing
│   │   ├── html.py             # HTML parsing (BeautifulSoup)
│   │   ├── plaintext.py        # Plain text
│   │   ├── json_parser.py      # JSON parsing
│   │   └── csv_parser.py       # CSV parsing (pandas)
│   │
│   ├── connectors/
│   │   ├── base.py             # BaseConnector (abstract)
│   │   ├── registry.py         # Connector registry
│   │   ├── slack.py            # Slack integration
│   │   └── webhook.py          # Webhook integration
│   │
│   ├── models/
│   │   ├── base.py             # SQLAlchemy base
│   │   ├── source.py           # Source (uploaded file metadata)
│   │   └── document.py         # Document (parsed chunks)
│   │
│   ├── schemas/
│   │   ├── documents.py        # Document response schemas
│   │   └── search.py           # Search request/response schemas
│   │
│   └── db/
│       └── session.py          # Database session management
│
├── tests/
│   └── test_retrieval.py       # Retrieval pipeline tests
│
├── scripts/
│   └── download_models.py      # Download BM25/reranker models
│
├── alembic/                    # Database migrations
├── pyproject.toml              # uv dependencies
├── uv.lock                     # Lockfile
└── Dockerfile                  # Multi-stage build
```

**Key Flow:**
```
POST /upload
  ↓
parse file (detector → registry → parser)
  ↓
chunk text (chunker + tokenizer)
  ↓
store Source + Documents in DB

POST /search/stream
  ↓
retrieve docs (BM25 rank + reranker)
  ↓
stream search results (SSE)
  ↓
(optional) generate answer (LLM + citations)
```

---

## Frontend (Next.js 15 + React 19)

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout (AppShell)
│   │   ├── page.tsx            # / (home)
│   │   ├── upload/
│   │   │   └── page.tsx        # /upload page
│   │   ├── search/
│   │   │   └── page.tsx        # /search page
│   │   └── demo/
│   │       └── page.tsx        # /demo page
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── AppShell.tsx    # Main wrapper (layout structure)
│   │   │   ├── Header.tsx      # Top nav
│   │   │   └── Sidebar.tsx     # Sidebar (workspace selector)
│   │   │
│   │   ├── upload/
│   │   │   ├── UploadWorkspace.tsx    # Upload page container
│   │   │   ├── FileDropzone.tsx       # Drag-drop file upload
│   │   │   ├── PasteBox.tsx           # Paste text input
│   │   │   └── ConnectorPicker.tsx    # Slack/Webhook selector
│   │   │
│   │   └── search/
│   │       ├── SearchWorkspace.tsx    # Search page container
│   │       ├── SearchBar.tsx          # Search input
│   │       ├── ResultList.tsx         # Results container
│   │       ├── ResultCard.tsx         # Individual result item
│   │       └── AnswerCard.tsx         # LLM answer display
│   │
│   └── lib/
│       └── api-client.ts       # Typed fetch wrapper (all backend calls)
│
├── public/                     # Static assets
├── next.config.ts              # Next.js config
├── tsconfig.json               # TypeScript strict mode
├── tailwind.config.ts          # Tailwind 4.0 config
└── package.json                # npm dependencies
```

**Key Flow:**
```
/upload
  ↓
FileDropzone or PasteBox input
  ↓
POST /upload → backend parse
  ↓
workspace documents displayed

/search
  ↓
SearchBar input + workspace selection
  ↓
POST /search/stream (EventSource)
  ↓
ResultList streams results
  ↓
(optional) AnswerCard displays LLM answer
```

---

## Cross-Cutting

| Concept | Location | Purpose |
|---------|----------|---------|
| **Workspace namespacing** | Request params | Isolate documents per workspace_id |
| **Streaming (SSE)** | `/search/stream` endpoint + EventSource | Real-time result delivery |
| **CORS** | `app/main.py` | Cross-origin requests via `CORS_ORIGINS` |
| **Database** | `app/models/` + `alembic/` | Postgres (Neon prod), SQLite dev |
| **Type hints** | `app/schemas/` | Request/response validation |

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Health check |
| POST | `/upload` | Upload file (FormData, workspace_id required) |
| POST | `/search` | Search with optional LLM answer |
| POST | `/search/stream` | Streaming search (SSE) |
| GET | `/sources` | List workspace sources |
| DELETE | `/sources/{id}` | Delete source + documents |

**Interactive docs:** `http://localhost:8000/docs`

---

## Environment Variables (Key)

```
DATABASE_URL=postgres://...       # Postgres connection (SQLite fallback local dev)
OPENROUTER_API_KEY=...            # OpenRouter (LLM answers)
ANTHROPIC_API_KEY=...             # Anthropic (LLM answers)
NEXT_PUBLIC_API_BASE_URL=...      # Frontend → backend URL
WAVER_RERANKER_MODEL=...          # Cross-encoder model name
CORS_ORIGINS=...                  # Allowed origins
```

See `.env.example` for full list.

---

## Key Dependencies

**Backend:**
- `fastapi` — web framework
- `sqlalchemy` — ORM
- `alembic` — migrations
- `bm25s` — BM25 ranking
- `transformers` — reranker inference
- `pypdf`, `beautifulsoup4`, `pandas` — parsers

**Frontend:**
- `next` — framework
- `react` — UI
- `tailwindcss` — styling

**Package Managers:**
- Backend: `uv`
- Frontend: `npm`

---

## Deployment

| Service | Platform | Config |
|---------|----------|--------|
| Backend | Fly.io | `fly.toml` |
| Frontend | Vercel | `vercel.json` |
| Database | Neon | Managed Postgres |
| Docker | Local dev | `docker-compose.dev.yml` |

---

## Dev Commands

```bash
make dev              # Docker: full stack
make install          # Install deps (uv + npm)
make backend-dev      # FastAPI :8000
make frontend-dev     # Next.js :3000
make test             # pytest backend
make lint             # ruff + next lint
make format           # ruff + next format
```

Single test: `cd backend && uv run pytest tests/test_retrieval.py::test_name -v`

---

## Conventions

| Aspect | Rule |
|--------|------|
| **Python** | `snake_case`, type hints, Ruff (100 line-length) |
| **TypeScript** | Strict mode, `PascalCase` components, `camelCase` helpers |
| **Commits** | Conventional: `feat(scope):`, `fix(scope):` |
| **Styling** | Tailwind CSS (no CSS modules) |
| **Import alias** | `@/*` → `src/` |

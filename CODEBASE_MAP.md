# Waver Codebase Map

## Overview
Full-stack MVP: retrieval-first search over messy data sources. Search can combine saved
workspace sources, inline raw sources, and transient connector payloads. The backend parses
sources on demand, retrieves/reranks candidate text, and returns exact span payloads with
source offsets. Upload remains available for saved-source management; LLM answers are
explicitly opt-in.

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
│   │   ├── search.py           # Stored/raw/connector source loading + retrieval orchestration
│   │   └── sources.py          # Source/document CRUD
│   │
│   ├── retrieval/
│   │   ├── pipeline.py         # Parses, chunks, retrieves, reranks, builds span payloads
│   │   ├── chunker.py          # Exact-offset text chunking with overlap
│   │   ├── tokenizer.py        # Tokenization logic
│   │   ├── bm25.py             # BM25 scoring wrapper
│   │   ├── bm25_cache.py       # Stored-only BM25 index caching
│   │   ├── reranker.py         # ONNX cross-encoder inference
│   │   ├── retriever.py        # Unified retriever entry point
│   │   ├── trie.py             # Cortical trie index (prefix structure)
│   │   ├── cortical.py         # Cortical Trie Search retriever
│   │   ├── sparse_projection.py # Sparse vector projection
│   │   ├── diffusion.py        # Score diffusion over adjacency graph
│   │   ├── adjacency.py        # Chunk adjacency graph
│   │   ├── gating.py           # Retrieval gating / routing
│   │   └── query_patterns.py   # Query pattern detection
│   │
│   ├── answer/
│   │   ├── generator.py        # Opt-in LLM answer generation (OpenRouter/Anthropic)
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
│   │   ├── source.py           # Saved source metadata
│   │   └── document.py         # Saved parsed document content
│   │
│   ├── schemas/
│   │   ├── documents.py        # Document response schemas
│   │   └── search.py           # Search request/response schemas
│   │
│   └── db/
│       └── session.py          # Database session management
│
├── tests/
│   ├── test_retrieval.py       # Retrieval pipeline/span/cache tests
│   ├── test_search_service.py  # Raw source, connector, answer-mode tests
│   ├── test_cortical.py        # Cortical retrieval tests
│   └── test_parsers.py         # Parser tests
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
POST /search or /search/stream
  ↓
load saved sources + raw_sources + connector_configs
  ↓
parse and chunk with exact source offsets
  ↓
retrieve/rerank candidates
  ↓
return results with primary_span, matched_spans, source_origin, source_errors
  ↓
optional answer_mode="llm" calls AnswerGenerator

POST /upload
  ↓
parse file or pasted text
  ↓
store Source + Document rows for future searches
  ↓
invalidate stored-source BM25 cache
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
│   │       ├── SourceControls.tsx     # Saved/raw/webhook source controls
│   │       ├── ResultList.tsx         # Results container
│   │       ├── ResultCard.tsx         # Individual result item
│   │       └── AnswerCard.tsx         # Optional LLM answer display
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
/search
  ↓
SearchBar + SourceControls
  ↓
POST /search/stream with include_stored_sources, raw_sources, connector_configs
  ↓
ResultList displays exact spans, source origin, offsets, score
  ↓
Optional "Generate answer" reruns search with answer_mode="llm"

/upload
  ↓
FileDropzone or PasteBox input
  ↓
POST /upload → backend parse/store
  ↓
workspace saved sources displayed
```

---

## Cross-Cutting

| Concept | Location | Purpose |
|---------|----------|---------|
| **Workspace namespacing** | Request params | Isolate documents per workspace_id |
| **Raw source search** | `raw_sources` request field | Search transient request content without storing |
| **Connector search** | `connector_configs` request field | Search transient connector results |
| **Span payloads** | `primary_span`, `matched_spans` | Return exact snippets and source offsets |
| **Streaming (SSE)** | `/search/stream` endpoint + fetch stream | Stream retrieval response payloads |
| **CORS** | `app/main.py` | Cross-origin requests via `CORS_ORIGINS` |
| **Database** | `app/models/` + `alembic/` | Saved sources only; raw/connector searches are transient |
| **Type hints** | `app/schemas/` | Request/response validation |

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Health check |
| POST | `/upload` | Upload file (FormData, workspace_id required) |
| POST | `/search` | Retrieval over saved/raw/connector sources |
| POST | `/search/stream` | Streaming retrieval response (SSE) |
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
ENABLE_LIVE_CONNECTORS=true       # Enable external live connector fetches
CORS_ORIGINS=...                  # Allowed origins
```

See `.env.example` for full list.

---

## Key Dependencies

**Backend:**
- `fastapi` — web framework
- `sqlalchemy` — ORM
- `alembic` — migrations
- `PyStemmer` — optional stemming for BM25 tokenization
- `onnxruntime` — cross-encoder reranker inference
- `pyahocorasick` — query pattern scanning for exact spans/cortical retrieval
- `numpy`, `scipy`, `scikit-learn` — retrieval math/projection helpers
- `pymupdf`, `beautifulsoup4` — parsers

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

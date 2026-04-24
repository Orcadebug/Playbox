# Waver Codebase Map

## Overview
Full-stack MVP: retrieval-first search over messy data sources. Search can combine saved
workspace sources, inline raw sources, and transient connector payloads. The backend parses
sources on demand, executes query-time span retrieval/reranking, and returns exact span
payloads with source offsets. Upload remains available for saved-source management; LLM
answers are explicitly opt-in.

Frontend note: the UI is an API-key onboarding and smoke-test surface only. The primary
integration surface is the backend API.

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
│   │   ├── pipeline.py          # Adaptive budget, reranking, span building
│   │   ├── source_executor.py   # SourceRecord/SourceWindow creation from parsed docs
│   │   ├── planner.py           # Query-time budget + channel planning
│   │   ├── channels.py          # Exact/proxy/structure channel scoring
│   │   ├── span_executor.py     # Query-time span execution and phase outputs
│   │   ├── chunker.py           # Exact-offset text chunking with overlap
│   │   ├── tokenizer.py         # Tokenization logic
│   │   ├── bm25.py              # BM25 scoring wrapper
│   │   ├── bm25_cache.py        # Stored-only BM25 index caching
│   │   ├── reranker.py          # ONNX cross-encoder inference (+ remote gRPC seam)
│   │   ├── retriever.py         # MultiHeadRetriever (RRF over sps/bm25/phrase)
│   │   ├── sps.py               # SpsRetriever (BM25 + sparse projection cosine)
│   │   ├── exact_phrase.py      # ExactPhraseRetriever (substring/n-gram matches)
│   │   ├── sparse_projection.py # SparseProjection + deterministic hash projection + polarity injection
│   │   ├── sentence_scorer.py   # Sentence-level span refinement inside top chunk
│   │   ├── rust_core.py         # Python wrappers around waver_core Rust exports
│   │   ├── eval_harness.py      # Retrieval eval harness
│   │   ├── eval_layers.py       # Per-layer eval probes
│   │   ├── eval_dynamic.py      # Dynamic corpus eval (add/delete/mutate/reorder)
│   │   ├── eval_fixtures/       # AI-authored eval corpora
│   │   ├── trie.py              # Cortical trie index (prefix structure)
│   │   ├── cortical.py          # Cortical Trie Search retriever
│   │   ├── diffusion.py         # Score diffusion over adjacency graph
│   │   ├── adjacency.py         # Chunk adjacency graph
│   │   ├── gating.py            # Retrieval gating / routing
│   │   └── query_patterns.py    # Query pattern detection
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
│   ├── test_retrieval.py           # Retrieval pipeline/span/cache tests
│   ├── test_search_service.py      # Raw source, connector, answer-mode tests
│   ├── test_cortical.py            # Cortical retrieval tests
│   ├── test_parsers.py             # Parser tests
│   ├── test_planner.py             # Planner tier/budget tests
│   ├── test_span_executor.py       # Span executor phase tests
│   ├── test_source_executor.py     # SourceRecord/SourceWindow tests
│   ├── test_ephemeral_api.py       # Raw / connector path tests
│   ├── test_eval_harness.py        # Eval harness unit tests
│   ├── test_eval_layers.py         # Per-layer eval probes
│   ├── test_eval_dynamic.py        # Dynamic corpus mutations
│   ├── test_reranker_remote.py     # Remote reranker fallback tests
│   ├── test_ghost.py               # GhostProxy bloom/CMS tests
│   └── fixtures/                   # Eval fixtures (tickets, mutations)
│
├── scripts/
│   ├── download_models.py          # Fetch reranker + projection artifacts
│   ├── train_projection.py         # Train sparse projection on corpus
│   ├── run_retrieval_eval.py       # Manual eval run
│   └── run_vision_eval.py          # Vision/promise eval driver
│
├── waver_core/                     # Rust pyo3 core (rrf_fuse, prefilter_windows,
│                                   #   mrl_encode, phrase_search, RustBm25Index).
│                                   #   Built via `maturin develop --release`.
├── waver_ghost/                    # GhostProxy: fixed-memory Bloom + Count-Min Sketch
│                                   #   for edge-side zero-hit short-circuit.
├── services/
│   └── reranker/                   # Optional stage-2 gRPC reranker service
│                                   #   (WAVER_RERANKER_GRPC_TARGET).
│
├── alembic/                        # Database migrations
├── pyproject.toml                  # uv dependencies + ruff config
├── uv.lock                         # Lockfile
└── Dockerfile                      # Multi-stage build
```

**Key Flow:**
```
POST /search or /search/stream
  ↓
load saved sources + raw_sources + connector_configs
  ↓
parse + source/window execution with exact source offsets
  ↓
planner selects tier/budget/channels
  ↓
Stage-0 trigram prefilter (waver_core.prefilter_windows, AVX-512/AVX2/scalar)
  ↓
heads run in parallel: sps (BM25+projection), bm25, exact_phrase
  ↓
RRF fusion (Python, or waver_core.rrf_fuse when WAVER_RUST_RRF=true)
  ↓
rerank shortlist (remote gRPC → local ONNX → heuristic)
  ↓
sentence-level refinement of primary_span
  ↓
return phased SSE events + final retrieval payload
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
│   │   │   └── page.tsx        # /search page (smoke harness)
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
│   │       ├── SearchWorkspace.tsx    # Stream phase viewer + smoke harness
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

**Key Flow (UI as onboarding/smoke surface):**
```
/search
  ↓
SearchBar + SourceControls
  ↓
POST /search/stream with include_stored_sources, raw_sources, connector_configs
  ↓
SearchWorkspace consumes typed stream events and progressively updates state
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
| **Streaming (SSE)** | `/search/stream` endpoint + fetch stream | Stream typed retrieval phases + final response |
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
| POST | `/search/stream` | Progressive typed SSE retrieval stream |
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
WAVER_RERANKER_GRPC_TARGET=...    # Optional remote reranker endpoint
WAVER_RETRIEVER=sps               # sps (default) | bm25 | cortical
WAVER_SPS_ALPHA=0.6               # BM25 vs projection blend
WAVER_MULTIHEAD=true              # RRF fusion sps/bm25/phrase
WAVER_RRF_K=60                    # RRF damping
WAVER_RUST_RRF=false              # Run rrf_fuse via waver_core (Rust)
WAVER_RUST_RRF_SHADOW=false       # Shadow-log Rust vs Python fusion deltas
WAVER_SPS_NEGATION=true           # Polarity-aware ranking
WAVER_ADAPTIVE_BUDGET=true        # Expand budget on flat score spread
PROJECTION_MODEL_PATH=...         # sparse projection .npz (optional)
WAVER_ORT_DYLIB_PATH=...          # ONNX Runtime shared lib for Rust MRL
ENABLE_LIVE_CONNECTORS=true       # Enable external live connector fetches
MAX_UPLOAD_BYTES=52428800         # 50 MB default
CORS_ORIGINS=...                  # Allowed origins
```

See `.env.example` for full list.

---

## Key Dependencies

**Backend:**
- `fastapi` — web framework
- `sqlalchemy` — ORM
- `alembic` — migrations
- `bm25s` + `PyStemmer` — BM25 scoring / optional stemming
- `tokenizers` — fast tokenization
- `onnxruntime` — cross-encoder reranker inference
- `pyahocorasick` — query pattern scanning for exact spans/cortical retrieval
- `numpy`, `scipy`, `scikit-learn` — retrieval math/projection helpers
- `pymupdf`, `pypdfium2`, `beautifulsoup4` — parsers
- `structlog` — structured logging
- `maturin` / `pyo3` (build-time) — native `waver_core` compilation

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

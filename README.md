# Waver

Waver is a retrieval-first API for searching messy sources and returning the exact
source spans that matter. It is built for situations where a user has fresh source
input now - tickets, logs, CSV exports, JSON payloads, PDFs, markdown notes, HTML, or
connector results - and needs trustworthy snippets, offsets, and citations without a
required pre-ingestion or vector-indexing step.

The product center is retrieval, not chat. LLM answers are optional and should sit on
top of retrieved source spans, never replace them.

## What Waver Does

Waver can currently:

- Search saved workspace sources, inline raw request sources, and transient connector
  payloads in one request.
- Upload files or pasted text when content should be saved and searched later.
- Parse common formats including plain text, markdown, CSV, JSON, PDF, HTML, logs, and
  newline-delimited text-like files.
- Search fresh raw sources without persisting them or warming a corpus-specific index.
- Return ranked results with `primary_span`, `matched_spans`, source offsets, source
  origin, channel metadata, and execution telemetry.
- Mark partial scans in execution/result metadata instead of silently pretending every
  large corpus was fully scanned.
- Stream retrieval phases over SSE: `sources_loaded`, `exact_results`,
  `proxy_results`, `reranked_results`, optional `answer_delta`, and `done`.
- Use exact matching, semantic-proxy scoring, structure scoring, and reranking inside
  a query-time span executor.
- Use a BM25 cache only as an optimization for stored-only searches.
- Search webhook connector payloads transiently; Slack connector code exists but live
  connectors are disabled by default.
- Run a support-ticket promise eval that checks exact spans, semantic context,
  dynamic corpus changes, and cold-corpus behavior.

Good early use cases:

- Support ticket lookup where an agent needs the exact sentence or field that explains
  a customer problem.
- Searching exported CSV/JSON/log bundles without building a long-lived index first.
- API-first retrieval for products that need cited snippets and offsets.
- Connector-backed search over webhook payloads, with Slack-style live connectors as a
  gated extension.
- Source-grounded answer generation where synthesis is optional and evidence remains
  visible.

## Architecture

```text
Client or frontend
  -> FastAPI routes in backend/app/api/v1
  -> SearchService loads stored, raw, and connector documents
  -> parsers normalize files into searchable documents/windows
  -> SpanExecutor plans the scan and scores exact, semantic, and structure channels
  -> reranker orders the shortlist
  -> SearchService serializes spans, citations, source errors, and execution metadata
  -> optional AnswerGenerator produces an LLM answer from retrieved results
```

Key backend areas:

- `backend/app/api/v1/`: FastAPI routes for health, upload, search, streaming search,
  and source management.
- `backend/app/services/search.py`: request orchestration, raw-source loading,
  connector loading, streaming envelopes, result serialization, and optional answers.
- `backend/app/services/sources.py`: saved-source upload, parsing, listing, deletion,
  and cache invalidation.
- `backend/app/parsers/`: format-specific parsers and parser detection.
- `backend/app/retrieval/`: BM25 cache, source windows, planner, channels, span
  executor, rerankers, sparse projection, cortical retriever experiments, and evals.
- `backend/app/answer/`: optional source-grounded answer generation through
  OpenRouter when `OPENROUTER_API_KEY` is configured.
- `backend/tests/`: pytest coverage for parsers, retrieval, search service behavior,
  source execution, planner behavior, API-key work in progress, and promise evals.

Frontend status:

- The Next.js app is intentionally secondary to the API.
- The UI should be treated as beta/demo/onboarding surface, not the canonical product
  interface yet.
- The frontend has had both live search/upload workspace work and API-key onboarding
  work in flight. Before public beta, the routes and README claims should be kept in
  lockstep so users know whether they are using a live backend UI or a seeded demo.

## API Surface

Core search endpoints:

- `POST /api/v1/search`
- `POST /api/v1/search/stream`

Search accepts:

- `query`
- `top_k`
- `source_ids`
- `workspace_id`
- `raw_sources`
- `connector_configs`
- `include_stored_sources`
- `answer_mode`: `"off"` or `"llm"`

Saved-source endpoints:

- `POST /api/v1/upload`
- `GET /api/v1/sources`
- `DELETE /api/v1/sources/{source_id}`

Beta/API-key work is in progress in this repository. When that flow is enabled, hosted
usage should mint guest keys, scope workspaces by key, and store only hashed key
material. Until that is finalized and committed cleanly with migrations, treat local
API-key behavior as beta infrastructure rather than the core retrieval contract.

## Search Example

```json
{
  "query": "billing refund",
  "top_k": 8,
  "include_stored_sources": true,
  "answer_mode": "off",
  "raw_sources": [
    {
      "id": "scratchpad",
      "name": "Raw scratchpad",
      "content": "Acme asked for a billing refund after a duplicate charge.",
      "media_type": "text/plain",
      "source_type": "raw"
    }
  ],
  "connector_configs": [
    {
      "connector_id": "webhook",
      "documents": [
        {
          "name": "ticket-7.txt",
          "content": "Acme opened an invoice dispute yesterday.",
          "media_type": "text/plain"
        }
      ]
    }
  ]
}
```

The response is retrieval-first:

- `results`: ranked source results.
- `results[].primary_span`: the best span for the result.
- `results[].matched_spans`: additional matched spans.
- `results[].source_origin`: `stored`, `raw`, or `connector`.
- `results[].metadata`: phase/channel/source metadata.
- `sources`: unique source summaries.
- `source_errors`: raw/connector loading errors that did not stop the whole search.
- `execution`: planner, partial-scan, timing, and candidate metadata.
- `answer` and `answer_error`: present only when `answer_mode` asks for synthesis.

Offset note: plain text and raw text sources can preserve source-like character ranges.
CSV, JSON, HTML, markdown, and PDF sources may be transformed by parsers first, so spans
can report `offset_basis: "parsed"` when offsets refer to normalized parsed text rather
than raw file bytes.

## Quick Start

1. Copy `.env.example` to `.env` and fill in local values.
2. Install dependencies:

```bash
make install
```

3. Run the backend and frontend locally:

```bash
make backend-dev
make frontend-dev
```

4. Open:

- Backend docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:3000`

For Docker-based local development:

```bash
make dev
```

## Configuration

Important environment variables:

- `DATABASE_URL`: app database for saved sources and documents.
- `SUPABASE_DB_URL`: beta key-storage database when the API-key flow is enabled and
  committed.
- `OPENROUTER_API_KEY`: enables non-fallback LLM answers.
- `CORS_ORIGINS` and `CORS_ORIGIN_REGEX`: frontend origins allowed by the backend.
- `NEXT_PUBLIC_API_BASE_URL`: frontend-to-backend URL.
- `WAVER_RETRIEVER`: `bm25` by default; cortical retrieval code exists as an
  experimental path.
- `WAVER_MODEL_DIR`, `WAVER_RERANKER_MODEL`, `WAVER_RETRIEVER`, and projection settings:
  model/config knobs for retrieval experiments.
- `MAX_UPLOAD_BYTES`: upload size ceiling.

## Promise Eval

The promise eval checks Waver's raw-source contract directly: fresh support-ticket
sources should return exact spans plus semantically relevant spans quickly, without
pre-ingestion or corpus-specific vector indexing.

Fixtures live under `backend/tests/fixtures/eval/`:

- `tickets/*.json`: 150 AI-authored support-ticket cases across exact lookup, phrase
  lookup, semantic paraphrase, mixed exact/semantic, freshness, and negative cases.
- `dynamic/mutations.json`: add, delete, mutate, and reorder checks against transient
  raw sources.

Run the main evals from `backend/`:

```bash
uv run python -m scripts.run_vision_eval --profile smoke --cold-gate
uv run python -m scripts.run_vision_eval --layer exact_only
uv run python -m scripts.run_vision_eval --layer semantic_only
uv run python -m scripts.run_vision_eval --layer rerank_only
uv run python -m scripts.run_vision_eval --layer planner_only
uv run python -m scripts.run_vision_eval --dynamic dynamic/mutations.json
uv run pytest tests/test_eval_layers.py tests/test_eval_dynamic.py -v
```

`--semantic-mode real` skips cleanly when the projection model is missing. That is
intentional for local/dev machines without model artifacts.

## Verification

Backend:

```bash
make test
cd backend && uv run pytest tests/test_eval_layers.py tests/test_eval_dynamic.py -v
cd backend && uv run python -m scripts.run_vision_eval --profile smoke --cold-gate
```

Frontend:

```bash
cd frontend && npm run typecheck
cd frontend && npm run build
```

`make lint` runs backend Ruff plus the frontend lint script. If frontend lint tooling
changes with the Next.js version, `npm run typecheck` and `npm run build` are the
stronger frontend checks to keep green.

## Deployment Notes

- Docker dev runs Postgres, backend on port `8000`, and frontend on port `3000`.
- For hosted beta key creation, provision the key table before deploy using the
  migration/helper that ships with the finalized API-key PR.
- Set backend secrets for `DATABASE_URL` and, when key provisioning is enabled,
  `SUPABASE_DB_URL`.
- Set frontend `NEXT_PUBLIC_API_BASE_URL` to the deployed backend origin.
- Keep backend `CORS_ORIGINS` or `CORS_ORIGIN_REGEX` aligned with production and
  preview frontend domains.
- If the deployment health check points at API-key storage, missing key-table
  configuration should fail health before users hit key creation.

## What Still Needs To Be Done

High-priority product and engineering work:

- Finalize the beta auth/key story. Decide whether API-key generation, Supabase-backed
  key storage, and key-scoped workspaces are part of the next PR, then commit the
  route/model/migration/test set together.
- Reconcile the frontend surface. Either ship a real live search/upload workspace or
  keep the web UI clearly limited to landing, docs, key onboarding, and backend-free
  demo. The README and routes should not drift.
- Add frontend client wrappers and UI flows for live search, streaming search, upload,
  source listing, and source deletion if the frontend is meant to be more than demo
  and docs.
- Add frontend e2e coverage. Current confidence is mostly backend tests plus retrieval
  evals.
- Decide how production users get real semantic projection and reranker model artifacts.
  Local development can fall back to deterministic/heuristic behavior, but benchmark
  claims need real-model runs.
- Keep BM25/cortical retrieval experiments and the current span-executor API path
  clearly separated in docs and tests so performance claims map to the path users hit.
- Expand promise eval coverage beyond support tickets into API payloads, logs, larger
  JSON corpora, and connector transcripts.
- Harden connector support. Webhook payload search is the practical path today; Slack
  needs live-connector enablement, secret handling, rate-limit behavior, and broader
  tests before it should be marketed as production-ready.
- Add observability around latency, partial scans, connector failures, source load
  errors, and answer generation failures.
- Document production limits: max source sizes, supported media types, retention,
  workspace isolation, quotas, and expected p95 latency by corpus size.

## Repository Layout

```text
backend/
  app/
    api/          FastAPI routers
    answer/       Optional LLM answer generation
    connectors/   Webhook and Slack connector adapters
    models/       SQLAlchemy models
    parsers/      File/content parsers
    retrieval/    Planner, channels, span executor, rerankers, evals
    schemas/      API/search schemas
    services/     Search and saved-source orchestration
  tests/          Backend pytest suite and eval fixtures

frontend/
  src/app/        Next.js app routes
  src/components/ Reusable UI components
  src/lib/        Frontend API/demo client code
```

Do not put new product code in the top-level `Waver/` scaffold.

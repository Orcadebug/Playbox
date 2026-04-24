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
- Use SPS (Semantic Projection Search) as the default retriever: fuses BM25 lexical
  scoring with a sparse projection cosine at query time, zero corpus preprocessing
  required.
- Run multi-head retrieval (SPS + BM25 + exact phrase) fused via Reciprocal Rank
  Fusion (RRF), so synonyms, exact tokens, and quoted phrases all have dedicated signal.
- Apply negation/polarity feature injection so "refund denied" and "refund approved"
  land in disjoint score space rather than colliding on shared nouns.
- Refine `primary_span` to sentence level: after reranking, the best sentence inside
  the top chunk is scored and used as the highlight anchor instead of the full chunk.
- Adapt the candidate budget per query: expand the retrieval pool when score spread is
  flat (ambiguous queries), contract when a high-confidence hit is clear.
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
  Wires the SPS retriever, multi-head union, and projection singleton into the pipeline.
- `backend/app/services/sources.py`: saved-source upload, parsing, listing, deletion,
  and cache invalidation.
- `backend/app/parsers/`: format-specific parsers and parser detection.
- `backend/app/retrieval/`: retrieval stack —
  - `sps.py`: `SpsRetriever` — BM25 + sparse projection cosine fusion.
  - `retriever.py`: `MultiHeadRetriever` — RRF fusion across named retriever heads.
  - `exact_phrase.py`: `ExactPhraseRetriever` — binary substring match on query n-grams.
  - `sparse_projection.py`: `SparseProjection`, `DeterministicSemanticProjection`,
    polarity feature injection (`_augment_with_polarity`).
  - `sentence_scorer.py`: sentence-level span refinement using the loaded projection.
  - `pipeline.py`: `RetrievalPipeline` — adaptive budget, reranking, span building.
  - BM25 cache, source windows, planner, channels, span executor, and evals.
- `backend/app/answer/`: optional source-grounded answer generation through
  OpenRouter when `OPENROUTER_API_KEY` is configured.
- `backend/tests/`: pytest coverage for parsers, retrieval, search service behavior,
  source execution, planner behavior, negation/polarity, sentence scorer, adaptive
  budget, multi-head RRF, API-key work in progress, and promise evals.

Native retrieval runtime (`backend/waver_core/`, Rust via `pyo3`/`maturin`):

- `prefilter_windows` — Stage-0 byte-level trigram overlap prefilter used by the span
  executor. On `x86_64`, runtime-dispatches to AVX-512 first, then AVX2, with scalar
  fallback on non-x86 targets.
- `rrf_fuse` — Rust RRF fusion for the multi-head retriever. Gated by
  `WAVER_RUST_RRF=true`, with `WAVER_RUST_RRF_SHADOW=true` running Rust alongside
  Python and logging deltas without swapping.
- `mrl_encode` — ONNX-backed Matryoshka embedding runtime. Expects a bundle under
  `backend/models/mrl/` with `model.onnx` and `tokenizer.json` (optional tokenizer
  metadata files can sit beside them).
- `phrase_search`, `splade_encode`, `RustBm25Index` — additional exports; the Python
  retrieval layer records `using_fallback=True` + `fallback_reason` on telemetry
  whenever a Rust path cannot load or execute.

Architectural layers that are **not** wired into the main web loop yet:

- `backend/waver_ghost/` — `GhostProxy` library: fixed-memory Bloom (bit-array,
  Kirsch–Mitzenmacher double-hashing over `blake2b`) and Count-Min Sketch
  (`array("I")` with explicit `< 0xFFFFFFFF` saturation guard). Memory is O(1) in
  payload size (~128 KiB bloom + ~2 MiB CMS). Auto-`reset()` at the
  `approx_items()` threshold keeps FPR bounded across many payloads. Intended for
  edge-side zero-hit short-circuit on chunked uploads; see its README.
- `backend/services/reranker/` — optional stage-2 gRPC reranker. Preferred when
  `WAVER_RERANKER_GRPC_TARGET` is set; falls through to local ONNX, then heuristic.
  gRPC stubs are not vendored (see the service README for generation).

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
- `budget_hint`: `"auto"` (default), `"fast"` (skip expansion), or `"thorough"` (always expand candidate pool)

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
  "query": "refund was not approved",
  "top_k": 8,
  "budget_hint": "auto",
  "include_stored_sources": true,
  "answer_mode": "off",
  "raw_sources": [
    {
      "id": "scratchpad",
      "name": "notes.txt",
      "content": "First refund was approved after review.\nSecond refund was denied due to policy.\nUnrelated shipping note.",
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

Expected behaviour: rank 1 is the "denied" sentence, not the "approved" sentence.
`primary_span.text` is the single relevant sentence; `channels` shows which retriever
heads contributed.

The response is retrieval-first:

- `results`: ranked source results.
- `results[].primary_span`: the best span — sentence-level when the projection is
  available, chunk-level otherwise.
- `results[].matched_spans`: additional matched spans (chunk-level context preserved).
- `results[].channels`: which retriever heads surfaced this result (e.g. `["sps", "phrase"]`).
- `results[].channel_scores`: per-head raw scores before RRF fusion.
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
- `WAVER_RETRIEVER`: `sps` by default (BM25 + sparse projection fusion with multi-head
  RRF). Set to `bm25` for pure lexical retrieval or `cortical` for the experimental
  cortical path.
- `WAVER_SPS_ALPHA`: weight of the projection cosine score in SPS fusion (default `0.6`).
- `WAVER_MULTIHEAD`: enables RRF fusion across SPS, BM25, and exact-phrase heads
  (default `true`).
- `WAVER_RRF_K`: RRF damping constant (default `60`).
- `WAVER_RUST_RRF`: execute RRF fusion via `waver_core.rrf_fuse` when `true`
  (default `false`).
- `WAVER_RUST_RRF_SHADOW`: shadow-run Rust fusion alongside Python and log deltas
  without swapping the primary path (default `false`).
- `WAVER_ADAPTIVE_BUDGET`: expand candidate pool when score spread is flat (default `true`).
- `WAVER_BUDGET_SPREAD_THRESHOLD`: spread below which budget expands (default `0.15`).
- `WAVER_SPS_NEGATION`: polarity feature injection — separates negated from affirmed
  queries (default `true`).
- `PROJECTION_MODEL_PATH`: path to a trained `projection.npz` matrix. If absent, the
  built-in deterministic semantic projection (alias dict + stable hashing) is used — no
  model artifact required for local development.
- `WAVER_RERANKER_GRPC_TARGET`: optional remote reranker endpoint. When set, the
  backend prefers the gRPC reranker; when unset or unreachable, it falls back to the
  local ONNX cross-encoder, then to heuristic reranking.
- `WAVER_ORT_DYLIB_PATH` / `ORT_DYLIB_PATH`: ONNX Runtime shared library used by the
  Rust MRL path. Falls back to the Python `onnxruntime` package when unset.
- `WAVER_MODEL_DIR`, `WAVER_RERANKER_MODEL`, and reranker settings: cross-encoder
  reranker config knobs. When the Matryoshka runtime is provisioned, its bundle should
  live under `backend/models/mrl/` with `model.onnx` and `tokenizer.json`.
- `WAVER_TINY_MAX_BYTES` / `WAVER_MEDIUM_MAX_BYTES`: planner tier thresholds.
- `MAX_UPLOAD_BYTES`: upload size ceiling (default 50 MB).

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
    api/             FastAPI routers
    answer/          Optional LLM answer generation
    connectors/      Webhook and Slack connector adapters
    models/          SQLAlchemy models
    parsers/         File/content parsers
    retrieval/       Planner, channels, span executor, SPS/BM25/phrase heads,
                     RRF retriever, sentence refinement, rerankers, evals,
                     and rust_core.py wrapper around the native core
    schemas/         API/search schemas
    services/        Search and saved-source orchestration
  tests/             Backend pytest suite and eval fixtures
  scripts/           download_models.py, train_projection.py, run_*_eval.py
  waver_core/        Rust pyo3 core (rrf_fuse, prefilter_windows, mrl_encode,
                     phrase_search, RustBm25Index). Build: `maturin develop --release`
  waver_ghost/       GhostProxy — fixed-memory Bloom + Count-Min Sketch
                     (edge-side zero-hit short-circuit, library-only today)
  services/
    reranker/        Optional stage-2 gRPC reranker contract

frontend/
  src/app/           Next.js app routes
  src/components/    Reusable UI components
  src/lib/           Frontend API/demo client code
```

Do not put new product code in the top-level `Waver/` scaffold.

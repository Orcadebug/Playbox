# Waver

Waver is a retrieval-first API for finding the exact span that matters in messy sources
such as PDFs, CSVs, JSON, logs, HTML, markdown, pasted text, and connector payloads.
Uploading saved workspace sources is supported, but raw request sources and connectors are
first-class search inputs too.

The UI is intentionally minimal and should be treated as API-key onboarding and smoke-test
surface only. Primary product usage is API-first.

The core product is retrieval: parse sources on demand, rank candidate text, return exact
snippets/spans with source offsets, and cite where each match came from. LLM-generated
answers are optional frosting and run only when explicitly requested.

## Quick start

1. Copy `.env.example` to `.env`.
   API keys (`OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY`) are optional for the current
   retrieval-first flow.
2. Run `make dev` for Docker-based development or `make install` followed by
   `make backend-dev` and `make frontend-dev` in separate terminals.
3. Open `http://localhost:8000/docs` for API docs. `http://localhost:3000` is for API-key
   onboarding and quick stream smoke checks.

## Current scope

This repository contains a runnable MVP for:

- Searching saved workspace sources, inline raw sources, and transient connector payloads
- Uploading files or pasted text when content should be saved for later searches
- Parsing common document formats
- Running a query-time span execution engine with planner + multi-channel first pass + reranking
- Returning ranked results with `primary_span`, `matched_spans`, source offsets, metadata,
  and citation labels
- Skipping persistent chunk/index prep for raw and connector searches
- Using BM25 cache only as an optimization for stored-only searches
- Progressive SSE retrieval phases (`sources_loaded`, `exact_results`, `proxy_results`,
  `reranked_results`, `done`) plus optional `answer_delta`
- A lightweight Next.js UI for API-key onboarding and stream verification
- Webhook connector payload search, plus Slack scaffolding gated by live connector config

## Current status

- Active search endpoints:
  - `POST /api/v1/search`
  - `POST /api/v1/search/stream`
- Search accepts:
  - `raw_sources`
  - `connector_configs`
  - `include_stored_sources`
  - `answer_mode`
- Current search response shape is retrieval-first:
  - `query`
  - `answer`
  - `answer_error`
  - `results`
  - `sources`
  - `source_errors`
  - `execution`
- Result objects include compatibility fields plus span-first fields:
  - `primary_span`
  - `matched_spans`
  - `source_origin`
  - `metadata.phase`
  - `metadata.channels`
- LLM answer generation exists under `backend/app/answer` and is only called when
  `answer_mode` is `"llm"`.

## Ticket promise eval

The promise eval checks Waver's raw-source contract directly: fresh support-ticket
sources should return exact spans plus semantically relevant spans quickly, without
pre-ingestion or corpus-specific vector indexing. The new eval fixtures live under
`backend/tests/fixtures/eval/`, separate from the older retrieval eval fixtures.

- Static corpus: `backend/tests/fixtures/eval/tickets/*.json` contains 150
  AI-authored support-ticket cases across exact lookups, phrase lookups, semantic
  paraphrases, mixed exact/semantic queries, freshness cases, and negatives.
- Dynamic corpus: `backend/tests/fixtures/eval/dynamic/mutations.json` covers
  immediate add, delete, mutate, and reorder behavior against transient raw sources.
- Layer runners isolate exact, semantic proxy, reranking, and planner behavior through
  `backend/app/retrieval/eval_layers.py`.
- The cold-corpus gate verifies raw eval runs do not warm or depend on BM25 caches or
  projection model writes.

Run the promise eval from `backend/`:

```bash
uv run python -m scripts.run_vision_eval --profile smoke --cold-gate
uv run python -m scripts.run_vision_eval --layer exact_only
uv run python -m scripts.run_vision_eval --layer semantic_only
uv run python -m scripts.run_vision_eval --dynamic dynamic/mutations.json
uv run pytest tests/test_eval_layers.py tests/test_eval_dynamic.py -v
```

## Search request example

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

## Verification notes

- `make test` should pass the backend test suite.
- `cd frontend && npm run lint`, `npm run typecheck`, and `npm run build` should pass.
- `make lint` currently runs repo-wide Ruff and may fail on pre-existing lint debt in
  untouched backend files.

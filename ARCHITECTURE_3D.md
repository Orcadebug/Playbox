# Waver 3D Architecture Map

Waver is a retrieval-first system. The first-class path is search over saved workspace
sources, inline raw sources, and transient connector payloads. Upload is still useful, but
only as a way to save content for later searches. The product surface is exact retrieval:
ranked spans, source offsets, and citations. LLM answers are optional and explicitly
requested.

UI note: the web UI is a thin onboarding/smoke surface for API key setup and stream
verification. The backend API is the primary surface.

---

## Layer 1: User Interaction

```
API Consumer
  |
  |-- /api/v1/search
  |-- /api/v1/search/stream
  `-- /api/v1/upload

Minimal Browser UI
  |
  |-- API-key onboarding
  |-- stream smoke harness
  `-- optional demo interactions
  |
  `-- not the primary product surface
```

The API is the primary workflow. The UI is intentionally lightweight.

---

## Layer 2: API Boundary

```
POST /api/v1/search
POST /api/v1/search/stream
  |
  |-- query
  |-- top_k
  |-- workspace_id
  |-- source_ids
  |-- include_stored_sources
  |-- raw_sources[]
  |-- connector_configs[]
  `-- answer_mode: "off" | "llm"
```

Search responses are retrieval-first:

```
{
  query,
  answer,
  answer_error,
  results: [
    {
      content,
      snippet,
      score,
      source_name,
      source_origin,
      primary_span,
      matched_spans,
      metadata,
      citation_label
    }
  ],
  sources,
  source_errors,
  execution
}
```

`/search/stream` emits progressive typed events:
`sources_loaded`, `exact_results`, `proxy_results`, `reranked_results`, optional
`answer_delta`, then `done` or `error`.

`POST /api/v1/upload` persists saved sources. It is not required before search.

---

## Layer 3: Source Loading

```
SearchService
  |
  |-- stored provider
  |     `-- loads Source + Document rows for workspace_id/source_ids
  |
  |-- raw provider
  |     `-- converts raw_sources[] to transient SearchDocument objects
  |
  `-- connector provider
        |-- webhook: transient request documents
        `-- slack: live fetch only when ENABLE_LIVE_CONNECTORS is enabled
```

Important behavior:

- Raw and connector sources are never written to `Source` or `Document`.
- Connector metadata is sanitized so obvious secret fields do not leak into results.
- Connector failures are returned in `source_errors` when possible; retrieval still returns
  successful source results.

---

## Layer 4: Parsing And Windowing

```
SearchDocument
  |
  |-- ParserDetector
  |-- format parser
  |     |-- plaintext
  |     |-- markdown
  |     |-- csv
  |     |-- json
  |     |-- html
  |     `-- pdf
  |
  `-- source executor
        |-- SourceRecord extraction from parsed documents
        |-- SourceWindow generation (line/record scoped)
        |-- source_start/source_end exact character offsets
        `-- neighboring window metadata
```

Windows are first-pass retrieval units. The public result is a span payload, not a window
object.

---

## Layer 5: Retrieval

```
SpanExecutor
  |
  |-- planner
  |     |-- tiny / medium / huge tier
  |     |-- scan_limit, candidate_limit, rerank_limit
  |     `-- partial marker
  |-- exact channel
  |     |-- QueryTrie token/phrase spans
  |     |-- substring matches
  |     `-- regex("...") literals
  |-- proxy channel
  |     `-- sparse projection score (NullProjection fallback)
  |-- structure channel
  |     `-- metadata/location/neighbor/source-origin boosts
  |
  |-- rerank shortlist (existing Reranker seam)
  `-- build SearchResult span payloads
```

BM25 cache policy:

- Stored-only searches may use the in-memory BM25 cache.
- Raw, connector, and mixed searches build ephemeral indexes and skip the cache.
- Upload/delete invalidates the stored workspace cache.

---

## Layer 6: Span Payloads

Each result exposes:

```
primary_span:
  text: matched text or context text
  snippet: surrounding snippet
  source_start: absolute source character offset
  source_end: absolute source character offset
  snippet_start: absolute snippet start offset
  snippet_end: absolute snippet end offset
  highlights: offsets inside snippet plus absolute source offsets
  location: page/row/line/section metadata when available

matched_spans:
  additional exact or context spans

source_origin:
  stored | raw | connector
```

If a scored candidate has no exact lexical hit, Waver returns a context span with empty
`highlights`. That keeps retrieval useful even when ranking came from semantic or fallback
signals.

---

## Layer 7: Optional Answer Generation

```
answer_mode = "off"
  `-- default; return retrieval only

answer_mode = "llm"
  |
  |-- AnswerGenerator
  |-- build prompt from retrieved span context
  |-- call OpenRouter if configured
  `-- return answer or answer_error beside retrieval results
```

Answer errors do not discard retrieval results. The source spans are the product; answers are
secondary synthesis.

---

## Layer 8: Persistence

```
Source table
  |-- id
  |-- workspace_id
  |-- source_type
  |-- name
  |-- media_type
  |-- parser_name
  `-- source_metadata

Document table
  |-- id
  |-- source_id
  |-- title
  |-- content
  |-- order_index
  `-- document_metadata
```

Only saved uploads/pastes go into these tables. Raw request sources and connector payloads
remain request-scoped.

---

## Complete Search Cycle

```
API Client
  |
  |-- sends /api/v1/search or /api/v1/search/stream
  v
POST /api/v1/search/stream
  |
  v
SearchService loads stored + raw + connector SearchDocuments
  |
  v
SpanExecutor parses, windows, plans, runs channels, reranks
  |
  v
Service emits phase events (sources/exact/proxy/reranked/answer deltas/done)
  |
  v
Client reads final retrieval payload with spans and execution metadata
```

---

## Complete Save Cycle

```
Browser /upload
  |
  |-- file upload or pasted text
  v
POST /api/v1/upload
  |
  v
SourceService parses content
  |
  v
Source + Document rows are inserted
  |
  v
Stored-source BM25 cache is invalidated
  |
  v
Saved source appears in /upload and can be included from /search
```

---

## Architecture Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| Request -> Service -> Domain | API routes, services, retrieval/parsers | Keep HTTP concerns separate from retrieval logic |
| Provider fan-in | `SearchService` | Merge stored, raw, and connector sources into one retrieval input |
| Registry | parsers and connectors | Add formats/services without hardcoding call sites |
| Pipeline | `SpanExecutor` + services | Compose parsing, windowing, planning, channels, reranking, and span building |
| Transient indexing | raw/connector searches | Avoid making persistent prep foundational |
| Optional synthesis | `answer_mode` | Keep LLM answers secondary to retrieval |

---

## Performance Notes

| Component | Optimization | Scope |
|-----------|--------------|-------|
| BM25 cache | Reuse stored-only workspace index | Saved sources only |
| Chunking | Generator path for larger parsed document sets | Query-time parsing |
| Reranker | ONNX cross-encoder with heuristic fallback | Candidate reranking |
| Cortical retriever | Query trie + projection + diffusion + gating | Alternative retrieval mode |
| SSE endpoint | Streams response payload | Frontend responsiveness |

---

## Deployment Topology

```
Vercel frontend
  |
  v
Fly.io FastAPI backend
  |
  v
Neon Postgres / local SQLite
```

Local development can also run the full stack with Docker via `make dev`.

---

## Summary

1. **Retrieve first**: search saved, raw, and connector sources from `/search`.
2. **Span first**: return exact source spans and offsets, not just opaque chunks.
3. **Persist optionally**: use `/upload` when content should be saved across searches.
4. **Cache opportunistically**: cache stored-only BM25 indexes; keep raw/connector ephemeral.
5. **Synthesize optionally**: run LLM answers only with `answer_mode="llm"`.

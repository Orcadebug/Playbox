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
  |     |-- tiny / medium / huge tier (WAVER_TINY_MAX_BYTES, WAVER_MEDIUM_MAX_BYTES)
  |     |-- scan_limit, candidate_limit, rerank_limit
  |     `-- partial marker
  |
  |-- Stage-0 prefilter (waver_core.prefilter_windows)
  |     `-- byte-level trigram overlap; AVX-512 -> AVX2 -> scalar
  |
  |-- parallel retrieval heads (MultiHeadRetriever in retriever.py)
  |     |-- sps    (sps.py: BM25 + sparse projection cosine, alpha=WAVER_SPS_ALPHA)
  |     |          |-- polarity feature injection when WAVER_SPS_NEGATION=true
  |     |          `-- DeterministicSemanticProjection fallback (no model artifact)
  |     |-- bm25   (bm25.py: BM25Okapi; cache used only for stored-only)
  |     `-- phrase (exact_phrase.py: substring / n-gram matches)
  |
  |-- RRF fusion (score = sum 1/(rrf_k + rank + 1), rrf_k=WAVER_RRF_K)
  |     |-- Python default
  |     `-- waver_core.rrf_fuse when WAVER_RUST_RRF=true
  |        (WAVER_RUST_RRF_SHADOW=true runs both and logs deltas)
  |
  |-- adaptive budget (pipeline.py)
  |     `-- expand candidate pool when score spread < WAVER_BUDGET_SPREAD_THRESHOLD
  |
  |-- rerank shortlist
  |     |-- remote gRPC (WAVER_RERANKER_GRPC_TARGET) ->
  |     |-- local ONNX cross-encoder ->
  |     `-- heuristic fallback
  |
  |-- sentence-level refinement (sentence_scorer.py)
  |     `-- anchor primary_span to best sentence inside top chunk
  |
  `-- build SearchResult span payloads (channels + channel_scores per result)
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

## Layer 9: Native Runtime (`backend/waver_core/`)

Rust core compiled with `maturin` / `pyo3`, imported as `waver_core`:

```
waver_core
  |
  |-- rrf_fuse(head_results, top_k, rrf_k)    # RRF fusion, gated by WAVER_RUST_RRF
  |-- prefilter_windows(query, windows, top_k) # AVX-512 / AVX2 / scalar trigram overlap
  |-- phrase_search(phrases, haystacks, top_k) # substring phrase matcher
  |-- mrl_encode(model_path, texts, dim, ...)  # ONNX Matryoshka embedding
  |-- splade_encode(...)                       # reserved hook; Python fallback stays
  `-- RustBm25Index                            # alt BM25 index
```

Python wrappers live in `app/retrieval/rust_core.py` and record
`using_fallback=True` + `fallback_reason` on telemetry when the Rust path cannot
load or execute.

---

## Layer 10: Edge Short-Circuit (`backend/waver_ghost/`)

Library-only today; not yet wired into the request path.

```
GhostProxy
  |
  |-- Bloom: bytearray bit-array + k hashes via Kirsch-Mitzenmacher over blake2b
  |-- CMS:   array("I") depth x width, explicit <0xFFFFFFFF saturation guard
  |-- fixed memory: ~128 KiB bloom + ~2 MiB CMS, regardless of payload size
  |-- approx_items() via -m/k * ln(1 - bits_set/m)
  `-- auto reset() when approx_items() exceeds saturation_threshold (default 150k)
```

Intended use: cheap zero-hit short-circuit for chunked uploads before the main
retrieval stack is touched.

---

## Layer 11: Optional Stage-2 Reranker Service (`backend/services/reranker/`)

Optional gRPC seam. `WAVER_RERANKER_GRPC_TARGET` points at a remote reranker; the
backend falls through to local ONNX, then heuristic, recording the tier on
retrieval telemetry. gRPC stubs are not vendored and must be generated locally
(`grpc_tools.protoc`) for the remote path to activate.

---

## Performance Notes

| Component | Optimization | Scope |
|-----------|--------------|-------|
| BM25 cache | Reuse stored-only workspace index | Saved sources only |
| Stage-0 prefilter | AVX-512 / AVX2 / scalar trigram overlap | Candidate pruning before scoring |
| Rust RRF fusion | `waver_core.rrf_fuse` behind `WAVER_RUST_RRF` | Multi-head fusion |
| Chunking | Generator path for larger parsed document sets | Query-time parsing |
| Reranker | Remote gRPC → local ONNX cross-encoder → heuristic | Candidate reranking |
| Adaptive budget | Expand pool on flat score spread | Ambiguous queries |
| Sentence refinement | Projection-scored sentence anchor | primary_span precision |
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
3. **Fuse heads**: RRF across SPS (BM25 + projection), BM25, and exact phrase.
4. **Persist optionally**: use `/upload` when content should be saved across searches.
5. **Cache opportunistically**: cache stored-only BM25 indexes; keep raw/connector ephemeral.
6. **Native when available**: Rust `waver_core` accelerates RRF, prefilter, and MRL when present; Python fallback is always live.
7. **Synthesize optionally**: run LLM answers only with `answer_mode="llm"`.

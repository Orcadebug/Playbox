# waver_core

Rust core for retrieval primitives, built with `pyo3` + `maturin` and imported from
Python as `import waver_core`.

## Exports

| Symbol | Purpose |
|--------|---------|
| `RustBm25Index` | Rust BM25 index (alt to `bm25s` path); built from tokenized docs, ranked via `query`. |
| `rrf_fuse(head_results, top_k, rrf_k)` | Reciprocal Rank Fusion across `sps` / `bm25` / `phrase` heads. Returns per-doc fused score + channels + per-head scores + bm25 score passthrough. Gated by `WAVER_RUST_RRF` (opt-in), with `WAVER_RUST_RRF_SHADOW` for side-by-side parity logging against the Python implementation. |
| `prefilter_windows(query, windows, top_k)` | Stage-0 byte-level UTF-8 trigram overlap prefilter. On `x86_64` runtime-dispatches to AVX-512, then AVX2; scalar fallback elsewhere. Ranking matches the Python fallback exactly: unique overlap count, descending overlap, ascending `window_id`. |
| `phrase_search(phrases, haystacks, top_k)` | Substring phrase match over `(id, text)` haystacks; returns `(id, score)` per hit. |
| `mrl_encode(model_path, texts, dim, is_query)` | ONNX-backed Matryoshka embedding runtime. Bundle lives at `backend/models/mrl/` with `model.onnx` + `tokenizer.json`. |
| `splade_encode(model_path, texts, dim, is_query)` | Reserved SPLADE hook — Python callers keep their fallback path when the artifact is missing or inference fails. |

Python wrappers live at `backend/app/retrieval/rust_core.py`; they record
`using_fallback=True` / `fallback_reason=…` on the retrieval telemetry when the Rust
path cannot load or execute.

## Build and install locally

```bash
cd backend/waver_core
maturin develop --release
```

After installation the backend can call the exported helpers directly (`import
waver_core`; `waver_core.rrf_fuse`, `waver_core.prefilter_windows`,
`waver_core.mrl_encode`, `waver_core.phrase_search`, `waver_core.RustBm25Index`, …).

## Runtime notes

- ONNX Runtime shared library is loaded dynamically. `waver_core` checks
  `WAVER_ORT_DYLIB_PATH`, then `ORT_DYLIB_PATH`, then the Python `onnxruntime`
  package installed in the active environment.
- If the MRL runtime cannot load its artifacts or execute inference, the Rust call
  raises; the Python retrieval layer records
  `using_fallback=True, fallback_reason="mrl_runtime_failed"` and continues with the
  deterministic fallback.
- `rrf_fuse` is behind two env flags: `WAVER_RUST_RRF=true` swaps it in for the
  Python fusion step; `WAVER_RUST_RRF_SHADOW=true` runs both and logs deltas without
  swapping.

## Source layout

```
src/
├── lib.rs         pymodule registration + rrf_fuse
├── bm25.rs        RustBm25Index
├── mrl.rs         ONNX Matryoshka runtime
├── phrase.rs      substring phrase matcher
├── semantic.rs    splade_encode stub
└── simd_ngram.rs  AVX-512 / AVX2 / scalar trigram prefilter
```

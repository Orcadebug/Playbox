# waver_core

Stage-1 Rust core for retrieval primitives.

Current exports:

- `prefilter_windows(query, windows, top_k)`: exact lowercase UTF-8 byte n-gram overlap
  prefilter used by Stage-0 candidate pruning. On `x86_64`, the trigram scan runtime-dispatches
  to AVX-512 first, then AVX2; other targets use the scalar fallback. Ranking semantics match
  the Python fallback exactly: unique overlap count, descending overlap, ascending `window_id`.
- `mrl_encode(model_path, texts, dim, is_query)`: ONNX-backed Matryoshka embedding runtime.
  The model bundle is expected to live alongside the ONNX file, typically under
  `backend/models/mrl/`, with required files `model.onnx` and `tokenizer.json`.
- `splade_encode(...)`: reserved hook; Python callers still keep their existing fallback path
  when runtime artifacts are missing or inference fails.

## Build and install locally

```bash
cd backend/waver_core
maturin develop --release
```

After installation, the backend can call the exported Rust helpers directly:

- `waver_core.rrf_fuse`
- `waver_core.prefilter_windows`
- `waver_core.mrl_encode`

## Runtime Notes

- The ONNX Runtime shared library is loaded dynamically. `waver_core` first checks
  `WAVER_ORT_DYLIB_PATH`, then `ORT_DYLIB_PATH`, then falls back to the Python
  `onnxruntime` package if it is installed in the active environment.
- If the MRL runtime cannot load its artifacts or execute inference, the Rust call raises
  and the Python retrieval layer records `using_fallback=True` with
  `fallback_reason="mrl_runtime_failed"`.

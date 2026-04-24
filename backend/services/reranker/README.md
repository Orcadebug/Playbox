# Remote Reranker

Optional gRPC service for stage-2 reranking.

The contract is defined in `reranker.proto`:

- request: `{query, top_k, candidates[{id, text}]}`
- response: `{scored_candidates[{id, score}]}`

The application prefers this service when `WAVER_RERANKER_GRPC_TARGET` is set, then
falls back to the local ONNX cross-encoder, then to heuristic reranking. Fallback
reasons are recorded in retrieval telemetry so the caller can see which tier served
the request.

## Stubs

Client-side gRPC stubs are not vendored. Generate them into
`backend/services/reranker/generated/` before the remote path can activate:

```bash
cd backend
uv run python -m grpc_tools.protoc \
  -I services/reranker \
  --python_out=services/reranker/generated \
  --grpc_python_out=services/reranker/generated \
  services/reranker/reranker.proto
```

When the stubs are absent, `client.py` logs a warning once at startup
(`reranker gRPC stubs not generated; remote reranker disabled`) and the call path
transparently drops to ONNX/heuristic.

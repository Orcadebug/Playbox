# Remote Reranker

Optional gRPC service for stage-2 reranking.

The contract is defined in `reranker.proto`:

- request: `{query, top_k, candidates[{id, text}]}`
- response: `{scored_candidates[{id, score}]}`

The application prefers this service when `WAVER_RERANKER_GRPC_TARGET` is set,
then falls back to local ONNX, then to heuristic reranking.

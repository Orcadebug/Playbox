# Repository Guidelines

## Project Structure & Module Organization
This repo is a two-part MVP:

- `backend/`: FastAPI service. Main code lives in `backend/app/` (`api/`, `services/`, `retrieval/`, `parsers/`, `models/`, `schemas/`, `answer/`, `connectors/`).
- `backend/tests/`: Pytest suite for parsers, retrieval, span executor, planner, ghost proxy, remote reranker, and eval harnesses.
- `backend/waver_core/`: Rust `pyo3` core (`rrf_fuse`, `prefilter_windows`, `mrl_encode`, `phrase_search`, `RustBm25Index`). Build with `maturin develop --release`.
- `backend/waver_ghost/`: `GhostProxy` library — fixed-memory Bloom + Count-Min Sketch for edge-side zero-hit short-circuit (not yet wired into request path).
- `backend/services/reranker/`: optional stage-2 gRPC reranker service contract.
- `backend/scripts/`: `download_models.py`, `train_projection.py`, `run_retrieval_eval.py`, `run_vision_eval.py`.
- `frontend/`: Next.js 15 + React 19 app. Routes are in `frontend/src/app/`; reusable UI is in `frontend/src/components/`; API client code is in `frontend/src/lib/`.
- Root dev tooling lives in `Makefile`, `.env.example`, and `docker-compose.dev.yml`.

Use `backend/` and `frontend/` for new work. The top-level `Waver/` directory is an empty scaffold and should not receive new code.

## Product Architecture Notes
Waver is retrieval-first:

- Raw request sources and connector configs are first-class search inputs.
- Upload/store is saved-source management, not a required search setup step.
- Search results should center exact snippets/spans (`primary_span`, `matched_spans`) and source offsets.
- Persistent chunk/index prep is optional; raw and connector searches should remain transient.
- BM25 cache is an optimization for stored-only searches, not a product dependency.
- Default retrieval path is multi-head: `sps` (BM25 + sparse projection), `bm25`, and `phrase` heads fused via RRF. Sentence-level span refinement anchors `primary_span` to the best sentence inside the top chunk.
- Native acceleration (`waver_core` Rust core: SIMD trigram prefilter, Rust RRF, ONNX MRL encoder) is opt-in via `WAVER_RUST_RRF` / feature gates; Python fallback is always live and records `using_fallback` on telemetry.
- LLM answers are opt-in via `answer_mode="llm"` and should never replace source retrieval results.

## Build, Test, and Development Commands
- `make dev`: Start full Docker dev stack (Postgres, backend, frontend).
- `make backend-dev`: Run FastAPI locally with reload via `uv`.
- `make frontend-dev`: Run Next.js locally on port 3000.
- `make test`: Run backend tests (`pytest`).
- `make eval-smoke` / `make eval-regression`: Retrieval eval profiles.
- `make lint`: Run backend Ruff checks and frontend Next lint.
- `make format`: Format backend with Ruff; attempts frontend format if script exists.
- `cd frontend && npm run typecheck`: Run strict TypeScript checks.
- `cd frontend && npm run build`: Production build validation.
- `cd backend/waver_core && maturin develop --release`: Build the native Rust core locally (optional; Python fallback works without it).

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints, snake_case modules/functions, PascalCase classes.
- Lint/format: Ruff (`line-length = 100`, rules `E,F,I,UP,B`).
- TypeScript/React: strict mode enabled; PascalCase for components (`SearchBar.tsx`), camelCase for helpers/vars, route files named `page.tsx` and `layout.tsx`.
- Use the `@/*` import alias in frontend when importing from `src`.

## Testing Guidelines
- Framework: `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"` — no decorator needed). Backend only currently.
- Location/pattern: `backend/tests/test_*.py`; keep test names descriptive (`test_<behavior>()`). DB tests use `sqlite+aiosqlite:///:memory:`.
- Existing coverage spans parsers, retrieval pipeline, span executor, planner, source executor, ephemeral (raw/connector) API path, eval layers and dynamic eval, remote reranker fallback, and `GhostProxy` bloom/CMS. Add or update tests for parser, retrieval, API, raw-source, connector, span, cache, rerank, and answer-mode behavior when touching backend logic.
- Eval fixtures under `backend/tests/fixtures/eval/` (tickets + dynamic mutations) are AI-authored — guard against label leakage; avoid “hand-labeled” phrasing.
- For frontend changes, at minimum run `npm run lint` and `npm run typecheck` until UI tests are introduced.

## Commit & Pull Request Guidelines
Use Conventional Commits — established convention on this repo:

- `feat(search): add streaming token events`
- `feat(retrieval): add rust-backed rrf behind feature flags`
- `fix(ghost): replace fake bloom/cms with fixed-memory sketches`
- `chore(backend): resolve ruff lint debt`
- `docs(readme): describe product architecture and roadmap`

PRs should include: concise summary, linked issue (if available), commands run (`make test`, `make lint`, `uv run pytest tests/test_<area>.py`), and screenshots/GIFs for UI changes.

## Security & Configuration Tips
- Copy `.env.example` to `.env` for local development; never commit real secrets.
- Keep `ANTHROPIC_API_KEY` and database credentials local only.
- Confirm `CORS_ORIGINS` and `NEXT_PUBLIC_API_BASE_URL` match your runtime environment.

# Repository Guidelines

## Project Structure & Module Organization
This repo is a two-part MVP:

- `backend/`: FastAPI service. Main code lives in `backend/app/` (`api/`, `services/`, `retrieval/`, `parsers/`, `models/`, `schemas/`).
- `backend/tests/`: Pytest suite for parser and retrieval behavior.
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
- LLM answers are opt-in via `answer_mode="llm"` and should never replace source retrieval results.

## Build, Test, and Development Commands
- `make dev`: Start full Docker dev stack (Postgres, backend, frontend).
- `make backend-dev`: Run FastAPI locally with reload via `uv`.
- `make frontend-dev`: Run Next.js locally on port 3000.
- `make test`: Run backend tests (`pytest`).
- `make lint`: Run backend Ruff checks and frontend Next lint.
- `make format`: Format backend with Ruff; attempts frontend format if script exists.
- `cd frontend && npm run typecheck`: Run strict TypeScript checks.
- `cd frontend && npm run build`: Production build validation.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints, snake_case modules/functions, PascalCase classes.
- Lint/format: Ruff (`line-length = 100`, rules `E,F,I,UP,B`).
- TypeScript/React: strict mode enabled; PascalCase for components (`SearchBar.tsx`), camelCase for helpers/vars, route files named `page.tsx` and `layout.tsx`.
- Use the `@/*` import alias in frontend when importing from `src`.

## Testing Guidelines
- Framework: `pytest` + `pytest-asyncio` (backend only currently).
- Location/pattern: `backend/tests/test_*.py`; keep test names descriptive (`test_<behavior>()`).
- Add or update tests for parser, retrieval, API, raw-source, connector, span, cache, and answer-mode behavior when touching backend logic.
- For frontend changes, at minimum run `npm run lint` and `npm run typecheck` until UI tests are introduced.

## Commit & Pull Request Guidelines
This branch currently has no commit history, so no established convention can be inferred yet. Use Conventional Commit style going forward, e.g.:

- `feat(search): add streaming token events`
- `fix(parser): handle malformed csv rows`

PRs should include: concise summary, linked issue (if available), commands run (`make test`, `make lint`, etc.), and screenshots/GIFs for UI changes.

## Security & Configuration Tips
- Copy `.env.example` to `.env` for local development; never commit real secrets.
- Keep `ANTHROPIC_API_KEY` and database credentials local only.
- Confirm `CORS_ORIGINS` and `NEXT_PUBLIC_API_BASE_URL` match your runtime environment.

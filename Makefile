SHELL := /bin/zsh

.PHONY: dev dev-local backend-dev frontend-dev test lint format install eval-smoke eval-regression

dev:
	docker compose -f docker-compose.dev.yml up --build

dev-local:
	$(MAKE) backend-dev

backend-dev:
	cd backend && uv sync && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend-dev:
	cd frontend && npm install && npm run dev

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check app tests
	cd frontend && npm run lint

format:
	cd backend && uv run ruff format app tests
	cd frontend && npm run format || true

install:
	cd backend && uv sync
	cd frontend && npm install

eval-smoke:
	cd backend && uv run python scripts/run_retrieval_eval.py --profile smoke --semantic-mode deterministic

eval-regression:
	cd backend && uv run python scripts/run_retrieval_eval.py --profile regression --semantic-mode deterministic

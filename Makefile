SHELL := /bin/zsh

.PHONY: dev dev-local backend-dev frontend-dev test lint format install eval-smoke eval-regression latency-gate cleanup-corpora export-mrl train-projection artifact-manifest artifacts-local artifacts-kaggle

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
	cd backend && uv run ruff check app scripts tests
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

latency-gate:
	cd backend && uv run python scripts/run_latency_gate.py --output .reports/latency-gate.json

cleanup-corpora:
	cd backend && uv run python scripts/cleanup_corpora.py --delete-expired

export-mrl:
	cd backend && uv run --with torch --with transformers --with "optimum[onnxruntime]" python scripts/export_mrl_onnx.py --model-id mixedbread-ai/mxbai-embed-large-v1 --output-dir models/mrl

train-projection:
	cd backend && uv run python scripts/train_projection.py --teacher mrl --mrl-model models/mrl/model.onnx --out models/projection.npz

artifact-manifest:
	cd backend && uv run python scripts/write_artifact_manifest.py --output models/artifacts.json

artifacts-local:
	$(MAKE) export-mrl
	cd backend && uv run python scripts/download_models.py --profile production --quantize
	$(MAKE) train-projection
	$(MAKE) latency-gate
	$(MAKE) artifact-manifest

artifacts-kaggle:
	cd backend && uv run --with kaggle python scripts/build_artifacts_kaggle.py

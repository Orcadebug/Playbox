# Waver

Waver is a web app and API for instant search over messy data sources such as PDFs, CSVs,
JSON, logs, HTML, markdown, and pasted text. It parses uploaded content into documents,
runs just-in-time retrieval, and returns ranked passages with citation labels.

## Quick start

1. Copy `.env.example` to `.env`.
   API keys (`OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY`) are optional for the current
   retrieval-first flow.
2. Run `make dev` for Docker-based development or `make install` followed by `make backend-dev` and `make frontend-dev` in separate terminals.
3. Open `http://localhost:3000` for the UI and `http://localhost:8000/docs` for the API docs.

## Current scope

This repository contains a runnable MVP for:

- Uploading files or pasted text
- Parsing common document formats
- Running ephemeral retrieval (BM25 + reranking) over stored parsed documents
- Returning ranked results with snippets, source metadata, and citation labels
- A Next.js frontend for upload, search, and demo flows (with backend-offline fallback)
- Connector scaffolding (Webhook/Slack) with live fetching disabled by default

## Current status

- Active search endpoints:
  - `POST /api/v1/search`
  - `POST /api/v1/search/stream`
- Current search response shape is retrieval-first:
  - `query`
  - `results`
  - `sources`
- LLM answer-generation code exists under `backend/app/answer`, but it is not currently wired into the active search response path.

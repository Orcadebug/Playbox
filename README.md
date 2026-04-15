# Waver

Waver is a web app and API for instant search over messy data sources such as PDFs, CSVs, JSON, logs, HTML, markdown, and pasted text. It parses uploaded content into documents, runs just-in-time retrieval, and optionally synthesizes answers with citations.

## Quick start

1. Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` if you want LLM answers.
2. Run `make dev` for Docker-based development or `make install` followed by `make backend-dev` and `make frontend-dev` in separate terminals.
3. Open `http://localhost:3000` for the UI and `http://localhost:8000/docs` for the API docs.

## Current scope

This repository contains a runnable MVP for:

- Uploading files or pasted text
- Parsing common document formats
- Running ephemeral retrieval over stored parsed documents
- Returning ranked results plus a citation-oriented answer
- A Next.js frontend for upload, search, and demo flows

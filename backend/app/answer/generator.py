from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence

import httpx

from app.answer.prompts import build_answer_messages
from app.config import get_settings

_log = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class AnswerGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(self, query: str, results: Sequence[dict]) -> tuple[dict | None, str | None]:
        if not results:
            return None, None

        if not self.settings.openrouter_api_key:
            return self._fallback_answer(query, results), None

        try:
            return await self._openrouter_answer(query, results), None
        except Exception as exc:  # pragma: no cover - network/credential errors
            _log.error("Answer generation failed", exc_info=exc)
            return self._fallback_answer(query, results), "Answer generation unavailable"

    async def stream_generate(self, query: str, results: Sequence[dict]) -> AsyncIterator[str]:
        """Yield answer tokens as they stream in from the LLM.

        Falls back to splitting the pre-generated fallback answer word-by-word
        when no API key is configured or the API call fails.
        """
        if not results:
            return

        if not self.settings.openrouter_api_key:
            answer = self._fallback_answer(query, results)
            for word in answer["markdown"].split():
                yield word + " "
            return

        try:
            async for token in self._openrouter_stream(query, results):
                yield token
        except Exception as exc:
            _log.error("Answer streaming failed (%s) — emitting fallback", exc)
            answer = self._fallback_answer(query, results)
            for word in answer["markdown"].split():
                yield word + " "

    async def _openrouter_answer(self, query: str, results: Sequence[dict]) -> dict:
        messages = build_answer_messages(query, results)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key or ''}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-3.5-sonnet",
                    "max_tokens": 600,
                    "messages": messages,
                },
            )
            response.raise_for_status()
            payload = response.json()

        text = payload["choices"][0]["message"]["content"].strip()
        return {"markdown": text, "confidence": "medium"}

    async def _openrouter_stream(self, query: str, results: Sequence[dict]) -> AsyncIterator[str]:
        """Stream tokens from OpenRouter SSE endpoint."""
        messages = build_answer_messages(query, results)
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key or ''}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-3.5-sonnet",
                    "max_tokens": 600,
                    "stream": True,
                    "messages": messages,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        payload = json.loads(chunk)
                        delta = payload["choices"][0]["delta"].get("content") or ""
                        if delta:
                            yield delta
                    except (KeyError, IndexError, json.JSONDecodeError):
                        pass

    def _fallback_answer(self, query: str, results: Sequence[dict]) -> dict:
        top = results[:3]
        bullets = []
        for index, result in enumerate(top, start=1):
            snippet = result.get("snippet") or result["content"][:220]
            bullets.append(f"- {snippet.strip()} [{index}]")
        markdown = (
            f"Search results relevant to **{query}**:\n\n"
            + "\n".join(bullets)
            + "\n\nThis fallback answer is extractive because no LLM API key is configured."
        )
        return {"markdown": markdown, "confidence": "low"}

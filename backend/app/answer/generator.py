from collections.abc import Sequence

import httpx

from app.answer.prompts import build_answer_messages
from app.config import get_settings


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
            return self._fallback_answer(query, results), str(exc)

    async def _openrouter_answer(self, query: str, results: Sequence[dict]) -> dict:
        messages = build_answer_messages(query, results)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
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


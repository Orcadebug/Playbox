from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.connectors.base import Connector

if TYPE_CHECKING:
    from app.schemas.search import SearchDocument


class WebhookConnector(Connector):
    @property
    def connector_id(self) -> str:
        return "webhook"

    async def test_connection(self, credentials: dict[str, Any]) -> bool:
        return True

    async def fetch(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Return documents embedded directly in the config payload."""
        return config.get("documents", [])

    def to_search_documents(self, raw_results: list[dict[str, Any]]) -> list[SearchDocument]:
        from app.schemas.search import SearchDocument  # noqa: PLC0415

        docs: list[SearchDocument] = []
        for item in raw_results:
            content = item.get("content", "")
            if not content:
                continue
            if isinstance(content, str):
                content_bytes: bytes = content.encode("utf-8")
            else:
                content_bytes = bytes(content)
            metadata: dict[str, Any] = {k: v for k, v in item.items() if k != "content"}
            metadata.setdefault("source_type", "webhook")
            docs.append(
                SearchDocument(
                    file_name=item.get("name", "webhook-document.txt"),
                    content=content_bytes,
                    media_type=item.get("media_type", "text/plain"),
                    metadata=metadata,
                )
            )
        return docs

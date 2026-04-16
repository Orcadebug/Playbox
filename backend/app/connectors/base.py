from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.schemas.search import SearchDocument


class Connector(ABC):
    """Base class for all data source connectors."""

    @property
    @abstractmethod
    def connector_id(self) -> str:
        """Unique identifier for this connector type (e.g. ``"slack"``)."""
        ...

    @abstractmethod
    async def test_connection(self, credentials: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch raw records from the external source."""
        raise NotImplementedError

    @abstractmethod
    def to_search_documents(self, raw_results: list[dict[str, Any]]) -> list[SearchDocument]:
        """Transform raw connector records into SearchDocument objects."""
        raise NotImplementedError

    async def fetch_as_search_documents(self, config: dict[str, Any]) -> list[SearchDocument]:
        """Convenience: fetch then convert. Respects ``max_documents`` if set in config."""
        raw = await self.fetch(config)
        max_docs = config.get("max_documents")
        if max_docs is not None:
            raw = raw[:max_docs]
        return self.to_search_documents(raw)

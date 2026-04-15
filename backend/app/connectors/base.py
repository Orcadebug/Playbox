from abc import ABC, abstractmethod
from typing import Any


class Connector(ABC):
    @abstractmethod
    async def test_connection(self, credentials: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError


from typing import Any

from app.connectors.base import Connector


class WebhookConnector(Connector):
    async def test_connection(self, credentials: dict[str, Any]) -> bool:
        return True

    async def fetch(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return config.get("documents", [])


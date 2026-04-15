from typing import Any

from app.connectors.base import Connector


class SlackConnector(Connector):
    async def test_connection(self, credentials: dict[str, Any]) -> bool:
        token = credentials.get("bot_token")
        return bool(token)

    async def fetch(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError("Slack integration is scaffolded but not implemented in the MVP.")


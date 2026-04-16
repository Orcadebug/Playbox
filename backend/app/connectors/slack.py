from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from app.connectors.base import Connector

if TYPE_CHECKING:
    from app.schemas.search import SearchDocument

_log = logging.getLogger(__name__)

# Slack API base
_SLACK_API = "https://slack.com/api"


class SlackConnector(Connector):
    """Fetches messages from Slack channels via the Web API.

    Config keys expected in ``fetch(config)``:
    - ``bot_token`` (str): Slack bot OAuth token
    - ``channels`` (list[str] | None): channel IDs to fetch from. Defaults to all joined.
    - ``limit`` (int): max messages per channel (default 100)
    """

    @property
    def connector_id(self) -> str:
        return "slack"

    async def test_connection(self, credentials: dict[str, Any]) -> bool:
        token = credentials.get("bot_token")
        if not token:
            return False
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.post(
                    f"{_SLACK_API}/auth.test",
                    headers={"Authorization": f"Bearer {token}"},
                )
                data = resp.json()
                return bool(data.get("ok"))
            except Exception:
                return False

    async def fetch(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        token: str | None = config.get("bot_token")
        if not token:
            _log.warning("SlackConnector: no bot_token in config — returning empty")
            return []

        channel_ids: list[str] | None = config.get("channels")
        limit: int = int(config.get("limit", 100))
        headers = {"Authorization": f"Bearer {token}"}
        messages: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Discover channels if not specified
            if not channel_ids:
                channel_ids = await self._list_channels(client, headers)

            for channel_id in channel_ids[:20]:  # cap channels to avoid runaway
                channel_msgs = await self._fetch_channel(client, headers, channel_id, limit)
                messages.extend(channel_msgs)

        return messages[: config.get("max_documents", 200)]

    async def _list_channels(self, client: httpx.AsyncClient, headers: dict[str, str]) -> list[str]:
        try:
            resp = await client.get(
                f"{_SLACK_API}/conversations.list",
                headers=headers,
                params={"types": "public_channel,private_channel", "limit": 100},
            )
            data = resp.json()
            if not data.get("ok"):
                _log.warning("Slack conversations.list failed: %s", data.get("error"))
                return []
            return [ch["id"] for ch in data.get("channels", [])]
        except Exception as exc:
            _log.warning("Slack channel list failed: %s", exc)
            return []

    async def _fetch_channel(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        channel_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            resp = await client.get(
                f"{_SLACK_API}/conversations.history",
                headers=headers,
                params={"channel": channel_id, "limit": min(limit, 200)},
            )
            data = resp.json()
            if not data.get("ok"):
                _log.warning("Slack history for %s failed: %s", channel_id, data.get("error"))
                return []
            return [
                {
                    "channel_id": channel_id,
                    "text": msg.get("text", ""),
                    "user": msg.get("user", ""),
                    "timestamp": msg.get("ts", ""),
                }
                for msg in data.get("messages", [])
                if msg.get("text")
            ]
        except Exception as exc:
            _log.warning("Slack history fetch for %s failed: %s", channel_id, exc)
            return []

    def to_search_documents(self, raw_results: list[dict[str, Any]]) -> list[SearchDocument]:
        from app.schemas.search import SearchDocument  # noqa: PLC0415

        docs: list[SearchDocument] = []
        for msg in raw_results:
            text = msg.get("text", "").strip()
            if not text:
                continue
            channel_id = msg.get("channel_id", "unknown")
            docs.append(
                SearchDocument(
                    file_name=f"slack:{channel_id}",
                    content=text.encode("utf-8"),
                    media_type="text/plain",
                    metadata={
                        "source_type": "slack",
                        "channel_id": channel_id,
                        "user": msg.get("user", ""),
                        "timestamp": msg.get("timestamp", ""),
                    },
                )
            )
        return docs

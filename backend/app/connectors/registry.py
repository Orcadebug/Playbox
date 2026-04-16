"""
Connector registry — maps connector_id strings to Connector instances.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.connectors.base import Connector


class ConnectorRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Connector] = {}

    def register(self, connector_id: str, connector: Connector) -> None:
        self._registry[connector_id] = connector

    def get(self, connector_id: str) -> Connector | None:
        return self._registry.get(connector_id)

    def all_ids(self) -> list[str]:
        return list(self._registry.keys())


def _build_default_registry() -> ConnectorRegistry:
    from app.connectors.slack import SlackConnector
    from app.connectors.webhook import WebhookConnector

    registry = ConnectorRegistry()
    registry.register("slack", SlackConnector())
    registry.register("webhook", WebhookConnector())
    return registry


# Module-level singleton — import and use directly.
default_registry: ConnectorRegistry = _build_default_registry()

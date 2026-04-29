# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Webhooks communication addon plugin."""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("webhooks")


class WebhooksPlugin:
    """Stub plugin for Webhooks communication bridge.

    Generic webhook bridge — POST JSON payloads to any URL on events, alerts, or state changes
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "webhook_url": "",
            "secret": "",
            "events": "",
            "format": "",
            "enabled": False,
        }

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.webhooks"

    @property
    def name(self) -> str:
        return "Webhooks"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> set:
        return {"alert_relay", "event_bridge"}

    @property
    def healthy(self) -> bool:
        return self._running and self._config.get("enabled", False)

    def configure(self, ctx: Any) -> None:
        self._logger = ctx.logger or log
        self._event_bus = ctx.event_bus

    def start(self) -> None:
        if not self._config.get("enabled"):
            log.info("[Webhooks] Not enabled — skipping start")
            return
        self._running = True
        log.info("[Webhooks] Started (stub — no real connection)")

    def stop(self) -> None:
        self._running = False
        log.info("[Webhooks] Stopped")

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via Webhooks. Returns True on success."""
        if not self.healthy:
            return False
        log.info(f"[Webhooks] Would send: {text[:80]}")
        return True  # Stub — always succeeds

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to Webhooks."""
        return await self.send_message(
            f"[{alert.get('level', 'info').upper()}] {alert.get('title', '')} — {alert.get('message', '')}"
        )

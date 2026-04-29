# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Satellite communication addon plugin."""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("satellite")


class SatellitePlugin:
    """Stub plugin for Satellite communication bridge.

    Satellite communication bridge — Iridium, Starlink, or Inmarsat for beyond-line-of-sight operations
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "provider": "",
            "modem_port": "",
            "baud_rate": "",
            "bridge_alerts": "",
            "enabled": False,
        }

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.satellite"

    @property
    def name(self) -> str:
        return "Satellite"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> set:
        return {"alert_relay", "beyond_los"}

    @property
    def healthy(self) -> bool:
        return self._running and self._config.get("enabled", False)

    def configure(self, ctx: Any) -> None:
        self._logger = ctx.logger or log
        self._event_bus = ctx.event_bus

    def start(self) -> None:
        if not self._config.get("enabled"):
            log.info("[Satellite] Not enabled — skipping start")
            return
        self._running = True
        log.info("[Satellite] Started (stub — no real connection)")

    def stop(self) -> None:
        self._running = False
        log.info("[Satellite] Stopped")

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via Satellite. Returns True on success."""
        if not self.healthy:
            return False
        log.info(f"[Satellite] Would send: {text[:80]}")
        return True  # Stub — always succeeds

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to Satellite."""
        return await self.send_message(
            f"[{alert.get('level', 'info').upper()}] {alert.get('title', '')} — {alert.get('message', '')}"
        )

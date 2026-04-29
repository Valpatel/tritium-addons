# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Email communication addon plugin."""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("email")


class EmailPlugin:
    """Stub plugin for Email communication bridge.

    Email notification bridge — SMTP relay for alert digests and daily reports
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "smtp_host": "",
            "smtp_port": "",
            "username": "",
            "password": "",
            "from_addr": "",
            "to_addrs": "",
            "enabled": False,
        }

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.email"

    @property
    def name(self) -> str:
        return "Email"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> set:
        return {"alert_relay", "daily_digest"}

    @property
    def healthy(self) -> bool:
        return self._running and self._config.get("enabled", False)

    def configure(self, ctx: Any) -> None:
        self._logger = ctx.logger or log
        self._event_bus = ctx.event_bus

    def start(self) -> None:
        if not self._config.get("enabled"):
            log.info("[Email] Not enabled — skipping start")
            return
        self._running = True
        log.info("[Email] Started (stub — no real connection)")

    def stop(self) -> None:
        self._running = False
        log.info("[Email] Stopped")

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via Email. Returns True on success."""
        if not self.healthy:
            return False
        log.info(f"[Email] Would send: {text[:80]}")
        return True  # Stub — always succeeds

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to Email."""
        return await self.send_message(
            f"[{alert.get('level', 'info').upper()}] {alert.get('title', '')} — {alert.get('message', '')}"
        )

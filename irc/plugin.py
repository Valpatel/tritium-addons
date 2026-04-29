# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""IRC communication addon plugin."""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("irc")


class IRCPlugin:
    """Stub plugin for IRC communication bridge.

    IRC bridge — connect to IRC networks for low-bandwidth, high-reliability text comms
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "server": "",
            "port": "",
            "nick": "",
            "channel": "",
            "use_tls": "",
            "enabled": False,
        }

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.irc"

    @property
    def name(self) -> str:
        return "IRC"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> set:
        return {"chat_bridge", "alert_relay"}

    @property
    def healthy(self) -> bool:
        return self._running and self._config.get("enabled", False)

    def configure(self, ctx: Any) -> None:
        self._logger = ctx.logger or log
        self._event_bus = ctx.event_bus

    def start(self) -> None:
        if not self._config.get("enabled"):
            log.info("[IRC] Not enabled — skipping start")
            return
        self._running = True
        log.info("[IRC] Started (stub — no real connection)")

    def stop(self) -> None:
        self._running = False
        log.info("[IRC] Stopped")

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via IRC. Returns True on success."""
        if not self.healthy:
            return False
        log.info(f"[IRC] Would send: {text[:80]}")
        return True  # Stub — always succeeds

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to IRC."""
        return await self.send_message(
            f"[{alert.get('level', 'info').upper()}] {alert.get('title', '')} — {alert.get('message', '')}"
        )

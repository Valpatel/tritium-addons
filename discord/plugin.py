# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Discord communication addon plugin."""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("discord")


class DiscordPlugin:
    """Stub plugin for Discord communication bridge.

    Discord bot integration — post embeds to channels, receive commands, voice channel status
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "bot_token": "",
            "guild_id": "",
            "channel_id": "",
            "bridge_alerts": "",
            "enabled": False,
        }

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.discord"

    @property
    def name(self) -> str:
        return "Discord"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> set:
        return {"chat_bridge", "alert_relay", "voice_status"}

    @property
    def healthy(self) -> bool:
        return self._running and self._config.get("enabled", False)

    def configure(self, ctx: Any) -> None:
        self._logger = ctx.logger or log
        self._event_bus = ctx.event_bus

    def start(self) -> None:
        if not self._config.get("enabled"):
            log.info("[Discord] Not enabled — skipping start")
            return
        self._running = True
        log.info("[Discord] Started (stub — no real connection)")

    def stop(self) -> None:
        self._running = False
        log.info("[Discord] Stopped")

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via Discord. Returns True on success."""
        if not self.healthy:
            return False
        log.info(f"[Discord] Would send: {text[:80]}")
        return True  # Stub — always succeeds

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to Discord."""
        return await self.send_message(
            f"[{alert.get('level', 'info').upper()}] {alert.get('title', '')} — {alert.get('message', '')}"
        )

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Matrix communication addon plugin."""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("matrix")


class MatrixPlugin:
    """Stub plugin for Matrix communication bridge.

    Matrix/Element bridge — federated, encrypted messaging for distributed operations
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "homeserver": "",
            "user_id": "",
            "access_token": "",
            "room_id": "",
            "enabled": False,
        }

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.matrix"

    @property
    def name(self) -> str:
        return "Matrix"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> set:
        return {"chat_bridge", "alert_relay", "encrypted", "federated"}

    @property
    def healthy(self) -> bool:
        return self._running and self._config.get("enabled", False)

    def configure(self, ctx: Any) -> None:
        self._logger = ctx.logger or log
        self._event_bus = ctx.event_bus

    def start(self) -> None:
        if not self._config.get("enabled"):
            log.info("[Matrix] Not enabled — skipping start")
            return
        self._running = True
        log.info("[Matrix] Started (stub — no real connection)")

    def stop(self) -> None:
        self._running = False
        log.info("[Matrix] Stopped")

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via Matrix. Returns True on success."""
        if not self.healthy:
            return False
        log.info(f"[Matrix] Would send: {text[:80]}")
        return True  # Stub — always succeeds

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to Matrix."""
        return await self.send_message(
            f"[{alert.get('level', 'info').upper()}] {alert.get('title', '')} — {alert.get('message', '')}"
        )

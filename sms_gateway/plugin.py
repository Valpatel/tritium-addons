# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SMS Gateway communication addon plugin."""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("sms_gateway")


class SMSGatewayPlugin:
    """Stub plugin for SMS Gateway communication bridge.

    SMS/text message gateway via Twilio or local GSM modem — send alerts to phone numbers
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "provider": "",
            "account_sid": "",
            "auth_token": "",
            "from_number": "",
            "alert_numbers": "",
            "enabled": False,
        }

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.sms_gateway"

    @property
    def name(self) -> str:
        return "SMS Gateway"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> set:
        return {"alert_relay", "sms"}

    @property
    def healthy(self) -> bool:
        return self._running and self._config.get("enabled", False)

    def configure(self, ctx: Any) -> None:
        self._logger = ctx.logger or log
        self._event_bus = ctx.event_bus

    def start(self) -> None:
        if not self._config.get("enabled"):
            log.info("[SMS Gateway] Not enabled — skipping start")
            return
        self._running = True
        log.info("[SMS Gateway] Started (stub — no real connection)")

    def stop(self) -> None:
        self._running = False
        log.info("[SMS Gateway] Stopped")

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via SMS Gateway. Returns True on success."""
        if not self.healthy:
            return False
        log.info(f"[SMS Gateway] Would send: {text[:80]}")
        return True  # Stub — always succeeds

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to SMS Gateway."""
        return await self.send_message(
            f"[{alert.get('level', 'info').upper()}] {alert.get('title', '')} — {alert.get('message', '')}"
        )

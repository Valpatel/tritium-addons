# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Webhooks communication addon plugin."""

from __future__ import annotations
import logging
import os
import threading
from typing import Any

log = logging.getLogger("webhooks")


class WebhooksPlugin:
    """Webhook bridge — POST JSON payloads to any URL on events / alerts.

    Configuration is read from kwargs passed to ``configure()`` first,
    then env vars (``WEBHOOKS_URL``, ``WEBHOOKS_SECRET``,
    ``WEBHOOKS_ENABLED``) so the comms dispatcher can wire it up
    without a config file.

    The plugin remains a stub at boot — a real HTTP request only fires
    when ``webhook_url`` is set AND ``enabled`` is true.  This keeps the
    "9 stubs, 1 real" invariant the dispatcher relies on.
    """

    def __init__(self) -> None:
        self._running = False
        self._config = {
            "webhook_url": os.environ.get("WEBHOOKS_URL", ""),
            "secret": os.environ.get("WEBHOOKS_SECRET", ""),
            "events": "",
            "format": "json",
            "enabled": os.environ.get("WEBHOOKS_ENABLED", "").lower() in ("1", "true", "yes"),
            "timeout_s": 5.0,
        }
        self._send_count = 0
        self._send_failures = 0
        self._last_status_code: int | None = None
        self._lock = threading.Lock()

    @property
    def plugin_id(self) -> str:
        return "tritium.comms.webhooks"

    @property
    def name(self) -> str:
        return "Webhooks"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def capabilities(self) -> set:
        return {"alert_relay", "event_bridge"}

    @property
    def healthy(self) -> bool:
        return (
            self._running
            and bool(self._config.get("enabled", False))
            and bool(self._config.get("webhook_url", ""))
        )

    def configure(self, ctx: Any = None, **overrides) -> None:
        """Configure with optional addon context and direct overrides."""
        if ctx is not None:
            self._logger = getattr(ctx, "logger", None) or log
            self._event_bus = getattr(ctx, "event_bus", None)
        else:
            self._logger = log
            self._event_bus = None

        for key, value in overrides.items():
            if key in self._config:
                self._config[key] = value

    def start(self) -> None:
        """Start the plugin.  Real HTTP requests gated on ``enabled``."""
        self._running = True
        if not self._config.get("enabled"):
            log.info("[Webhooks] Started (disabled — set enabled=True to relay)")
            return
        if not self._config.get("webhook_url"):
            log.info("[Webhooks] Started (no URL — set webhook_url to relay)")
            return
        log.info("[Webhooks] Started → %s", self._config.get("webhook_url"))

    def stop(self) -> None:
        self._running = False
        log.info("[Webhooks] Stopped")

    def send_message_sync(self, text: str, payload: dict | None = None) -> bool:
        """Synchronous variant of :meth:`send_message` for thread callers.

        Returns ``True`` on a 2xx HTTP response, ``False`` otherwise.
        Used by the comms dispatcher which runs in a non-async listener
        thread and cannot easily await.
        """
        if not self.healthy:
            return False
        url = self._config["webhook_url"]
        body = dict(payload) if payload else {}
        body.setdefault("text", text)
        if self._config.get("secret"):
            body.setdefault("secret", self._config["secret"])
        try:
            import httpx
        except ImportError:
            log.warning("[Webhooks] httpx not available — cannot relay")
            return False
        try:
            resp = httpx.post(
                url,
                json=body,
                timeout=float(self._config.get("timeout_s", 5.0)),
            )
            with self._lock:
                self._send_count += 1
                self._last_status_code = resp.status_code
                if not (200 <= resp.status_code < 300):
                    self._send_failures += 1
            log.info("[Webhooks] POST %s → %d", url, resp.status_code)
            return 200 <= resp.status_code < 300
        except Exception as exc:
            with self._lock:
                self._send_count += 1
                self._send_failures += 1
            log.warning("[Webhooks] POST failed: %s", exc)
            return False

    async def send_message(self, text: str, **kwargs) -> bool:
        """Send a message via webhook.  Returns True on 2xx response."""
        # Lift to a thread so we don't block the event loop on slow URLs.
        import asyncio
        loop = asyncio.get_event_loop()
        payload = kwargs.get("payload") or kwargs
        return await loop.run_in_executor(
            None, self.send_message_sync, text, payload
        )

    async def relay_alert(self, alert: dict) -> bool:
        """Relay a Tritium alert to a webhook URL."""
        text = (
            f"[{alert.get('severity', alert.get('level', 'info')).upper()}] "
            f"{alert.get('title', '')} — {alert.get('message', '')}"
        )
        return await self.send_message(text, payload=alert)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "send_count": self._send_count,
                "send_failures": self._send_failures,
                "last_status_code": self._last_status_code,
                "running": self._running,
                "enabled": bool(self._config.get("enabled", False)),
                "url_set": bool(self._config.get("webhook_url", "")),
            }

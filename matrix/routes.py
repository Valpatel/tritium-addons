# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Matrix communication addon API routes."""

from __future__ import annotations
from fastapi import APIRouter
from typing import Any


def create_router(plugin: Any) -> APIRouter:
    router = APIRouter(prefix="/api/comms/matrix", tags=["matrix"])

    @router.get("/config")
    async def get_config():
        return plugin._config

    @router.put("/config")
    async def set_config(body: dict):
        for key, value in body.items():
            if key in plugin._config:
                plugin._config[key] = value
        return plugin._config

    @router.get("/status")
    async def get_status():
        return {
            "plugin": "matrix",
            "name": "Matrix",
            "healthy": plugin.healthy,
            "enabled": plugin._config.get("enabled", False),
            "running": plugin._running,
        }

    @router.post("/test")
    async def test_connection():
        """Test the Matrix connection with a ping message."""
        ok = await plugin.send_message("[Tritium] Connection test")
        return {"success": ok, "message": "Test message sent" if ok else "Not configured or not enabled"}

    @router.post("/send")
    async def send_message(body: dict):
        """Send a message via Matrix."""
        text = body.get("text", "")
        if not text:
            return {"error": "Missing 'text' field"}
        ok = await plugin.send_message(text)
        return {"sent": ok}

    return router

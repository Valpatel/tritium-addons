# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Regression: addon registration must NOT block on a slow serial connect.

A serial device that is present but never completes its Meshtastic config
exchange (e.g. a phone/other CDC-ACM board enumerating as /dev/ttyACM0) used
to freeze lifespan startup — register() awaited connect_serial inline, so the
server bind waited the full ~60s x attempts.  The auto-connect now runs on a
background task; register() must return promptly and the connect finishes (or
fails + retries) off the boot path.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from meshtastic_addon import MeshtasticAddon
from meshtastic_addon.connection import ConnectionManager


def _fake_serial_port():
    return [{
        "port": "/dev/fake-mesh0",
        "device_id": "mesh-fake0",
        "transport": "serial",
        "vid": "303a", "pid": "1001",
        "description": "fake meshtastic", "manufacturer": "test",
        "serial_number": "TEST0", "meshtastic_match": True,
    }]


def _ctx():
    ctx = MagicMock()
    ctx.target_tracker = MagicMock()
    ctx.event_bus = MagicMock()
    ctx.mqtt_client = MagicMock(spec=[])  # no subscribe attr
    ctx.site_id = "test-site"
    ctx.router_handler = MagicMock()
    ctx.router_handler.include_router = MagicMock()
    ctx.get_state = MagicMock(return_value=None)
    ctx.set_state = MagicMock()
    return ctx


@pytest.mark.asyncio
@patch("meshtastic_addon.detect_meshtastic_ports", side_effect=_fake_serial_port)
async def test_register_returns_fast_despite_slow_connect(_detect):
    """register() returns in well under a second even if connect blocks ~60s."""
    connect_started = asyncio.Event()
    release = asyncio.Event()

    async def slow_connect(self, port, timeout=60, retries=1, noNodes=False):
        connect_started.set()
        # Simulate a device that opens but never completes config exchange.
        try:
            await asyncio.wait_for(release.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass

    addon = MeshtasticAddon()
    ctx = _ctx()
    with patch.object(ConnectionManager, "connect_serial", slow_connect):
        t0 = time.monotonic()
        await addon.register(context=ctx)
        elapsed = time.monotonic() - t0

        # The inline-await bug made this ~60-122s.  Non-blocking => sub-second.
        assert elapsed < 3.0, f"register() blocked for {elapsed:.1f}s"

        # The connect is scheduled and running in the background, not awaited.
        assert addon._connect_task is not None
        assert not addon._connect_task.done()

        # Give the loop a tick so the background task actually starts.
        await asyncio.sleep(0.05)
        assert connect_started.is_set(), "background auto-connect never ran"

        # Clean shutdown must cancel the in-flight connect promptly.
        t1 = time.monotonic()
        await addon.unregister(context=ctx)
        shutdown = time.monotonic() - t1
        assert shutdown < 3.0, f"unregister() blocked for {shutdown:.1f}s"

    # No orphaned task after unregister.
    assert addon._connect_task is None


@pytest.mark.asyncio
async def test_background_connect_success_path(tmp_path):
    """A quick successful connect flips registry to CONNECTED + registers callbacks."""
    # Use a REAL file as the port so the poll loop's exists() check passes
    # (otherwise it correctly marks the radio unplugged).
    fake_port = tmp_path / "ttyACM-mesh"
    fake_port.write_bytes(b"")
    port_str = str(fake_port)

    def _detect_real_port():
        return [{
            "port": port_str, "device_id": "mesh-real0", "transport": "serial",
            "vid": "303a", "pid": "1001", "description": "fake meshtastic",
            "manufacturer": "test", "serial_number": "TEST0",
            "meshtastic_match": True,
        }]

    async def quick_connect(self, port, timeout=60, retries=1, noNodes=False):
        # Simulate a live interface arriving quickly.
        self.interface = MagicMock()
        self.is_connected = True
        self.transport_type = "serial"
        self.port = port

    addon = MeshtasticAddon()
    ctx = _ctx()
    with patch("meshtastic_addon.detect_meshtastic_ports", side_effect=_detect_real_port), \
            patch.object(ConnectionManager, "connect_serial", quick_connect):
        await addon.register(context=ctx)
        # Let the background task run to completion.
        await asyncio.wait_for(addon._connect_task, timeout=5)

        # Registry reflects a successful connect and the primary alias is live.
        from tritium_lib.sdk import DeviceState
        assert addon.registry.get_device("mesh-real0").state == DeviceState.CONNECTED
        assert addon.connection is not None and addon.connection.is_connected

        await addon.unregister(context=ctx)

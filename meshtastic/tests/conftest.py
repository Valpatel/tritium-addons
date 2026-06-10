# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared fixtures for the Meshtastic addon test suite.

Keeps BLE tests hermetic: ``ConnectionManager.connect_ble`` scans with
``bleak.BleakScanner`` before connecting (the scan-first BLEDevice fix) and
falls back to ``DirectBLEConnection`` when the meshtastic library fails.
With bleak installed, both paths would otherwise drive the host's real BLE
adapter from unit tests — 8 s real scans per test, nondeterministic results,
and the ability to connect to actual nearby radios.  Stub them out globally.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# The address every BLE test in this suite connects to.  The fake scan
# result matches it by address and, for auto-discover (""), by name.
FAKE_BLE_ADDRESS = "AA:BB:CC:DD:EE:FF"


@pytest.fixture(autouse=True)
def _no_real_ble_radio():
    """Never touch a real BLE adapter from unit tests."""
    fake_dev = SimpleNamespace(address=FAKE_BLE_ADDRESS, name="Meshtastic_TEST")
    fake_adv = SimpleNamespace(rssi=-60)
    fake_scan = AsyncMock(return_value={fake_dev.address: (fake_dev, fake_adv)})

    with patch("bleak.BleakScanner.discover", new=fake_scan), \
         patch("meshtastic_addon.ble_direct.DirectBLEConnection.connect",
               new=AsyncMock(return_value=False)), \
         patch("meshtastic_addon.ble_direct.DirectBLEConnection.disconnect",
               new=AsyncMock(return_value=None)):
        yield

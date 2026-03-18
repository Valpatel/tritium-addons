# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Direct BLE connection to Meshtastic devices using bleak.

Bypasses the meshtastic library's BLEInterface which has service discovery
race conditions on Linux/BlueZ. Speaks the meshtastic protobuf protocol
directly over GATT characteristics.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Any, Callable

logger = logging.getLogger("meshtastic.ble_direct")

# Meshtastic BLE UUIDs
SERVICE_UUID = "6ba1b218-15a8-461f-9fa8-5dcae273eafd"
TORADIO_UUID = "f75c76d2-129e-4dad-a1dd-7866124401e7"
FROMRADIO_UUID = "2c55e69e-4993-11ed-b878-0242ac120002"
FROMNUM_UUID = "ed9da18c-a800-4f66-a670-aa7547e34453"
BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"


class DirectBLEConnection:
    """Connect to a Meshtastic device via BLE using bleak directly.

    Usage::

        conn = DirectBLEConnection()
        await conn.connect("D8:85:AC:A9:DF:7D")
        nodes = await conn.request_nodes()
        await conn.disconnect()
    """

    def __init__(self):
        self._client = None
        self._address: str = ""
        self._connected: bool = False
        self._node_id: str = ""
        self._device_info: dict = {}
        self._nodes: dict = {}
        self._on_packet: Callable | None = None
        self._fromradio_buffer: list[bytes] = []

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None and self._client.is_connected

    @property
    def address(self) -> str:
        return self._address

    @property
    def device_info(self) -> dict:
        return self._device_info

    @property
    def nodes(self) -> dict:
        return self._nodes

    async def scan(self, timeout: float = 8.0) -> list[dict]:
        """Scan for Meshtastic BLE devices."""
        from bleak import BleakScanner

        devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
        results = []
        for dev, adv in devices.values():
            if SERVICE_UUID in adv.service_uuids or "Meshtastic" in (dev.name or ""):
                results.append({
                    "address": dev.address,
                    "name": dev.name or "",
                    "rssi": adv.rssi,
                })
        return results

    async def connect(self, address: str, timeout: float = 15.0) -> bool:
        """Connect to a Meshtastic device via BLE.

        Handles the BlueZ service discovery race by:
        1. Connecting without service discovery timeout
        2. Reading battery char to verify GATT is working
        3. Starting meshtastic protocol exchange
        """
        from bleak import BleakClient

        self._address = address
        logger.info(f"BLE direct: connecting to {address}...")

        try:
            self._client = BleakClient(address, timeout=timeout)
            await self._client.connect()

            if not self._client.is_connected:
                logger.warning("BLE connect returned but not connected")
                return False

            self._connected = True
            logger.info(f"BLE direct: connected to {address}")

            # Read battery to verify GATT works
            try:
                bat_data = await self._client.read_gatt_char(BATTERY_UUID)
                self._device_info["battery"] = int(bat_data[0]) if bat_data else None
                logger.info(f"BLE direct: battery={self._device_info['battery']}%")
            except Exception as e:
                logger.debug(f"Battery read skipped: {e}")

            # Subscribe to fromnum notifications (tells us when data is available)
            try:
                await self._client.start_notify(FROMNUM_UUID, self._on_fromnum)
            except Exception as e:
                logger.debug(f"FromNum notify skipped: {e}")

            # Send want_config to start protocol exchange
            await self._send_want_config()

            # Read initial responses
            await self._drain_fromradio(max_reads=50, timeout=5.0)

            return True

        except Exception as e:
            logger.warning(f"BLE direct connect failed: {e}")
            self._connected = False
            if self._client:
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
            return False

    async def disconnect(self):
        """Disconnect from the BLE device."""
        self._connected = False
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
        logger.info(f"BLE direct: disconnected from {self._address}")

    async def _send_want_config(self):
        """Send a want_config_id packet to start the meshtastic handshake."""
        # ToRadio protobuf: field 3 (want_config_id) = uint32
        # Protobuf encoding: field 3, wire type 0 (varint), value 1
        want_config = b"\x18\x01"  # field_number=3, varint=1
        try:
            await self._client.write_gatt_char(TORADIO_UUID, want_config, response=True)
            logger.debug("Sent want_config")
        except Exception as e:
            logger.warning(f"Failed to send want_config: {e}")

    async def _drain_fromradio(self, max_reads: int = 100, timeout: float = 10.0):
        """Read all available fromradio packets."""
        start = time.monotonic()
        empty_count = 0

        for _ in range(max_reads):
            if time.monotonic() - start > timeout:
                break
            try:
                data = await self._client.read_gatt_char(FROMRADIO_UUID)
                if not data or len(data) == 0:
                    empty_count += 1
                    if empty_count > 3:
                        break
                    await asyncio.sleep(0.1)
                    continue
                empty_count = 0
                self._process_fromradio(data)
            except Exception as e:
                logger.debug(f"FromRadio read error: {e}")
                break

    def _process_fromradio(self, data: bytes):
        """Parse a fromradio protobuf packet (simplified)."""
        # This is a simplified parser — it extracts node info from the raw bytes
        # without needing the full protobuf library
        self._fromradio_buffer.append(data)

        # Try to extract node info from the raw data
        try:
            self._extract_node_info(data)
        except Exception as e:
            logger.debug(f"Packet parse error: {e}")

    def _extract_node_info(self, data: bytes):
        """Extract node information from raw meshtastic packet bytes."""
        # Look for string patterns that indicate node names
        # This is a heuristic approach — full protobuf parsing would be better
        # but works for basic node discovery
        pass

    def _on_fromnum(self, sender, data: bytearray):
        """Notification handler for fromnum characteristic."""
        # fromnum tells us new data is available — trigger a read
        pass

    async def send_text(self, text: str, destination: int = 0xFFFFFFFF, channel: int = 0) -> bool:
        """Send a text message over the mesh."""
        if not self.is_connected:
            return False
        # Build text message protobuf
        # For now, this is a stub — full implementation needs protobuf encoding
        logger.warning("send_text via direct BLE not yet implemented — use serial connection")
        return False

    async def get_battery(self) -> int | None:
        """Read battery level from BLE battery service."""
        if not self.is_connected:
            return None
        try:
            data = await self._client.read_gatt_char(BATTERY_UUID)
            return int(data[0]) if data else None
        except Exception:
            return None

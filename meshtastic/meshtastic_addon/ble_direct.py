# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Direct BLE connection to Meshtastic devices using bleak.

Bypasses the meshtastic library's BLEInterface which has service discovery
race conditions on Linux/BlueZ. Uses the proven approach: scan first to get
a BLEDevice object, then pass it to BleakClient (not an address string).

Usage::

    conn = DirectBLEConnection()
    if await conn.connect("10:20:BA:33:FF:39"):
        nodes = conn.nodes
        print(f"Got {len(nodes)} nodes, our position: {conn.my_position}")
    await conn.disconnect()
"""

from __future__ import annotations

import asyncio
import logging
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
    """Connect to a Meshtastic device via BLE and pull node data.

    This implementation:
    1. Scans for BLE devices to get a BLEDevice object (critical for BlueZ)
    2. Connects using the BLEDevice object (not address string)
    3. Sends want_config to trigger the meshtastic protocol exchange
    4. Drains all fromradio packets and parses with meshtastic protobuf
    5. Extracts nodes, positions, device info
    """

    def __init__(self):
        self._client = None
        self._address: str = ""
        self._connected: bool = False
        self._my_node_num: int | None = None
        self._nodes: dict[str, dict] = {}
        self._device_info: dict = {}
        self._battery: int | None = None
        self._firmware: str = ""

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None and self._client.is_connected

    @property
    def nodes(self) -> dict[str, dict]:
        return self._nodes

    @property
    def my_position(self) -> tuple[float, float] | None:
        """Our node's GPS position as (lat, lng), or None."""
        if self._my_node_num:
            nid = f"!{self._my_node_num:08x}"
            node = self._nodes.get(nid)
            if node and node.get("lat") and node["lat"] != 0:
                return (node["lat"], node["lng"])
        return None

    @property
    def device_info(self) -> dict:
        return self._device_info

    @property
    def battery(self) -> int | None:
        return self._battery

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
                    "_ble_device": dev,  # Keep BLEDevice for connection
                })
        return results

    async def connect(self, address: str = "", timeout: float = 15.0, retries: int = 3) -> bool:
        """Connect to a Meshtastic device via BLE.

        Args:
            address: BLE address (e.g., "10:20:BA:33:FF:39"). If empty, connects to first found.
            timeout: Connection timeout per attempt.
            retries: Number of connection attempts.
        """
        from bleak import BleakClient

        # Step 1: Scan to get BLEDevice (critical for BlueZ reliability)
        logger.info(f"BLE: scanning for {address or 'any Meshtastic device'}...")
        devices = await self.scan(timeout=8.0)
        if not devices:
            logger.warning("BLE: no Meshtastic devices found")
            return False

        target = None
        if address:
            addr_upper = address.upper()
            for d in devices:
                if d["address"].upper() == addr_upper:
                    target = d
                    break
        if not target:
            # Pick strongest signal
            target = max(devices, key=lambda d: d["rssi"])

        ble_device = target["_ble_device"]
        self._address = target["address"]
        logger.info(f"BLE: connecting to {target['name']} ({target['address']}) RSSI={target['rssi']}")

        # Step 2: Connect with retries (BLE is inherently unreliable)
        for attempt in range(1, retries + 1):
            try:
                self._client = BleakClient(ble_device, timeout=timeout)
                await self._client.connect()
                if self._client.is_connected:
                    self._connected = True
                    logger.info(f"BLE: connected on attempt {attempt}")
                    break
            except Exception as e:
                logger.warning(f"BLE: attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    await asyncio.sleep(2)

        if not self._connected:
            logger.warning("BLE: all connection attempts failed")
            return False

        # Step 3: Read battery
        try:
            bat_data = await self._client.read_gatt_char(BATTERY_UUID)
            self._battery = int(bat_data[0]) if bat_data else None
        except Exception:
            pass

        # Step 4: Subscribe to notifications and send want_config
        notify_event = asyncio.Event()
        try:
            await self._client.start_notify(FROMNUM_UUID, lambda s, d: notify_event.set())
        except Exception:
            pass

        try:
            await self._client.write_gatt_char(TORADIO_UUID, b"\x18\x01", response=True)
            logger.debug("BLE: want_config sent")
        except Exception as e:
            logger.warning(f"BLE: want_config failed: {e}")
            return True  # Still connected, just can't pull config

        # Step 5: Wait for notification then drain all packets
        try:
            await asyncio.wait_for(notify_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        packets = []
        empty_count = 0
        for _ in range(500):
            try:
                data = await self._client.read_gatt_char(FROMRADIO_UUID)
                if not data or len(data) == 0:
                    empty_count += 1
                    if empty_count > 5 and packets:
                        break
                    await asyncio.sleep(0.05)
                    continue
                empty_count = 0
                packets.append(data)
            except Exception:
                await asyncio.sleep(0.1)

        logger.info(f"BLE: received {len(packets)} packets")

        # Step 6: Parse with meshtastic protobuf
        self._parse_packets(packets)

        return True

    async def disconnect(self):
        """Disconnect from the BLE device."""
        self._connected = False
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
        logger.info(f"BLE: disconnected from {self._address}")

    def _parse_packets(self, packets: list[bytes]):
        """Parse meshtastic fromradio packets into nodes and device info."""
        try:
            from meshtastic.protobuf import mesh_pb2
        except ImportError:
            logger.warning("meshtastic protobuf not available — can't parse packets")
            return

        for pkt in packets:
            try:
                fr = mesh_pb2.FromRadio()
                fr.ParseFromString(pkt)
                variant = fr.WhichOneof("payload_variant")

                if variant == "my_info":
                    self._my_node_num = fr.my_info.my_node_num
                    self._device_info["my_node_num"] = fr.my_info.my_node_num
                    self._device_info["node_id"] = f"!{fr.my_info.my_node_num:08x}"

                elif variant == "node_info":
                    ni = fr.node_info
                    u = ni.user
                    p = ni.position
                    dm = ni.device_metrics
                    nid = f"!{ni.num:08x}"

                    self._nodes[nid] = {
                        "node_id": nid,
                        "num": ni.num,
                        "long_name": u.long_name,
                        "short_name": u.short_name,
                        "hw_model": str(u.hw_model),
                        "lat": p.latitude_i / 1e7 if p.latitude_i else None,
                        "lng": p.longitude_i / 1e7 if p.longitude_i else None,
                        "altitude": p.altitude if p.altitude else None,
                        "battery": dm.battery_level if dm.battery_level else None,
                        "voltage": dm.voltage if dm.voltage else None,
                        "snr": ni.snr if ni.snr else None,
                        "last_heard": ni.last_heard if ni.last_heard else int(time.time()),
                        "channel_util": dm.channel_utilization if dm.channel_utilization else None,
                        "air_util": dm.air_util_tx if dm.air_util_tx else None,
                    }

                elif variant == "metadata":
                    m = fr.metadata
                    self._firmware = m.firmware_version
                    self._device_info["firmware"] = m.firmware_version
                    self._device_info["hw_model"] = str(m.hw_model)

                elif variant == "config_complete_id":
                    logger.debug("BLE: config exchange complete")

            except Exception as e:
                logger.debug(f"BLE: packet parse error: {e}")

        gps_count = sum(1 for n in self._nodes.values() if n.get("lat") and n["lat"] != 0)
        logger.info(f"BLE: parsed {len(self._nodes)} nodes ({gps_count} with GPS)")

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Comprehensive API endpoint tests for the Meshtastic addon.

Tests every GET and POST endpoint using FastAPI TestClient with mocked
ConnectionManager, NodeManager, DeviceManager, and MessageBridge.
Realistic fake data based on T-LoRa Pager: node_id=!ba33ff38,
hw_model=T_LORA_PAGER, firmware=2.7.19, 250 nodes.

UX Loop 2 (Add Sensor) — verifying all mesh management API endpoints work.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meshtastic_addon.connection import ConnectionManager
from meshtastic_addon.device_manager import (
    ChannelInfo,
    DeviceInfo,
    DeviceManager,
    FirmwareInfo,
    create_device_routes,
)
from meshtastic_addon.message_bridge import MessageBridge, MeshMessage
from meshtastic_addon.node_manager import NodeManager
from meshtastic_addon.router import create_router


# ---------------------------------------------------------------------------
# Realistic test data — based on T-LoRa Pager with 250 nodes
# ---------------------------------------------------------------------------

LOCAL_NODE_ID = "!ba33ff38"
LOCAL_LONG_NAME = "Tritium Base"
LOCAL_SHORT_NAME = "TRIT"
LOCAL_HW_MODEL = "T_LORA_PAGER"
LOCAL_FIRMWARE = "2.7.19"
LOCAL_MAC = "ba:33:ff:38:00:01"

SAMPLE_NODES = {}
for i in range(5):
    nid = f"!ba33ff{i:02x}"
    SAMPLE_NODES[nid] = {
        "node_id": nid,
        "num": 0xBA33FF00 + i,
        "long_name": f"Node-{i:03d}",
        "short_name": f"N{i:02d}",
        "hw_model": "T_LORA_PAGER" if i < 3 else "HELTEC_V3",
        "mac": f"ba:33:ff:{i:02x}:00:01",
        "role": "CLIENT" if i != 2 else "ROUTER",
        "last_heard": time.time() - i * 60,
        "lat": 30.2672 + i * 0.001 if i < 4 else None,
        "lng": -97.7431 + i * 0.001 if i < 4 else None,
        "altitude": 150 + i * 10 if i < 4 else None,
        "battery": 85 - i * 5,
        "voltage": 4.1 - i * 0.1,
        "snr": 10.0 - i * 2,
        "channel_util": 5.0 + i,
        "air_util": 2.0 + i * 0.5,
        "uptime": 3600 * (i + 1),
        "neighbors": [f"!ba33ff{j:02x}" for j in range(5) if j != i and j < 3],
        "neighbor_snr": {},
    }

SAMPLE_MESSAGES = [
    MeshMessage(
        sender_id="!ba33ff00",
        sender_name="Node-000",
        text="Hello mesh!",
        timestamp=time.time() - 120,
        channel=0,
        type="text",
    ),
    MeshMessage(
        sender_id="!ba33ff01",
        sender_name="Node-001",
        text="Position: 30.267200, -97.743100",
        timestamp=time.time() - 60,
        type="position",
        lat=30.2672,
        lng=-97.7431,
    ),
    MeshMessage(
        sender_id="!ba33ff02",
        sender_name="Node-002",
        text="Telemetry: bat:75%, v:3.9V",
        timestamp=time.time() - 30,
        type="telemetry",
        battery=75,
        voltage=3.9,
        channel_util=6.5,
        air_util=3.0,
    ),
]


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

def make_mock_connection(connected=True):
    """Build a mock ConnectionManager that looks connected."""
    conn = MagicMock(spec=ConnectionManager)
    conn.is_connected = connected
    conn.transport_type = "serial" if connected else "none"
    conn.port = "/dev/ttyACM0" if connected else ""
    conn.interface = MagicMock() if connected else None
    conn.device_info = {
        "node_id": LOCAL_NODE_ID,
        "long_name": LOCAL_LONG_NAME,
        "short_name": LOCAL_SHORT_NAME,
        "hw_model": LOCAL_HW_MODEL,
        "mac": LOCAL_MAC,
        "firmware": LOCAL_FIRMWARE,
    } if connected else {}
    conn.connect_serial = AsyncMock()
    conn.connect_tcp = AsyncMock()
    conn.connect_ble = AsyncMock()
    conn.connect_mqtt = AsyncMock()
    conn.disconnect = AsyncMock()
    conn.send_text = AsyncMock(return_value=True)
    return conn


def make_mock_node_manager():
    """Build a NodeManager pre-loaded with sample nodes."""
    nm = NodeManager()
    nm.nodes = dict(SAMPLE_NODES)
    return nm


def make_mock_message_bridge(connection=None, node_manager=None):
    """Build a MessageBridge with sample message history."""
    bridge = MessageBridge(connection=connection, node_manager=node_manager)
    for msg in SAMPLE_MESSAGES:
        bridge._messages.append(msg)
    bridge.messages_received = 15
    bridge.messages_sent = 3
    bridge.position_reports = 42
    bridge.telemetry_reports = 100
    bridge.send_text = AsyncMock(return_value=True)
    return bridge


def make_mock_device_manager(connection=None):
    """Build a DeviceManager with mocked async methods returning realistic data."""
    if connection is None:
        connection = make_mock_connection()
    dm = DeviceManager(connection)

    # Mock all async methods
    dm.get_device_info = AsyncMock(return_value=DeviceInfo(
        node_id=LOCAL_NODE_ID,
        long_name=LOCAL_LONG_NAME,
        short_name=LOCAL_SHORT_NAME,
        hw_model=LOCAL_HW_MODEL,
        mac=LOCAL_MAC,
        firmware_version=LOCAL_FIRMWARE,
        has_wifi=True,
        has_bluetooth=True,
        has_ethernet=False,
        role="CLIENT",
        reboot_count=3,
        region="US",
        modem_preset="LONG_FAST",
        num_channels=2,
        tx_power=27,
        channels=[
            ChannelInfo(index=0, name="Primary", role="PRIMARY", psk="AQ=="),
            ChannelInfo(index=1, name="Admin", role="SECONDARY", psk=""),
        ],
    ))
    dm.get_channels = AsyncMock(return_value=[
        ChannelInfo(index=0, name="Primary", role="PRIMARY", psk="AQ=="),
        ChannelInfo(index=1, name="Admin", role="SECONDARY", psk=""),
    ])
    dm.get_firmware_info = AsyncMock(return_value=FirmwareInfo(
        current_version=LOCAL_FIRMWARE,
        latest_version="2.7.20",
        update_available=True,
        hw_model=LOCAL_HW_MODEL,
        esptool_available=True,
        meshtastic_cli_available=True,
    ))
    dm.get_available_versions = AsyncMock(return_value=[
        {"version": "2.7.20", "date": "2026-03-15"},
        {"version": "2.7.19", "date": "2026-03-10"},
        {"version": "2.7.18", "date": "2026-03-01"},
    ])
    dm.get_module_config = AsyncMock(return_value={
        "telemetry": {"device_update_interval": 900, "environment_measurement_enabled": True},
        "range_test": {"enabled": False},
        "store_forward": {"enabled": False},
    })
    dm.set_owner = AsyncMock(return_value=True)
    dm.set_role = AsyncMock(return_value=True)
    dm.set_lora_config = AsyncMock(return_value=True)
    dm.set_position = AsyncMock(return_value=True)
    dm.set_wifi = AsyncMock(return_value=True)
    dm.set_bluetooth = AsyncMock(return_value=True)
    dm.set_display_config = AsyncMock(return_value=True)
    dm.set_power_config = AsyncMock(return_value=True)
    dm.set_mqtt_config = AsyncMock(return_value=True)
    dm.set_telemetry_config = AsyncMock(return_value=True)
    dm.configure_channel = AsyncMock(return_value=True)
    dm.get_channel_url = AsyncMock(return_value="https://meshtastic.org/e/#CgMSAQ")
    dm.set_channel_url = AsyncMock(return_value=True)
    dm.reboot = AsyncMock(return_value=True)
    dm.factory_reset = AsyncMock(return_value=True)
    dm.shutdown = AsyncMock(return_value=True)
    dm.detect_device = AsyncMock(return_value={
        "chip": "ESP32-S3",
        "board": "T_LORA_PAGER",
        "flash_size": "8MB",
        "firmware": LOCAL_FIRMWARE,
    })
    dm.flash_firmware = AsyncMock(return_value={"success": True, "message": "Firmware flashed"})
    dm.flash_latest = AsyncMock(return_value={"success": True, "message": "Latest firmware flashed"})
    dm.export_config = AsyncMock(return_value={
        "device": {"role": "CLIENT"},
        "lora": {"region": "US", "modem_preset": "LONG_FAST", "tx_power": 27},
        "channels": [{"index": 0, "name": "Primary", "role": "PRIMARY"}],
    })
    dm.import_config = AsyncMock(return_value={"device": True, "lora": True, "channels": True})

    return dm


# ---------------------------------------------------------------------------
# Fixture: build FastAPI app with both routers mounted
# ---------------------------------------------------------------------------

@pytest.fixture
def app_and_mocks():
    """Create a FastAPI app with mocked meshtastic addon routes."""
    conn = make_mock_connection()
    nm = make_mock_node_manager()
    bridge = make_mock_message_bridge(connection=conn, node_manager=nm)
    dm = make_mock_device_manager(connection=conn)

    app = FastAPI()
    router = create_router(conn, nm, bridge)
    device_router = create_device_routes(dm)
    app.include_router(router, prefix="/api/addons/meshtastic")
    app.include_router(device_router, prefix="/api/addons/meshtastic")

    return app, {
        "connection": conn,
        "node_manager": nm,
        "message_bridge": bridge,
        "device_manager": dm,
    }


@pytest.fixture
def client(app_and_mocks):
    """TestClient for the mocked app."""
    app, _mocks = app_and_mocks
    return TestClient(app)


@pytest.fixture
def mocks(app_and_mocks):
    """Dict of mocked components."""
    _app, m = app_and_mocks
    return m


# ===========================================================================
# Core router endpoints
# ===========================================================================


class TestStatusEndpoint:
    """GET /api/addons/meshtastic/status"""

    def test_status_connected(self, client):
        r = client.get("/api/addons/meshtastic/status")
        assert r.status_code == 200
        data = r.json()
        assert data["connected"] is True
        assert data["transport"] == "serial"
        assert data["port"] == "/dev/ttyACM0"
        assert data["node_count"] == 5
        assert "device" in data

    def test_status_has_device_info(self, client):
        r = client.get("/api/addons/meshtastic/status")
        data = r.json()
        device = data["device"]
        assert device["node_id"] == LOCAL_NODE_ID
        assert device["long_name"] == LOCAL_LONG_NAME
        assert device["hw_model"] == LOCAL_HW_MODEL


class TestPortsEndpoint:
    """GET /api/addons/meshtastic/ports"""

    def test_ports_returns_list(self, client):
        r = client.get("/api/addons/meshtastic/ports")
        assert r.status_code == 200
        data = r.json()
        assert "ports" in data
        assert "count" in data
        assert isinstance(data["ports"], list)


class TestNodesEndpoints:
    """GET /api/addons/meshtastic/nodes and /nodes/{node_id}"""

    def test_get_all_nodes(self, client):
        r = client.get("/api/addons/meshtastic/nodes")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 5
        assert len(data["nodes"]) == 5

    def test_nodes_have_expected_fields(self, client):
        r = client.get("/api/addons/meshtastic/nodes")
        node = r.json()["nodes"][0]
        for field_name in [
            "node_id", "long_name", "short_name", "hw_model",
            "lat", "lng", "battery", "voltage", "snr", "last_heard",
        ]:
            assert field_name in node, f"Missing field: {field_name}"

    def test_get_single_node_found(self, client):
        r = client.get("/api/addons/meshtastic/nodes/!ba33ff00")
        assert r.status_code == 200
        data = r.json()
        assert data["long_name"] == "Node-000"

    def test_get_single_node_not_found(self, client):
        r = client.get("/api/addons/meshtastic/nodes/!deadbeef")
        assert r.status_code == 200
        data = r.json()
        assert data.get("error") == "not_found"


class TestLinksEndpoint:
    """GET /api/addons/meshtastic/links"""

    def test_links_returns_list(self, client):
        r = client.get("/api/addons/meshtastic/links")
        assert r.status_code == 200
        data = r.json()
        assert "links" in data
        assert isinstance(data["links"], list)
        # With 5 nodes having neighbors, we should get some links
        assert len(data["links"]) > 0


class TestTargetsEndpoint:
    """GET /api/addons/meshtastic/targets"""

    def test_targets_returns_list(self, client):
        r = client.get("/api/addons/meshtastic/targets")
        assert r.status_code == 200
        data = r.json()
        assert "targets" in data
        assert len(data["targets"]) == 5

    def test_targets_have_tritium_format(self, client):
        r = client.get("/api/addons/meshtastic/targets")
        target = r.json()["targets"][0]
        assert target["target_id"].startswith("mesh_")
        assert target["source"] == "mesh"
        assert target["asset_type"] == "mesh_radio"
        assert target["alliance"] == "friendly"


class TestMessagesEndpoint:
    """GET /api/addons/meshtastic/messages"""

    def test_messages_returns_history(self, client):
        r = client.get("/api/addons/meshtastic/messages")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 3
        assert len(data["messages"]) == 3

    def test_messages_filter_by_type(self, client):
        r = client.get("/api/addons/meshtastic/messages?type=text")
        assert r.status_code == 200
        data = r.json()
        # Only text messages
        for msg in data["messages"]:
            assert msg["type"] == "text"

    def test_messages_limit(self, client):
        r = client.get("/api/addons/meshtastic/messages?limit=1")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] <= 1


class TestBridgeStatsEndpoint:
    """GET /api/addons/meshtastic/bridge/stats"""

    def test_bridge_stats(self, client):
        r = client.get("/api/addons/meshtastic/bridge/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["messages_received"] == 15
        assert data["messages_sent"] == 3
        assert data["position_reports"] == 42
        assert data["telemetry_reports"] == 100
        assert "history_size" in data


class TestNetworkStatsEndpoint:
    """GET /api/addons/meshtastic/stats"""

    def test_network_stats(self, client):
        r = client.get("/api/addons/meshtastic/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_nodes"] == 5
        assert "online_nodes" in data
        assert "with_gps" in data
        assert "routers" in data
        assert "avg_snr" in data
        assert "link_count" in data


class TestHealthEndpoint:
    """GET /api/addons/meshtastic/health"""

    def test_health_connected(self, client):
        r = client.get("/api/addons/meshtastic/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["connected"] is True
        assert data["node_count"] == 5


# ===========================================================================
# Connection lifecycle endpoints
# ===========================================================================


class TestConnectEndpoint:
    """POST /api/addons/meshtastic/connect"""

    def test_connect_serial_default(self, client, mocks):
        # The connect endpoint checks Path(serial_port).exists()
        # For testing, we use tcp transport which doesn't check paths
        r = client.post("/api/addons/meshtastic/connect", json={
            "transport": "tcp",
            "port": "192.168.1.100",
        })
        assert r.status_code == 200
        data = r.json()
        # Connection mock returns interface=MagicMock (truthy)
        assert "connected" in data

    def test_connect_mqtt(self, client, mocks):
        r = client.post("/api/addons/meshtastic/connect", json={
            "transport": "mqtt",
            "host": "mqtt.meshtastic.org",
            "mqtt_port": 1883,
            "topic": "msh/US/2/e/#",
            "username": "meshdev",
            "password": "large4cats",
        })
        assert r.status_code == 200
        mocks["connection"].connect_mqtt.assert_called_once()

    def test_connect_ble(self, client, mocks):
        r = client.post("/api/addons/meshtastic/connect", json={
            "transport": "ble",
            "address": "AA:BB:CC:DD:EE:FF",
        })
        assert r.status_code == 200
        mocks["connection"].connect_ble.assert_called_once()

    def test_connect_unsupported_transport(self, client):
        r = client.post("/api/addons/meshtastic/connect", json={
            "transport": "carrier_pigeon",
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("error") == "unsupported transport: carrier_pigeon"


class TestDisconnectEndpoint:
    """POST /api/addons/meshtastic/disconnect"""

    def test_disconnect(self, client, mocks):
        r = client.post("/api/addons/meshtastic/disconnect")
        assert r.status_code == 200
        data = r.json()
        assert data["connected"] is False
        mocks["connection"].disconnect.assert_called_once()


class TestSendEndpoint:
    """POST /api/addons/meshtastic/send"""

    def test_send_broadcast(self, client, mocks):
        r = client.post("/api/addons/meshtastic/send", json={
            "text": "Hello mesh!",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["sent"] is True
        assert data["text"] == "Hello mesh!"
        assert data["destination"] is None

    def test_send_direct(self, client, mocks):
        r = client.post("/api/addons/meshtastic/send", json={
            "text": "Direct message",
            "destination": "!ba33ff00",
            "channel": 1,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["sent"] is True
        assert data["destination"] == "!ba33ff00"
        assert data["channel"] == 1

    def test_send_empty_message(self, client):
        r = client.post("/api/addons/meshtastic/send", json={
            "text": "",
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("error") == "empty_message"


# ===========================================================================
# Device management endpoints
# ===========================================================================


class TestDeviceInfoEndpoint:
    """GET /api/addons/meshtastic/device/info"""

    def test_device_info(self, client):
        r = client.get("/api/addons/meshtastic/device/info")
        assert r.status_code == 200
        data = r.json()
        assert data["node_id"] == LOCAL_NODE_ID
        assert data["long_name"] == LOCAL_LONG_NAME
        assert data["hw_model"] == LOCAL_HW_MODEL
        assert data["firmware_version"] == LOCAL_FIRMWARE
        assert data["has_wifi"] is True
        assert data["has_bluetooth"] is True
        assert data["region"] == "US"
        assert data["modem_preset"] == "LONG_FAST"
        assert data["tx_power"] == 27
        assert len(data["channels"]) == 2


class TestDeviceChannelsEndpoint:
    """GET /api/addons/meshtastic/device/channels"""

    def test_channels(self, client):
        r = client.get("/api/addons/meshtastic/device/channels")
        assert r.status_code == 200
        data = r.json()
        assert len(data["channels"]) == 2
        assert data["channels"][0]["name"] == "Primary"
        assert data["channels"][0]["role"] == "PRIMARY"
        assert data["channels"][1]["name"] == "Admin"
        assert data["channels"][1]["role"] == "SECONDARY"


class TestFirmwareEndpoints:
    """GET /api/addons/meshtastic/device/firmware and /firmware/versions"""

    def test_firmware_info(self, client):
        r = client.get("/api/addons/meshtastic/device/firmware")
        assert r.status_code == 200
        data = r.json()
        assert data["current_version"] == LOCAL_FIRMWARE
        assert data["latest_version"] == "2.7.20"
        assert data["update_available"] is True
        assert data["hw_model"] == LOCAL_HW_MODEL
        assert data["esptool_available"] is True

    def test_firmware_versions(self, client):
        r = client.get("/api/addons/meshtastic/device/firmware/versions")
        assert r.status_code == 200
        data = r.json()
        assert len(data["versions"]) == 3
        assert data["versions"][0]["version"] == "2.7.20"


class TestModulesEndpoint:
    """GET /api/addons/meshtastic/device/modules"""

    def test_modules(self, client):
        r = client.get("/api/addons/meshtastic/device/modules")
        assert r.status_code == 200
        data = r.json()
        assert "telemetry" in data["modules"]
        assert data["modules"]["telemetry"]["device_update_interval"] == 900


class TestDetectEndpoint:
    """GET /api/addons/meshtastic/device/detect"""

    def test_detect_device(self, client):
        r = client.get("/api/addons/meshtastic/device/detect")
        assert r.status_code == 200
        data = r.json()
        assert data["chip"] == "ESP32-S3"
        assert data["board"] == "T_LORA_PAGER"
        assert data["flash_size"] == "8MB"


class TestDeviceConfigEndpoint:
    """POST /api/addons/meshtastic/device/config"""

    def test_set_owner_name(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "long_name": "New Base Name",
            "short_name": "NBN",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["results"]["owner"] is True
        mocks["device_manager"].set_owner.assert_called_once_with(
            long_name="New Base Name", short_name="NBN",
        )

    def test_set_role(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "role": "ROUTER",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["role"] is True
        mocks["device_manager"].set_role.assert_called_once_with("ROUTER")

    def test_set_lora_config(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "region": "EU_868",
            "modem_preset": "SHORT_SLOW",
            "tx_power": 20,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["lora"] is True

    def test_set_position(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "lat": 30.2672,
            "lng": -97.7431,
            "altitude": 150,
            "gps_mode": "DISABLED",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["position"] is True

    def test_set_wifi(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "wifi_enabled": True,
            "wifi_ssid": "Tritium-Net",
            "wifi_password": "s3cret!",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["wifi"] is True

    def test_set_bluetooth(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "bluetooth_enabled": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["bluetooth"] is True

    def test_set_channel(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "channel": {
                "index": 1,
                "name": "Ops",
                "psk": "random",
                "role": "SECONDARY",
                "uplink_enabled": True,
            },
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["channel"] is True

    def test_set_channel_missing_index(self, client):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "channel": {"name": "NoIndex"},
        })
        assert r.status_code == 400

    def test_set_display(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "screen_on_secs": 60,
            "flip_screen": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["display"] is True

    def test_set_power(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "is_power_saving": True,
            "on_battery_shutdown_after_secs": 3600,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["power"] is True

    def test_set_mqtt_module(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "mqtt_enabled": True,
            "mqtt_address": "mqtt.local:1883",
            "mqtt_username": "mesh",
            "mqtt_password": "pass",
            "mqtt_encryption_enabled": True,
            "mqtt_json_enabled": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["mqtt"] is True

    def test_set_telemetry_module(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "telemetry_device_interval": 300,
            "telemetry_env_enabled": True,
            "telemetry_env_interval": 600,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["results"]["telemetry"] is True

    def test_set_multiple_configs(self, client, mocks):
        """Test setting multiple config sections in one request."""
        r = client.post("/api/addons/meshtastic/device/config", json={
            "long_name": "Multi Test",
            "role": "TRACKER",
            "region": "US",
            "tx_power": 27,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["results"]["owner"] is True
        assert data["results"]["role"] is True
        assert data["results"]["lora"] is True

    def test_empty_config_rejected(self, client):
        r = client.post("/api/addons/meshtastic/device/config", json={
            "unknown_key": "whatever",
        })
        assert r.status_code == 400


class TestRebootEndpoint:
    """POST /api/addons/meshtastic/device/reboot"""

    def test_reboot(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/reboot", json={
            "delay": 10,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mocks["device_manager"].reboot.assert_called_once_with(seconds=10)

    def test_reboot_default_delay(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/reboot")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mocks["device_manager"].reboot.assert_called_once_with(seconds=5)


class TestFactoryResetEndpoint:
    """POST /api/addons/meshtastic/device/factory-reset"""

    def test_factory_reset(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/factory-reset")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mocks["device_manager"].factory_reset.assert_called_once()


class TestFlashEndpoints:
    """POST /api/addons/meshtastic/device/flash and /flash-latest"""

    def test_flash_firmware(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/flash", json={
            "firmware_path": "/tmp/firmware.bin",
            "port": "/dev/ttyACM0",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mocks["device_manager"].flash_firmware.assert_called_once()

    def test_flash_latest(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/flash-latest")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mocks["device_manager"].flash_latest.assert_called_once()


class TestChannelUrlEndpoints:
    """GET/POST /api/addons/meshtastic/device/channel-url"""

    def test_get_channel_url(self, client, mocks):
        r = client.get("/api/addons/meshtastic/device/channel-url")
        assert r.status_code == 200
        data = r.json()
        assert data["url"] == "https://meshtastic.org/e/#CgMSAQ"

    def test_set_channel_url(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/channel-url", json={
            "url": "https://meshtastic.org/e/#CgMSAQ_NEW",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mocks["device_manager"].set_channel_url.assert_called_once_with(
            "https://meshtastic.org/e/#CgMSAQ_NEW"
        )

    def test_set_channel_url_empty(self, client):
        r = client.post("/api/addons/meshtastic/device/channel-url", json={
            "url": "",
        })
        assert r.status_code == 400


# ===========================================================================
# Additional device sub-endpoints (/device/display, /device/power, etc.)
# ===========================================================================


class TestDeviceDisplayEndpoint:
    """POST /api/addons/meshtastic/device/display"""

    def test_set_display(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/display", json={
            "screen_on_secs": 120,
            "auto_screen_carousel_secs": 10,
            "flip_screen": False,
            "units": "METRIC",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True


class TestDevicePowerEndpoint:
    """POST /api/addons/meshtastic/device/power"""

    def test_set_power(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/power", json={
            "is_power_saving": False,
            "on_battery_shutdown_after_secs": 0,
        })
        assert r.status_code == 200
        assert r.json()["success"] is True


class TestDeviceMqttEndpoint:
    """POST /api/addons/meshtastic/device/mqtt"""

    def test_set_mqtt(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/mqtt", json={
            "enabled": True,
            "address": "mqtt.local:1883",
            "username": "mesh",
            "password": "pass",
            "encryption_enabled": True,
            "json_enabled": False,
        })
        assert r.status_code == 200
        assert r.json()["success"] is True


class TestDeviceTelemetryEndpoint:
    """POST /api/addons/meshtastic/device/telemetry"""

    def test_set_telemetry(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/telemetry", json={
            "device_update_interval": 300,
            "environment_measurement_enabled": True,
            "environment_update_interval": 600,
        })
        assert r.status_code == 200
        assert r.json()["success"] is True


class TestDeviceShutdownEndpoint:
    """POST /api/addons/meshtastic/device/shutdown"""

    def test_shutdown(self, client, mocks):
        r = client.post("/api/addons/meshtastic/device/shutdown")
        assert r.status_code == 200
        assert r.json()["success"] is True
        mocks["device_manager"].shutdown.assert_called_once()


class TestDeviceExportImportEndpoints:
    """GET /api/addons/meshtastic/device/export and POST /device/import"""

    def test_export_config(self, client, mocks):
        r = client.get("/api/addons/meshtastic/device/export")
        assert r.status_code == 200
        data = r.json()
        assert "device" in data
        assert "lora" in data
        assert "channels" in data

    def test_import_config(self, client, mocks):
        config = {
            "device": {"role": "ROUTER"},
            "lora": {"region": "EU_868"},
            "channels": [{"index": 0, "name": "Primary", "role": "PRIMARY"}],
        }
        r = client.post("/api/addons/meshtastic/device/import", json=config)
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mocks["device_manager"].import_config.assert_called_once_with(config)


# ===========================================================================
# Disconnected state tests — verify graceful degradation
# ===========================================================================


class TestDisconnectedState:
    """Verify endpoints handle disconnected state gracefully."""

    @pytest.fixture
    def disconnected_client(self):
        conn = make_mock_connection(connected=False)
        nm = NodeManager()  # empty
        bridge = MessageBridge()
        dm = make_mock_device_manager(connection=conn)
        # Override device_manager to return empty/failed data for disconnected
        dm.get_device_info = AsyncMock(return_value=DeviceInfo())
        dm.reboot = AsyncMock(return_value=False)
        dm.factory_reset = AsyncMock(return_value=False)
        dm.export_config = AsyncMock(return_value={})

        app = FastAPI()
        router = create_router(conn, nm, bridge)
        device_router = create_device_routes(dm)
        app.include_router(router, prefix="/api/addons/meshtastic")
        app.include_router(device_router, prefix="/api/addons/meshtastic")
        return TestClient(app)

    def test_status_disconnected(self, disconnected_client):
        r = disconnected_client.get("/api/addons/meshtastic/status")
        assert r.status_code == 200
        data = r.json()
        assert data["connected"] is False
        assert data["node_count"] == 0

    def test_nodes_empty_when_disconnected(self, disconnected_client):
        r = disconnected_client.get("/api/addons/meshtastic/nodes")
        assert r.status_code == 200
        assert r.json()["nodes"] == []

    def test_targets_empty_when_disconnected(self, disconnected_client):
        r = disconnected_client.get("/api/addons/meshtastic/targets")
        assert r.status_code == 200
        assert r.json()["targets"] == []

    def test_messages_empty_when_disconnected(self, disconnected_client):
        r = disconnected_client.get("/api/addons/meshtastic/messages")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_health_degraded_when_disconnected(self, disconnected_client):
        r = disconnected_client.get("/api/addons/meshtastic/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "degraded"
        assert data["connected"] is False

    def test_reboot_fails_disconnected(self, disconnected_client):
        r = disconnected_client.post("/api/addons/meshtastic/device/reboot")
        assert r.status_code == 503

    def test_factory_reset_fails_disconnected(self, disconnected_client):
        r = disconnected_client.post("/api/addons/meshtastic/device/factory-reset")
        assert r.status_code == 503

    def test_export_fails_disconnected(self, disconnected_client):
        r = disconnected_client.get("/api/addons/meshtastic/device/export")
        assert r.status_code == 503

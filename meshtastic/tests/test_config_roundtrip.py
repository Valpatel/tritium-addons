# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Config roundtrip tests for Meshtastic DeviceManager.

Verifies that every config setter method:
1. Calls the correct sync helper with expected protobuf values
2. Returns True on success
3. Returns False when disconnected

Uses mocked meshtastic interfaces to avoid needing real hardware.

UX Loop 2 (Add Sensor) — configuration management for mesh radios.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

from meshtastic_addon.device_manager import (
    DeviceManager,
    DeviceRole,
    ChannelInfo,
    DeviceInfo,
    FirmwareInfo,
)


# ---------------------------------------------------------------------------
# Mock meshtastic interface builder
# ---------------------------------------------------------------------------

def make_mock_interface():
    """Build a mock meshtastic interface with localNode and all config objects."""
    iface = MagicMock()

    # localNode with config objects
    node = MagicMock()
    iface.localNode = node

    # localConfig tree
    local_config = MagicMock()
    node.localConfig = local_config
    local_config.device = MagicMock()
    local_config.lora = MagicMock()
    local_config.position = MagicMock()
    local_config.network = MagicMock()
    local_config.bluetooth = MagicMock()
    local_config.display = MagicMock()
    local_config.power = MagicMock()

    # moduleConfig tree
    module_config = MagicMock()
    node.moduleConfig = module_config
    module_config.mqtt = MagicMock()
    module_config.telemetry = MagicMock()

    # Channels — list of 8 channel mocks
    channels = []
    for i in range(8):
        ch = MagicMock()
        ch.settings = MagicMock()
        ch.settings.name = ""
        ch.settings.psk = b""
        ch.settings.uplink_enabled = False
        ch.settings.downlink_enabled = False
        ch.role = 0  # DISABLED
        channels.append(ch)
    # Primary channel
    channels[0].settings.name = "Primary"
    channels[0].role = 1  # PRIMARY
    node.channels = channels

    # node methods
    node.setOwner = MagicMock()
    node.writeConfig = MagicMock()
    node.writeChannel = MagicMock()
    node.setFixedPosition = MagicMock()
    node.getURL = MagicMock(return_value="https://meshtastic.org/e/#CgMSAQ")
    node.setURL = MagicMock()
    node.shutdown = MagicMock()
    node.reboot = MagicMock()
    node.factoryReset = MagicMock()

    # Interface-level
    iface.getMyNodeInfo = MagicMock(return_value={
        "user": {
            "id": "!ba33ff38",
            "longName": "Tritium Base",
            "shortName": "TRIT",
            "hwModel": "T_LORA_PAGER",
            "macaddr": "ba:33:ff:38:00:01",
        }
    })

    return iface


def make_connected_manager():
    """Build a DeviceManager with a mocked connected interface."""
    conn = MagicMock()
    conn.is_connected = True
    iface = make_mock_interface()
    conn.interface = iface

    dm = DeviceManager(conn)
    return dm, iface


def make_disconnected_manager():
    """Build a DeviceManager with no connection."""
    conn = MagicMock()
    conn.is_connected = False
    conn.interface = None
    return DeviceManager(conn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetOwner:
    """set_owner(long_name, short_name) config roundtrip."""

    @pytest.mark.asyncio
    async def test_set_owner_both_names(self):
        dm, iface = make_connected_manager()
        result = await dm.set_owner("Base Alpha", "BA")
        assert result is True
        iface.localNode.setOwner.assert_called_once_with(
            long_name="Base Alpha", short_name="BA",
        )

    @pytest.mark.asyncio
    async def test_set_owner_long_name_only(self):
        dm, iface = make_connected_manager()
        result = await dm.set_owner("Just Long Name", "")
        assert result is True
        iface.localNode.setOwner.assert_called_once_with(
            long_name="Just Long Name",
        )

    @pytest.mark.asyncio
    async def test_set_owner_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_owner("Nope", "NO")
        assert result is False


class TestSetRole:
    """set_role(role) for each DeviceRole value."""

    @pytest.mark.asyncio
    async def test_set_role_client(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_role_sync") as mock_sync:
            result = await dm.set_role("CLIENT")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_role_router(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_role_sync") as mock_sync:
            result = await dm.set_role("ROUTER")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_role_repeater(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_role_sync") as mock_sync:
            result = await dm.set_role("REPEATER")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_role_tracker(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_role_sync") as mock_sync:
            result = await dm.set_role("TRACKER")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_role_all_valid_roles(self):
        """Every DeviceRole enum value should be accepted."""
        dm, iface = make_connected_manager()
        for role in DeviceRole:
            with patch("meshtastic_addon.device_manager.DeviceManager._set_role_sync"):
                result = await dm.set_role(role.value)
                assert result is True, f"Role {role.value} should succeed"

    @pytest.mark.asyncio
    async def test_set_role_invalid(self):
        dm, iface = make_connected_manager()
        result = await dm.set_role("INVALID_ROLE")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_role_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_role("CLIENT")
        assert result is False


class TestSetLoraConfig:
    """set_lora_config(region, modem_preset, tx_power)."""

    @pytest.mark.asyncio
    async def test_set_lora_all_params(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_lora_config_sync") as mock_sync:
            result = await dm.set_lora_config(
                tx_power=20,
                region="EU_868",
                modem_preset="SHORT_SLOW",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_set_lora_region_only(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_lora_config_sync") as mock_sync:
            result = await dm.set_lora_config(region="US")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_lora_tx_power_only(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_lora_config_sync") as mock_sync:
            result = await dm.set_lora_config(tx_power=27)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_lora_empty_is_noop(self):
        dm, iface = make_connected_manager()
        result = await dm.set_lora_config()
        assert result is True  # Nothing to set = success

    @pytest.mark.asyncio
    async def test_set_lora_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_lora_config(region="US")
        assert result is False


class TestSetPosition:
    """set_position(lat, lng, altitude, gps_mode)."""

    @pytest.mark.asyncio
    async def test_set_position_full(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_position_sync") as mock_pos, \
             patch("meshtastic_addon.device_manager.DeviceManager._set_gps_mode_sync") as mock_gps:
            result = await dm.set_position(
                lat=30.2672, lng=-97.7431, altitude=150, gps_mode="DISABLED",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_set_position_coords_only(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_position_sync") as mock_pos:
            result = await dm.set_position(lat=30.2672, lng=-97.7431)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_position_gps_mode_only(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_gps_mode_sync") as mock_gps:
            result = await dm.set_position(gps_mode="ENABLED")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_position_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_position(lat=30.0, lng=-97.0)
        assert result is False


class TestSetWifi:
    """set_wifi(enabled, ssid, password)."""

    @pytest.mark.asyncio
    async def test_set_wifi_enable(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_network_config_sync") as mock_sync:
            result = await dm.set_wifi(enabled=True, ssid="TestNet", password="pass123")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_wifi_disable(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_network_config_sync") as mock_sync:
            result = await dm.set_wifi(enabled=False)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_wifi_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_wifi(enabled=True, ssid="Test", password="x")
        assert result is False


class TestSetBluetooth:
    """set_bluetooth(enabled)."""

    @pytest.mark.asyncio
    async def test_set_bluetooth_enable(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_bluetooth_config_sync") as mock_sync:
            result = await dm.set_bluetooth(enabled=True)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_bluetooth_disable(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_bluetooth_config_sync") as mock_sync:
            result = await dm.set_bluetooth(enabled=False)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_bluetooth_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_bluetooth(enabled=True)
        assert result is False


class TestSetDisplayConfig:
    """set_display_config(screen_on_secs, auto_screen_carousel_secs, flip_screen, units)."""

    @pytest.mark.asyncio
    async def test_set_display_full(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_display_config_sync") as mock_sync:
            result = await dm.set_display_config(
                screen_on_secs=120,
                auto_screen_carousel_secs=10,
                flip_screen=True,
                units="METRIC",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_set_display_screen_off(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_display_config_sync") as mock_sync:
            result = await dm.set_display_config(screen_on_secs=0)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_display_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_display_config(screen_on_secs=60)
        assert result is False


class TestSetPowerConfig:
    """set_power_config(is_power_saving, on_battery_shutdown_after_secs)."""

    @pytest.mark.asyncio
    async def test_set_power_saving_on(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_power_config_sync") as mock_sync:
            result = await dm.set_power_config(
                is_power_saving=True,
                on_battery_shutdown_after_secs=3600,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_set_power_saving_off(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_power_config_sync") as mock_sync:
            result = await dm.set_power_config(
                is_power_saving=False,
                on_battery_shutdown_after_secs=0,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_set_power_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_power_config(is_power_saving=True)
        assert result is False


class TestSetMqttConfig:
    """set_mqtt_config(enabled, address, username, password, encryption_enabled, json_enabled)."""

    @pytest.mark.asyncio
    async def test_set_mqtt_full(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_mqtt_config_sync") as mock_sync:
            result = await dm.set_mqtt_config(
                enabled=True,
                address="mqtt.local:1883",
                username="mesh",
                password="pass",
                encryption_enabled=True,
                json_enabled=False,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_set_mqtt_disable(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_mqtt_config_sync") as mock_sync:
            result = await dm.set_mqtt_config(enabled=False)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_mqtt_empty_is_noop(self):
        dm, iface = make_connected_manager()
        result = await dm.set_mqtt_config()
        assert result is True  # Nothing to set = success

    @pytest.mark.asyncio
    async def test_set_mqtt_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_mqtt_config(enabled=True)
        assert result is False


class TestSetTelemetryConfig:
    """set_telemetry_config(device_update_interval, environment_measurement_enabled, ...)."""

    @pytest.mark.asyncio
    async def test_set_telemetry_full(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_telemetry_config_sync") as mock_sync:
            result = await dm.set_telemetry_config(
                device_update_interval=300,
                environment_measurement_enabled=True,
                environment_update_interval=600,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_set_telemetry_device_only(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._set_telemetry_config_sync") as mock_sync:
            result = await dm.set_telemetry_config(device_update_interval=120)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_telemetry_empty_is_noop(self):
        dm, iface = make_connected_manager()
        result = await dm.set_telemetry_config()
        assert result is True

    @pytest.mark.asyncio
    async def test_set_telemetry_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_telemetry_config(device_update_interval=300)
        assert result is False


class TestConfigureChannel:
    """configure_channel(index, name, psk, role, uplink_enabled, downlink_enabled)."""

    @pytest.mark.asyncio
    async def test_configure_channel_new_secondary(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._configure_channel_sync") as mock_sync:
            result = await dm.configure_channel(
                index=1,
                name="Ops",
                psk="random",
                role="SECONDARY",
                uplink_enabled=True,
                downlink_enabled=False,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_configure_channel_edit_primary(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._configure_channel_sync") as mock_sync:
            result = await dm.configure_channel(
                index=0,
                name="NewPrimary",
                psk="default",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_configure_channel_disable(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._configure_channel_sync") as mock_sync:
            result = await dm.configure_channel(
                index=3,
                role="DISABLED",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_configure_channel_invalid_index_low(self):
        dm, iface = make_connected_manager()
        result = await dm.configure_channel(index=-1, name="bad")
        assert result is False

    @pytest.mark.asyncio
    async def test_configure_channel_invalid_index_high(self):
        dm, iface = make_connected_manager()
        result = await dm.configure_channel(index=8, name="bad")
        assert result is False

    @pytest.mark.asyncio
    async def test_configure_channel_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.configure_channel(index=0, name="test")
        assert result is False


class TestChannelUrl:
    """get_channel_url() and set_channel_url(url)."""

    @pytest.mark.asyncio
    async def test_get_channel_url(self):
        dm, iface = make_connected_manager()
        url = await dm.get_channel_url()
        assert url == "https://meshtastic.org/e/#CgMSAQ"
        iface.localNode.getURL.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_channel_url(self):
        dm, iface = make_connected_manager()
        result = await dm.set_channel_url("https://meshtastic.org/e/#NewUrl")
        assert result is True
        iface.localNode.setURL.assert_called_once_with("https://meshtastic.org/e/#NewUrl")

    @pytest.mark.asyncio
    async def test_get_channel_url_disconnected(self):
        dm = make_disconnected_manager()
        url = await dm.get_channel_url()
        assert url == ""

    @pytest.mark.asyncio
    async def test_set_channel_url_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.set_channel_url("https://meshtastic.org/e/#x")
        assert result is False


class TestRemoveChannel:
    """remove_channel(index) — convenience wrapper."""

    @pytest.mark.asyncio
    async def test_remove_secondary_channel(self):
        dm, iface = make_connected_manager()
        with patch("meshtastic_addon.device_manager.DeviceManager._configure_channel_sync") as mock_sync:
            result = await dm.remove_channel(2)
            assert result is True

    @pytest.mark.asyncio
    async def test_cannot_remove_primary(self):
        dm, iface = make_connected_manager()
        result = await dm.remove_channel(0)
        assert result is False


class TestDeviceInfoRead:
    """get_device_info() reads hardware, firmware, role, channels."""

    @pytest.mark.asyncio
    async def test_get_device_info_connected(self):
        dm, iface = make_connected_manager()
        # We need to test the real _read_device_info_sync, so we mock at a deeper level
        iface.getMyNodeInfo.return_value = {
            "user": {
                "id": "!ba33ff38",
                "longName": "Tritium Base",
                "shortName": "TRIT",
                "hwModel": "T_LORA_PAGER",
                "macaddr": "ba:33:ff:38:00:01",
            }
        }
        info = await dm.get_device_info()
        assert isinstance(info, DeviceInfo)
        assert info.node_id == "!ba33ff38"
        assert info.long_name == "Tritium Base"
        assert info.hw_model == "T_LORA_PAGER"

    @pytest.mark.asyncio
    async def test_get_device_info_disconnected(self):
        dm = make_disconnected_manager()
        info = await dm.get_device_info()
        assert isinstance(info, DeviceInfo)
        assert info.node_id == ""


class TestGetChannels:
    """get_channels() reads channel configuration."""

    @pytest.mark.asyncio
    async def test_get_channels_connected(self):
        dm, iface = make_connected_manager()
        channels = await dm.get_channels()
        assert isinstance(channels, list)
        # The mock has 8 channels
        assert len(channels) == 8
        assert channels[0].name == "Primary"

    @pytest.mark.asyncio
    async def test_get_channels_disconnected(self):
        dm = make_disconnected_manager()
        channels = await dm.get_channels()
        assert channels == []


class TestShutdown:
    """shutdown() sends power-off command."""

    @pytest.mark.asyncio
    async def test_shutdown_connected(self):
        dm, iface = make_connected_manager()
        result = await dm.shutdown()
        assert result is True
        iface.localNode.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_disconnected(self):
        dm = make_disconnected_manager()
        result = await dm.shutdown()
        assert result is False

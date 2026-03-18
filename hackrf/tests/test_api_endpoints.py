# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for all HackRF addon API endpoints via FastAPI TestClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hackrf_addon.router import create_router


def _make_app(
    device=None,
    spectrum=None,
    receiver=None,
    fm_decoder=None,
    tpms_decoder=None,
    ism_monitor=None,
    continuous_scanner=None,
    rtl433=None,
    adsb_decoder=None,
):
    """Create a FastAPI app with mocked dependencies for testing."""
    if device is None:
        device = MagicMock()
        device.is_available = True
        device.get_info.return_value = {
            "serial": "abc123",
            "firmware_version": "2024.02.1",
            "board_name": "HackRF One",
        }
        device.detect = AsyncMock(return_value={
            "serial": "abc123",
            "firmware_version": "2024.02.1",
            "board_name": "HackRF One",
            "hardware_revision": "r9",
            "raw_output": "...",
        })
        device.flash_firmware = AsyncMock(return_value={"success": True, "output": "OK"})
        device.set_clock = AsyncMock(return_value={"success": True})
        device.get_clock_info = AsyncMock(return_value={"clkin": 0, "clkout": 0})
        device.get_operacake_boards = AsyncMock(return_value={"success": True, "boards": []})
        device.get_antenna_config = AsyncMock(return_value={"boards": []})
        device.set_antenna_port = AsyncMock(return_value={"success": True})
        device.set_bias_tee = AsyncMock(return_value={"success": True})
        device.get_board_id = AsyncMock(return_value={"success": True, "board_id": 2})
        device.get_debug_info = AsyncMock(return_value={"success": True, "pll": "locked"})
        device.get_cpld_checksum = AsyncMock(return_value={"success": True, "cpld_checksum": "0xABCD"})
        device.flash_cpld = AsyncMock(return_value={"success": True})
        device.reset_device = AsyncMock(return_value={"success": True})
        device.set_clkin = AsyncMock(return_value={"success": True})
        device.set_clkout = AsyncMock(return_value={"success": True})

    if spectrum is None:
        spectrum = MagicMock()
        spectrum.get_status.return_value = {"running": False, "sweep_count": 0, "measurement_count": 0}
        spectrum.start_sweep = AsyncMock(return_value={"success": True, "freq_start_mhz": 0, "freq_end_mhz": 6000})
        spectrum.stop_sweep = AsyncMock(return_value={"success": True, "sweep_count": 42})
        spectrum.get_data.return_value = [{"freq_hz": 100000000, "power_dbm": -40.0}]
        spectrum.is_running = False
        spectrum.signal_db = MagicMock()
        spectrum.signal_db.get_peaks.return_value = [{"freq_hz": 100000000, "power_dbm": -20.0, "timestamp": 1.0}]
        spectrum.signal_db.count = 0

    if receiver is None:
        receiver = MagicMock()
        receiver.get_status.return_value = {"running": False, "freq_hz": 100000000}
        receiver.tune.return_value = {"success": True, "freq_hz": 100000000}
        receiver.start = AsyncMock(return_value={"success": True})
        receiver.stop = AsyncMock(return_value={"success": True})
        receiver.get_captures.return_value = []
        receiver.is_running = False

    app = FastAPI()
    router = create_router(
        device, spectrum, receiver,
        fm_decoder=fm_decoder,
        tpms_decoder=tpms_decoder,
        ism_monitor=ism_monitor,
        continuous_scanner=continuous_scanner,
        rtl433=rtl433,
        adsb_decoder=adsb_decoder,
    )
    app.include_router(router, prefix="/api/hackrf")
    return app


class TestStatusEndpoints:
    """Tests for status/info/health endpoints."""

    def test_get_status(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["connected"] is True
        assert "sweep" in data
        assert "receiver" in data

    def test_get_info(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert "raw_output" not in data  # Should be filtered out

    def test_get_info_no_device(self):
        device = MagicMock()
        device.is_available = False
        device.detect = AsyncMock(return_value=None)
        client = TestClient(_make_app(device=device))
        resp = client.get("/api/hackrf/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False

    def test_health(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["available"] is True

    def test_health_degraded(self):
        device = MagicMock()
        device.is_available = True
        device.get_info.return_value = None
        client = TestClient(_make_app(device=device))
        resp = client.get("/api/hackrf/health")
        data = resp.json()
        assert data["status"] == "degraded"


class TestPortsEndpoint:
    """Tests for device detection endpoint."""

    def test_detect_ports_with_device(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/ports")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["devices"][0]["serial"] == "abc123"

    def test_detect_ports_no_device(self):
        device = MagicMock()
        device.is_available = True
        device.detect = AsyncMock(return_value=None)
        client = TestClient(_make_app(device=device))
        resp = client.get("/api/hackrf/ports")
        data = resp.json()
        assert data["count"] == 0


class TestSweepEndpoints:
    """Tests for sweep start/stop/data/peaks."""

    def test_sweep_start(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/sweep/start", json={"freq_start": 88, "freq_end": 108})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_sweep_start_defaults(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/sweep/start")
        assert resp.status_code == 200

    def test_sweep_stop(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/sweep/stop")
        assert resp.status_code == 200

    def test_sweep_data(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/sweep/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert "count" in data

    def test_sweep_peaks(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/sweep/peaks?threshold=-30.0")
        assert resp.status_code == 200
        data = resp.json()
        assert "peaks" in data
        assert data["threshold_dbm"] == -30.0


class TestTuneEndpoint:
    """Tests for receiver tuning."""

    def test_tune(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/tune", json={"freq_hz": 101100000})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_tune_with_capture(self):
        receiver = MagicMock()
        receiver.tune.return_value = {"success": True, "freq_hz": 101100000}
        receiver.start = AsyncMock(return_value={"success": True})
        receiver.get_status.return_value = {"running": False}
        client = TestClient(_make_app(receiver=receiver))
        resp = client.post("/api/hackrf/tune", json={"freq_hz": 101100000, "start_capture": True})
        assert resp.status_code == 200


class TestCaptureEndpoints:
    """Tests for IQ capture start/stop/list."""

    def test_capture_start(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/capture/start", json={"freq_hz": 100000000})
        assert resp.status_code == 200

    def test_capture_start_defaults(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/capture/start")
        assert resp.status_code == 200

    def test_capture_stop(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/capture/stop")
        assert resp.status_code == 200

    def test_capture_list(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/capture/list")
        assert resp.status_code == 200
        assert "captures" in resp.json()


class TestFirmwareEndpoints:
    """Tests for firmware info and flash."""

    def test_firmware_info(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/firmware")
        assert resp.status_code == 200
        data = resp.json()
        assert "firmware_version" in data

    def test_firmware_info_no_device(self):
        device = MagicMock()
        device.get_info.return_value = None
        device.detect = AsyncMock(return_value=None)
        client = TestClient(_make_app(device=device))
        resp = client.get("/api/hackrf/firmware")
        data = resp.json()
        assert "error" in data


class TestFMDecoderEndpoints:
    """Tests for FM decoder endpoints."""

    def test_fm_tune_no_decoder(self):
        client = TestClient(_make_app(fm_decoder=None))
        resp = client.post("/api/hackrf/fm/tune", json={"freq_hz": 101100000})
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_fm_stations_no_decoder(self):
        client = TestClient(_make_app(fm_decoder=None))
        resp = client.get("/api/hackrf/fm/stations")
        assert resp.status_code == 200
        assert resp.json()["stations"] == []


class TestTPMSEndpoints:
    """Tests for TPMS decoder endpoints."""

    def test_tpms_start_no_decoder(self):
        client = TestClient(_make_app(tpms_decoder=None))
        resp = client.post("/api/hackrf/tpms/start")
        assert resp.json()["success"] is False

    def test_tpms_stop_no_decoder(self):
        client = TestClient(_make_app(tpms_decoder=None))
        resp = client.post("/api/hackrf/tpms/stop")
        assert resp.json()["success"] is False

    def test_tpms_sensors_no_decoder(self):
        client = TestClient(_make_app(tpms_decoder=None))
        resp = client.get("/api/hackrf/tpms/sensors")
        assert resp.json()["sensors"] == []

    def test_tpms_transmissions_no_decoder(self):
        client = TestClient(_make_app(tpms_decoder=None))
        resp = client.get("/api/hackrf/tpms/transmissions")
        assert resp.json()["transmissions"] == []

    def test_tpms_start_with_decoder(self):
        tpms = MagicMock()
        tpms.start_monitoring = AsyncMock(return_value={"success": True})
        client = TestClient(_make_app(tpms_decoder=tpms))
        resp = client.post("/api/hackrf/tpms/start", json={"freq_hz": 315000000})
        assert resp.json()["success"] is True

    def test_tpms_sensors_with_decoder(self):
        tpms = MagicMock()
        tpms.get_sensors.return_value = [{"sensor_id": "abc", "freq_hz": 315000000}]
        tpms.get_status.return_value = {"running": True}
        client = TestClient(_make_app(tpms_decoder=tpms))
        resp = client.get("/api/hackrf/tpms/sensors")
        data = resp.json()
        assert data["count"] == 1


class TestISMEndpoints:
    """Tests for ISM band monitor endpoints."""

    def test_ism_start_no_monitor(self):
        client = TestClient(_make_app(ism_monitor=None))
        resp = client.post("/api/hackrf/ism/start")
        assert resp.json()["success"] is False

    def test_ism_stop_no_monitor(self):
        client = TestClient(_make_app(ism_monitor=None))
        resp = client.post("/api/hackrf/ism/stop")
        assert resp.json()["success"] is False

    def test_ism_devices_no_monitor(self):
        client = TestClient(_make_app(ism_monitor=None))
        resp = client.get("/api/hackrf/ism/devices")
        assert resp.json()["devices"] == []

    def test_ism_log_no_monitor(self):
        client = TestClient(_make_app(ism_monitor=None))
        resp = client.get("/api/hackrf/ism/log")
        assert resp.json()["log"] == []

    def test_ism_bands_no_monitor(self):
        client = TestClient(_make_app(ism_monitor=None))
        resp = client.get("/api/hackrf/ism/bands")
        assert resp.json()["bands"] == []

    def test_ism_start_with_monitor(self):
        ism = MagicMock()
        ism.start_monitoring = AsyncMock(return_value={"success": True, "bands": ["315 MHz"]})
        client = TestClient(_make_app(ism_monitor=ism))
        resp = client.post("/api/hackrf/ism/start", json={"threshold_dbm": -60})
        assert resp.json()["success"] is True


class TestScannerEndpoints:
    """Tests for continuous scanner endpoints."""

    def test_scanner_start_none(self):
        client = TestClient(_make_app(continuous_scanner=None))
        resp = client.post("/api/hackrf/scanner/start")
        assert "error" in resp.json()

    def test_scanner_stop_none(self):
        client = TestClient(_make_app(continuous_scanner=None))
        resp = client.post("/api/hackrf/scanner/stop")
        assert "error" in resp.json()

    def test_scanner_summary_none(self):
        client = TestClient(_make_app(continuous_scanner=None))
        resp = client.get("/api/hackrf/scanner/summary")
        assert "error" in resp.json()

    def test_scanner_status_none(self):
        client = TestClient(_make_app(continuous_scanner=None))
        resp = client.get("/api/hackrf/scanner/status")
        data = resp.json()
        assert data["running"] is False

    def test_scanner_start_with_scanner(self):
        scanner = MagicMock()
        scanner.start = AsyncMock(return_value={"success": True, "bands": 9})
        client = TestClient(_make_app(continuous_scanner=scanner))
        resp = client.post("/api/hackrf/scanner/start")
        assert resp.json()["success"] is True


class TestRtl433Endpoints:
    """Tests for rtl_433 wrapper endpoints."""

    def test_rtl433_start_none(self):
        client = TestClient(_make_app(rtl433=None))
        resp = client.post("/api/hackrf/rtl433/start")
        assert "error" in resp.json()

    def test_rtl433_stop_none(self):
        client = TestClient(_make_app(rtl433=None))
        resp = client.post("/api/hackrf/rtl433/stop")
        assert "error" in resp.json()

    def test_rtl433_events_none(self):
        client = TestClient(_make_app(rtl433=None))
        resp = client.get("/api/hackrf/rtl433/events")
        assert resp.json()["events"] == []

    def test_rtl433_devices_none(self):
        client = TestClient(_make_app(rtl433=None))
        resp = client.get("/api/hackrf/rtl433/devices")
        assert resp.json()["devices"] == []

    def test_rtl433_tpms_none(self):
        client = TestClient(_make_app(rtl433=None))
        resp = client.get("/api/hackrf/rtl433/tpms")
        assert resp.json()["sensors"] == []

    def test_rtl433_stats_none(self):
        client = TestClient(_make_app(rtl433=None))
        resp = client.get("/api/hackrf/rtl433/stats")
        assert "error" in resp.json()

    def test_rtl433_start_with_wrapper(self):
        rtl = MagicMock()
        rtl.start_monitoring = AsyncMock(return_value={"success": True, "freq_hz": 315000000})
        client = TestClient(_make_app(rtl433=rtl))
        resp = client.post("/api/hackrf/rtl433/start", json={"freq_hz": 433920000})
        assert resp.json()["success"] is True


class TestADSBEndpoints:
    """Tests for ADS-B aircraft tracking endpoints."""

    def test_adsb_start_none(self):
        client = TestClient(_make_app(adsb_decoder=None))
        resp = client.post("/api/hackrf/adsb/start")
        assert resp.json()["success"] is False

    def test_adsb_stop_none(self):
        client = TestClient(_make_app(adsb_decoder=None))
        resp = client.post("/api/hackrf/adsb/stop")
        assert resp.json()["success"] is False

    def test_adsb_aircraft_none(self):
        client = TestClient(_make_app(adsb_decoder=None))
        resp = client.get("/api/hackrf/adsb/aircraft")
        assert resp.json()["aircraft"] == []

    def test_adsb_stats_none(self):
        client = TestClient(_make_app(adsb_decoder=None))
        resp = client.get("/api/hackrf/adsb/stats")
        assert "error" in resp.json()

    def test_adsb_start_with_decoder(self):
        adsb = MagicMock()
        adsb.start_monitoring = AsyncMock(return_value={"success": True, "freq_mhz": 1090})
        client = TestClient(_make_app(adsb_decoder=adsb))
        resp = client.post("/api/hackrf/adsb/start", json={"cycle_s": 5})
        assert resp.json()["success"] is True

    def test_adsb_stop_with_decoder(self):
        adsb = MagicMock()
        adsb.stop_monitoring = AsyncMock(return_value={"success": True, "aircraft": 3})
        client = TestClient(_make_app(adsb_decoder=adsb))
        resp = client.post("/api/hackrf/adsb/stop")
        assert resp.json()["success"] is True

    def test_adsb_aircraft_with_decoder(self):
        adsb = MagicMock()
        adsb.get_aircraft.return_value = [
            {"icao": "a1b2c3", "callsign": "UAL123", "altitude_ft": 35000},
        ]
        client = TestClient(_make_app(adsb_decoder=adsb))
        resp = client.get("/api/hackrf/adsb/aircraft")
        data = resp.json()
        assert data["count"] == 1
        assert data["aircraft"][0]["icao"] == "a1b2c3"

    def test_adsb_stats_with_decoder(self):
        adsb = MagicMock()
        adsb.get_stats.return_value = {
            "running": True,
            "aircraft_total": 5,
            "messages_decoded": 100,
        }
        client = TestClient(_make_app(adsb_decoder=adsb))
        resp = client.get("/api/hackrf/adsb/stats")
        data = resp.json()
        assert data["aircraft_total"] == 5


class TestDiagnosticsEndpoints:
    """Tests for diagnostics and hardware management endpoints."""

    def test_diagnostics(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/diagnostics")
        assert resp.status_code == 200

    def test_clock_info(self):
        client = TestClient(_make_app())
        resp = client.get("/api/hackrf/clock")
        assert resp.status_code == 200

    def test_bias_tee(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/bias-tee", json={"enabled": True})
        assert resp.status_code == 200

    def test_device_reset(self):
        client = TestClient(_make_app())
        resp = client.post("/api/hackrf/device/reset")
        assert resp.status_code == 200

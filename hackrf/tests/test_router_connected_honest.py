# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Honesty tests for /api/hackrf/info.

Asserts that ``connected`` reflects the *actual* result of running
``hackrf_info`` — never unconditionally ``True``.

Gap-fix A regression: previously ``router.py:66`` set
``clean["connected"] = True`` after ``device.detect()`` returned
*anything* non-None. With buggy device implementations that returned
``{"connected": False, "error": "..."}`` (or fabricated payloads), the
API still reported ``connected: True`` and the frontend told the
operator "HackRF connected" with no hardware attached.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hackrf_addon.router import create_router


def _build_client(detect_return) -> TestClient:
    """Build a FastAPI client with a device whose detect() returns the given value.

    Mounts the router at the same ``/api/hackrf`` prefix used in production
    (see ``test_api_endpoints.py``).
    """
    device = MagicMock()
    device.is_available = True
    device.detect = AsyncMock(return_value=detect_return)
    device.get_info.return_value = None

    spectrum = MagicMock()
    spectrum.get_status.return_value = {}
    spectrum.signal_db = MagicMock()
    spectrum.is_running = False

    receiver = MagicMock()
    receiver.get_status.return_value = {}

    app = FastAPI()
    app.include_router(create_router(device, spectrum, receiver), prefix="/api/hackrf")
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Honest FALSE: no hardware → connected:false (this is the bug surface)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_info_returns_connected_false_when_detect_returns_none():
    """No HackRF on the bus → /info MUST say connected:false."""
    client = _build_client(detect_return=None)
    resp = client.get("/api/hackrf/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False, "Must NOT lie when no hardware present"
    assert "error" in body
    assert body["error_reason"] == "no_hackrf_signature_in_output"


@pytest.mark.unit
def test_info_returns_connected_false_when_detect_returns_error_envelope():
    """device.detect() returned a {connected:false, error:...} envelope."""
    client = _build_client(detect_return={
        "connected": False,
        "error": "hackrf_info exit code 1: usb_open() failed",
    })
    resp = client.get("/api/hackrf/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False, "Must propagate detect() error"
    assert "usb_open() failed" in body["error"]
    assert body["error_reason"] == "hackrf_info_error"


@pytest.mark.unit
def test_info_returns_connected_false_when_serial_is_empty():
    """Parsed payload but no serial = unreliable → must NOT claim connected."""
    client = _build_client(detect_return={
        "serial": "",
        "board_name": "HackRF One",
        "firmware_version": "2024.02.1",
    })
    resp = client.get("/api/hackrf/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False, "Empty serial is not a real connection"
    assert body["error_reason"] == "missing_serial"


@pytest.mark.unit
def test_info_returns_connected_false_when_serial_only_whitespace():
    """A serial of just spaces does not constitute a real device."""
    client = _build_client(detect_return={
        "serial": "    ",
        "board_name": "HackRF One",
    })
    resp = client.get("/api/hackrf/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False
    assert body["error_reason"] == "missing_serial"


# ---------------------------------------------------------------------------
# Honest TRUE: real hardware → connected:true
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_info_returns_connected_true_with_real_serial():
    """A parsed payload with a real serial → /info reports connected:true."""
    client = _build_client(detect_return={
        "serial": "0000000000000000 c66c63dc308d3d83",
        "firmware_version": "2024.02.1",
        "board_name": "HackRF One",
        "board_id": 2,
        "hardware_revision": "r9",
        "raw_output": "lots of text...",  # must be stripped
    })
    resp = client.get("/api/hackrf/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["serial"].endswith("c66c63dc308d3d83")
    # raw_output is large; must not be returned to clients
    assert "raw_output" not in body
    # No error envelope on success
    assert "error_reason" not in body

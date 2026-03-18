# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for HackRFDevice — detection, parsing, firmware flash."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from hackrf_addon.device import HackRFDevice


# Real hackrf_info output for testing
HACKRF_INFO_OUTPUT = """hackrf_info version: 2024.02.1
libhackrf version: 2024.02.1 (0.9)
Found HackRF
Index: 0
Serial number: 0000000000000000 c66c63dc308d3d83
Board ID Number: 2 (HackRF One)
Firmware Version: 2024.02.1 (API version 1.08)
Part ID Number: 0xa000cb3c 0x00724f61
Hardware Revision: r9
Hardware appears to have been manufactured by Great Scott Gadgets.
Hardware supported by installed firmware.
"""

HACKRF_INFO_MINIMAL = """Found HackRF
Serial number: abcdef1234567890
"""


class TestHackRFDeviceInit:
    """Tests for HackRFDevice initialization."""

    def test_init_default_state(self):
        dev = HackRFDevice()
        assert dev._info is None
        assert dev._available is None

    def test_get_info_before_detect(self):
        dev = HackRFDevice()
        assert dev.get_info() is None


class TestIsAvailable:
    """Tests for is_available property."""

    @patch("shutil.which", return_value="/usr/bin/hackrf_info")
    def test_available_when_binary_found(self, mock_which):
        dev = HackRFDevice()
        assert dev.is_available is True
        mock_which.assert_called_once_with("hackrf_info")

    @patch("shutil.which", return_value=None)
    def test_not_available_when_binary_missing(self, mock_which):
        dev = HackRFDevice()
        assert dev.is_available is False

    @patch("shutil.which", return_value="/usr/bin/hackrf_info")
    def test_caches_availability(self, mock_which):
        dev = HackRFDevice()
        _ = dev.is_available
        _ = dev.is_available
        mock_which.assert_called_once()  # Only called once, cached


class TestParseHackRFInfo:
    """Tests for _parse_hackrf_info parsing logic."""

    def test_parse_full_output(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert info is not None
        assert info["serial"] == "0000000000000000 c66c63dc308d3d83"
        assert info["board_id"] == 2
        assert info["board_name"] == "HackRF One"
        assert info["firmware_version"] == "2024.02.1"
        assert info["api_version"] == "1.08"
        assert info["part_id"] == "0xa000cb3c 0x00724f61"
        assert info["hardware_revision"] == "r9"
        assert info["tool_version"] == "2024.02.1"
        assert info["lib_version"] == "2024.02.1"
        assert info["manufacturer"] == "Great Scott Gadgets"

    def test_parse_serial_number(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert "c66c63dc308d3d83" in info["serial"]

    def test_parse_board_id(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert info["board_id"] == 2
        assert info["board_name"] == "HackRF One"

    def test_parse_firmware_version(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert info["firmware_version"] == "2024.02.1"
        assert info["api_version"] == "1.08"

    def test_parse_part_id(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert info["part_id"] == "0xa000cb3c 0x00724f61"

    def test_parse_hardware_revision(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert info["hardware_revision"] == "r9"

    def test_parse_manufacturer(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert info["manufacturer"] == "Great Scott Gadgets"

    def test_parse_minimal_output(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_MINIMAL)
        assert info is not None
        assert info["serial"] == "abcdef1234567890"

    def test_parse_no_device(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info("No HackRF boards found.\n")
        assert info is None

    def test_parse_empty_string(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info("")
        assert info is None

    def test_raw_output_preserved(self):
        dev = HackRFDevice()
        info = dev._parse_hackrf_info(HACKRF_INFO_OUTPUT)
        assert info["raw_output"] == HACKRF_INFO_OUTPUT


class TestDetect:
    """Tests for async detect method."""

    @pytest.mark.asyncio
    @patch("shutil.which", return_value=None)
    async def test_detect_not_available(self, mock_which):
        dev = HackRFDevice()
        result = await dev.detect()
        assert result.get("connected") is False

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_info")
    @patch("asyncio.create_subprocess_exec")
    async def test_detect_success(self, mock_exec, mock_which):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (HACKRF_INFO_OUTPUT.encode(), b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        dev = HackRFDevice()
        result = await dev.detect()
        assert result is not None
        assert result["serial"] == "0000000000000000 c66c63dc308d3d83"
        assert dev.get_info() is not None

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_info")
    @patch("asyncio.create_subprocess_exec")
    async def test_detect_nonzero_return(self, mock_exec, mock_which):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"usb error")
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        dev = HackRFDevice()
        result = await dev.detect()
        assert result.get("connected") is False

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_info")
    @patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError)
    async def test_detect_binary_not_found(self, mock_exec, mock_which):
        dev = HackRFDevice()
        result = await dev.detect()
        assert result.get("connected") is False
        assert dev._available is False

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_info")
    @patch("asyncio.create_subprocess_exec", side_effect=OSError("USB error"))
    async def test_detect_generic_error(self, mock_exec, mock_which):
        dev = HackRFDevice()
        result = await dev.detect()
        assert result.get("connected") is False


class TestFlashFirmware:
    """Tests for firmware flash."""

    @pytest.mark.asyncio
    @patch("shutil.which", return_value=None)
    async def test_flash_no_spiflash(self, mock_which):
        dev = HackRFDevice()
        result = await dev.flash_firmware("/tmp/test.bin")
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_spiflash")
    @patch("asyncio.create_subprocess_exec")
    async def test_flash_success(self, mock_exec, mock_which):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"Flashing...\nDone.", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        dev = HackRFDevice()
        dev._info = {"serial": "abc"}
        result = await dev.flash_firmware("/tmp/test.bin")
        assert result["success"] is True
        assert dev._info is None  # Cache invalidated

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_spiflash")
    @patch("asyncio.create_subprocess_exec")
    async def test_flash_failure(self, mock_exec, mock_which):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"Flash failed")
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        dev = HackRFDevice()
        result = await dev.flash_firmware("/tmp/bad.bin")
        assert result["success"] is False
        assert result["returncode"] == 1


class TestSetClock:
    """Tests for clock configuration."""

    @pytest.mark.asyncio
    @patch("shutil.which", return_value=None)
    async def test_clock_no_binary(self, mock_which):
        dev = HackRFDevice()
        result = await dev.set_clock(10_000_000)
        assert result["success"] is False

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_clock")
    @patch("asyncio.create_subprocess_exec")
    async def test_clock_success(self, mock_exec, mock_which):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"OK", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        dev = HackRFDevice()
        result = await dev.set_clock(10_000_000)
        assert result["success"] is True

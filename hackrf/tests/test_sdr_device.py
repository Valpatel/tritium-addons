# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests: HackRFSDRDevice satisfies the tritium_lib.sdr.SDRDevice ABC.

Proves the real HackRF backend implements the data-plane contract (not just
SimulatedSDR), and that every hardware path degrades gracefully with no radio.
"""

import inspect
import pytest
from unittest.mock import AsyncMock, patch

from tritium_lib.sdr.base import SDRDevice, SDRInfo, SweepResult, SweepPoint
from hackrf_addon.sdr_device import HackRFSDRDevice, HACKRF_FREQ_MAX_HZ
from hackrf_addon.device import HackRFDevice


# Parsed dict as HackRFDevice.detect() returns on success (no "connected" key).
PARSED_OK = {
    "serial": "0000000000000000 c66c63dc308d3d83",
    "board_id": 2,
    "board_name": "HackRF One",
    "firmware_version": "2024.02.1",
    "api_version": "1.08",
    "hardware_revision": "r9",
}

# Real-shape hackrf_sweep CSV: date, time, hz_low, hz_high, bin_width, nsamp, dB...
SWEEP_CSV = """2024-01-01, 12:00:00.000, 2400000000, 2405000000, 1000000, 20, -80.1, -79.5, -20.2, -78.9, -77.0
2024-01-01, 12:00:00.001, 2405000000, 2410000000, 1000000, 20, -76.0, -30.5, -75.1, -74.0, -73.2
"""


class TestABCConformance:
    def test_is_subclass(self):
        assert issubclass(HackRFSDRDevice, SDRDevice)

    def test_no_abstractmethods_remain(self):
        # Non-empty => an abstract method is unimplemented and instantiation
        # would raise TypeError. Empty proves the contract is fully satisfied.
        assert HackRFSDRDevice.__abstractmethods__ == frozenset()

    def test_instantiates_and_isinstance(self):
        dev = HackRFSDRDevice()
        assert isinstance(dev, SDRDevice)

    def test_abstract_methods_are_coroutines(self):
        for name in SDRDevice.__abstractmethods__:
            fn = getattr(HackRFSDRDevice, name)
            assert inspect.iscoroutinefunction(fn), f"{name} must be async"


class TestDetect:
    @pytest.mark.asyncio
    async def test_detect_maps_to_sdrinfo(self):
        control = HackRFDevice()
        control.detect = AsyncMock(return_value=dict(PARSED_OK))
        dev = HackRFSDRDevice(control=control)

        info = await dev.detect()
        assert isinstance(info, SDRInfo)
        assert info.detected is True
        assert info.serial == PARSED_OK["serial"]
        assert info.firmware == "2024.02.1"
        assert info.api_version == "1.08"
        assert info.name == "HackRF One"
        assert info.freq_max_hz == HACKRF_FREQ_MAX_HZ
        assert info.has_tx is True and info.has_bias_tee is True
        assert dev.info is info  # ABC info property returns the cached SDRInfo

    @pytest.mark.asyncio
    async def test_detect_error_dict_is_undetected(self):
        control = HackRFDevice()
        control.detect = AsyncMock(
            return_value={"connected": False, "error": "hackrf_info not found on PATH"}
        )
        dev = HackRFSDRDevice(control=control)

        info = await dev.detect()
        assert isinstance(info, SDRInfo)
        assert info.detected is False
        assert info.freq_max_hz == 0
        assert info.error

    @pytest.mark.asyncio
    async def test_detect_none_is_undetected(self):
        control = HackRFDevice()
        control.detect = AsyncMock(return_value=None)
        dev = HackRFSDRDevice(control=control)

        info = await dev.detect()
        assert info.detected is False


class TestSweep:
    @pytest.mark.asyncio
    async def test_sweep_graceful_without_binary(self):
        dev = HackRFSDRDevice()
        with patch("hackrf_addon.sdr_device.shutil.which", return_value=None):
            result = await dev.sweep(2_400_000_000, 2_410_000_000, bin_width_hz=1_000_000)
        assert isinstance(result, SweepResult)
        assert result.points == []
        assert result.freq_start_hz == 2_400_000_000
        assert result.freq_end_hz == 2_410_000_000
        assert result.bin_width_hz == 1_000_000

    def test_parse_sweep_csv(self):
        pts = HackRFSDRDevice._parse_sweep_csv(
            SWEEP_CSV, 2_400_000_000, 2_410_000_000, 123.0,
        )
        assert pts and all(isinstance(p, SweepPoint) for p in pts)
        freqs = [p.freq_hz for p in pts]
        assert freqs == sorted(freqs)  # ascending
        by_freq = {p.freq_hz: p.power_dbm for p in pts}
        assert by_freq.get(2_402_000_000) == -20.2  # strong bin, first band
        assert by_freq.get(2_406_000_000) == -30.5  # strong bin, second band

    def test_parse_sweep_csv_filters_out_of_band(self):
        pts = HackRFSDRDevice._parse_sweep_csv(
            SWEEP_CSV, 2_400_000_000, 2_403_000_000, 0.0,
        )
        assert pts  # some in-band points survive
        assert all(2_400_000_000 <= p.freq_hz <= 2_403_000_000 for p in pts)


class TestTuneStopReadIQ:
    @pytest.mark.asyncio
    async def test_tune_records_then_stop_clears(self):
        dev = HackRFSDRDevice()
        await dev.tune(915_000_000, sample_rate=8_000_000, bandwidth=1_750_000)
        assert dev.tuned_frequency == 915_000_000
        await dev.stop()
        assert dev.tuned_frequency == 0

    @pytest.mark.asyncio
    async def test_read_iq_requires_tune(self):
        dev = HackRFSDRDevice()
        with pytest.raises(RuntimeError):
            await dev.read_iq(1024)

    @pytest.mark.asyncio
    async def test_read_iq_zero_samples_is_empty(self):
        dev = HackRFSDRDevice()
        arr = await dev.read_iq(0)
        assert len(arr) == 0

    @pytest.mark.asyncio
    async def test_read_iq_graceful_without_binary(self):
        dev = HackRFSDRDevice()
        await dev.tune(1_090_000_000)
        with patch("hackrf_addon.sdr_device.shutil.which", return_value=None):
            with pytest.raises(NotImplementedError):
                await dev.read_iq(1024)

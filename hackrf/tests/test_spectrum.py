# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SpectrumAnalyzer — sweep parsing, start/stop, data retrieval."""

import asyncio
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from hackrf_addon.spectrum import SpectrumAnalyzer
from hackrf_addon.signal_db import SignalDatabase

# Real hackrf_sweep CSV line for testing
REAL_SWEEP_LINE = (
    "2026-03-17, 00:10:57.688868, 88000000, 93000000, 454545.45, 44, "
    "-53.50, -53.46, -58.04, -47.90, -40.15, -39.70, -43.33, -44.84, "
    "-57.48, -45.33, -45.03"
)


class TestParseSweepLine:
    """Tests for _parse_sweep_line CSV parsing."""

    def test_parse_real_sweep_line(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line(REAL_SWEEP_LINE)
        assert result is not None
        assert len(result) == 11  # 11 power values in the CSV line

    def test_parsed_frequencies_are_correct(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line(REAL_SWEEP_LINE)
        # freq_low=88000000, bin_width=454545 (int), first bin center = 88000000 + 227272
        bin_width_int = 454545
        expected_first_center = 88000000 + bin_width_int // 2
        assert result[0]["freq_hz"] == expected_first_center

    def test_parsed_power_values(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line(REAL_SWEEP_LINE)
        assert result[0]["power_dbm"] == pytest.approx(-53.50)
        assert result[1]["power_dbm"] == pytest.approx(-53.46)
        assert result[5]["power_dbm"] == pytest.approx(-39.70)

    def test_parsed_timestamps_present(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line(REAL_SWEEP_LINE)
        for m in result:
            assert "timestamp" in m
            assert m["timestamp"] > 0

    def test_bin_centers_evenly_spaced(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line(REAL_SWEEP_LINE)
        bin_width = 454545
        for i in range(1, len(result)):
            diff = result[i]["freq_hz"] - result[i - 1]["freq_hz"]
            assert diff == bin_width

    def test_parse_short_line_returns_none(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line("2026-03-17, 12:00:00, 88000000")
        assert result is None

    def test_parse_empty_line_returns_none(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line("")
        assert result is None

    def test_parse_invalid_freq_returns_none(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line("2026-03-17, 12:00:00, abc, 93000000, 500000, 44, -50.0")
        assert result is None

    def test_parse_zero_bin_width_returns_none(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line("2026-03-17, 12:00:00, 88000000, 93000000, 0, 44, -50.0")
        assert result is None

    def test_parse_negative_bin_width_returns_none(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line("2026-03-17, 12:00:00, 88000000, 93000000, -1000, 44, -50.0")
        assert result is None

    def test_parse_single_power_value(self):
        sa = SpectrumAnalyzer()
        result = sa._parse_sweep_line("2026-03-17, 12:00:00, 100000000, 101000000, 500000, 1, -45.0")
        assert result is not None
        assert len(result) == 1
        assert result[0]["power_dbm"] == pytest.approx(-45.0)

    def test_parse_with_trailing_spaces(self):
        sa = SpectrumAnalyzer()
        line = "2026-03-17, 12:00:00, 100000000, 101000000, 500000, 1, -45.0, -42.0  "
        result = sa._parse_sweep_line(line)
        assert result is not None
        assert len(result) == 2

    def test_parse_bad_power_value_skipped(self):
        sa = SpectrumAnalyzer()
        line = "2026-03-17, 12:00:00, 100000000, 101000000, 500000, 3, -45.0, bad, -42.0"
        result = sa._parse_sweep_line(line)
        assert result is not None
        # bad value is skipped but freq still advances
        assert len(result) == 2


class TestSpectrumAnalyzerState:
    """Tests for analyzer state and properties."""

    def test_initial_state(self):
        sa = SpectrumAnalyzer()
        assert not sa.is_running
        assert sa.sweep_count == 0

    def test_get_status(self):
        sa = SpectrumAnalyzer()
        status = sa.get_status()
        assert status["running"] is False
        assert status["sweep_count"] == 0
        assert "measurement_count" in status

    def test_get_data_empty(self):
        sa = SpectrumAnalyzer()
        data = sa.get_data()
        assert data == []

    def test_get_data_from_signal_db(self):
        db = SignalDatabase()
        sa = SpectrumAnalyzer(signal_db=db)
        now = time.time()
        db.store(100_000_000, -40.0, now)
        data = sa.get_data()
        assert len(data) == 1
        assert data[0]["freq_hz"] == 100_000_000


class TestSweepStartStop:
    """Tests for start_sweep and stop_sweep."""

    @pytest.mark.asyncio
    @patch("shutil.which", return_value=None)
    async def test_start_sweep_no_binary(self, mock_which):
        sa = SpectrumAnalyzer()
        result = await sa.start_sweep()
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_stop_sweep_not_running(self):
        sa = SpectrumAnalyzer()
        result = await sa.stop_sweep()
        assert result["success"] is True  # Idempotent — safe to stop when already stopped

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_sweep")
    @patch("asyncio.create_subprocess_exec")
    async def test_start_sweep_success(self, mock_exec, mock_which):
        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_exec.return_value = mock_proc

        sa = SpectrumAnalyzer()
        result = await sa.start_sweep(88, 108, 500000)
        assert result["success"] is True
        assert result["freq_start_mhz"] == 88
        assert result["freq_end_mhz"] == 108
        assert sa._running is True

        # Clean up
        await sa.stop_sweep()

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_sweep")
    @patch("asyncio.create_subprocess_exec")
    async def test_start_sweep_already_running(self, mock_exec, mock_which):
        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_exec.return_value = mock_proc

        sa = SpectrumAnalyzer()
        await sa.start_sweep()
        result = await sa.start_sweep()
        assert result["success"] is False
        assert "already" in result["error"].lower()
        await sa.stop_sweep()

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/bin/hackrf_sweep")
    @patch("asyncio.create_subprocess_exec", side_effect=OSError("fail"))
    async def test_start_sweep_exec_error(self, mock_exec, mock_which):
        sa = SpectrumAnalyzer()
        result = await sa.start_sweep()
        assert result["success"] is False

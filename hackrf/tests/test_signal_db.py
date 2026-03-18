# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SignalDatabase — store, query, peaks, ring buffer."""

import time
import pytest

from hackrf_addon.signal_db import SignalDatabase, SignalMeasurement


class TestSignalDatabaseInit:
    """Tests for initialization."""

    def test_empty_on_init(self):
        db = SignalDatabase()
        assert db.count == 0

    def test_custom_max_size(self):
        db = SignalDatabase(max_size=10)
        assert db.count == 0


class TestStore:
    """Tests for single measurement storage."""

    def test_store_single(self):
        db = SignalDatabase()
        db.store(100_000_000, -40.0)
        assert db.count == 1

    def test_store_with_timestamp(self):
        db = SignalDatabase()
        db.store(100_000_000, -40.0, timestamp=1000.0)
        results = db.query()
        assert results[0]["timestamp"] == 1000.0

    def test_store_auto_timestamp(self):
        db = SignalDatabase()
        before = time.time()
        db.store(100_000_000, -40.0)
        after = time.time()
        results = db.query()
        assert before <= results[0]["timestamp"] <= after

    def test_store_preserves_freq(self):
        db = SignalDatabase()
        db.store(433_920_000, -55.5)
        results = db.query()
        assert results[0]["freq_hz"] == 433_920_000

    def test_store_preserves_power(self):
        db = SignalDatabase()
        db.store(100_000_000, -72.3)
        results = db.query()
        assert results[0]["power_dbm"] == pytest.approx(-72.3)


class TestStoreBatch:
    """Tests for batch storage."""

    def test_store_batch_multiple(self):
        db = SignalDatabase()
        measurements = [
            {"freq_hz": 100_000_000, "power_dbm": -40.0},
            {"freq_hz": 101_000_000, "power_dbm": -45.0},
            {"freq_hz": 102_000_000, "power_dbm": -50.0},
        ]
        db.store_batch(measurements)
        assert db.count == 3

    def test_store_batch_with_timestamps(self):
        db = SignalDatabase()
        measurements = [
            {"freq_hz": 100_000_000, "power_dbm": -40.0, "timestamp": 1000.0},
            {"freq_hz": 101_000_000, "power_dbm": -45.0, "timestamp": 1001.0},
        ]
        db.store_batch(measurements)
        results = db.query()
        assert results[0]["timestamp"] == 1000.0
        assert results[1]["timestamp"] == 1001.0

    def test_store_batch_empty(self):
        db = SignalDatabase()
        db.store_batch([])
        assert db.count == 0


class TestMeasurementCap:
    """Tests for ring buffer max size enforcement."""

    def test_cap_enforced(self):
        db = SignalDatabase(max_size=5)
        for i in range(10):
            db.store(100_000_000 + i, -40.0 - i)
        assert db.count == 5

    def test_oldest_dropped(self):
        db = SignalDatabase(max_size=3)
        db.store(100_000_000, -40.0, timestamp=1.0)
        db.store(200_000_000, -50.0, timestamp=2.0)
        db.store(300_000_000, -60.0, timestamp=3.0)
        db.store(400_000_000, -70.0, timestamp=4.0)
        results = db.query()
        freqs = [r["freq_hz"] for r in results]
        assert 100_000_000 not in freqs
        assert 400_000_000 in freqs


class TestQuery:
    """Tests for query filtering."""

    def _populate(self):
        db = SignalDatabase()
        db.store(100_000_000, -40.0, timestamp=100.0)
        db.store(200_000_000, -50.0, timestamp=200.0)
        db.store(300_000_000, -60.0, timestamp=300.0)
        return db

    def test_query_all(self):
        db = self._populate()
        results = db.query()
        assert len(results) == 3

    def test_query_freq_start(self):
        db = self._populate()
        results = db.query(freq_start=200_000_000)
        assert len(results) == 2
        assert all(r["freq_hz"] >= 200_000_000 for r in results)

    def test_query_freq_end(self):
        db = self._populate()
        results = db.query(freq_end=200_000_000)
        assert len(results) == 2
        assert all(r["freq_hz"] <= 200_000_000 for r in results)

    def test_query_freq_range(self):
        db = self._populate()
        results = db.query(freq_start=150_000_000, freq_end=250_000_000)
        assert len(results) == 1
        assert results[0]["freq_hz"] == 200_000_000

    def test_query_since(self):
        db = self._populate()
        results = db.query(since=150.0)
        assert len(results) == 2

    def test_query_combined_filters(self):
        db = self._populate()
        results = db.query(freq_start=100_000_000, freq_end=250_000_000, since=150.0)
        assert len(results) == 1
        assert results[0]["freq_hz"] == 200_000_000


class TestGetPeaks:
    """Tests for peak detection."""

    def test_peaks_above_threshold(self):
        db = SignalDatabase()
        db.store(100_000_000, -20.0, timestamp=1.0)
        db.store(200_000_000, -50.0, timestamp=1.0)
        db.store(300_000_000, -10.0, timestamp=1.0)
        peaks = db.get_peaks(threshold_dbm=-30.0)
        assert len(peaks) == 2
        freqs = [p["freq_hz"] for p in peaks]
        assert 100_000_000 in freqs
        assert 300_000_000 in freqs

    def test_peaks_sorted_by_power(self):
        db = SignalDatabase()
        db.store(100_000_000, -20.0)
        db.store(200_000_000, -10.0)
        peaks = db.get_peaks(threshold_dbm=-30.0)
        assert peaks[0]["power_dbm"] > peaks[1]["power_dbm"]

    def test_peaks_latest_per_freq(self):
        db = SignalDatabase()
        db.store(100_000_000, -20.0, timestamp=1.0)
        db.store(100_000_000, -15.0, timestamp=2.0)
        peaks = db.get_peaks(threshold_dbm=-30.0)
        assert len(peaks) == 1
        assert peaks[0]["timestamp"] == 2.0

    def test_no_peaks(self):
        db = SignalDatabase()
        db.store(100_000_000, -80.0)
        peaks = db.get_peaks(threshold_dbm=-30.0)
        assert len(peaks) == 0


class TestGetLatestSweep:
    """Tests for latest sweep retrieval."""

    def test_empty_db(self):
        db = SignalDatabase()
        result = db.get_latest_sweep()
        assert result == []

    def test_returns_recent_measurements(self):
        db = SignalDatabase()
        now = time.time()
        db.store(100_000_000, -40.0, timestamp=now)
        db.store(200_000_000, -50.0, timestamp=now + 0.1)
        result = db.get_latest_sweep()
        assert len(result) == 2

    def test_excludes_old_measurements(self):
        db = SignalDatabase()
        now = time.time()
        db.store(100_000_000, -40.0, timestamp=now - 10.0)
        db.store(200_000_000, -50.0, timestamp=now)
        result = db.get_latest_sweep()
        assert len(result) == 1


class TestClear:
    """Tests for clearing the database."""

    def test_clear(self):
        db = SignalDatabase()
        db.store(100_000_000, -40.0)
        db.store(200_000_000, -50.0)
        assert db.count == 2
        db.clear()
        assert db.count == 0

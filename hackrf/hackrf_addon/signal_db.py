# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""In-memory signal measurement database for HackRF spectrum data."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalMeasurement:
    """A single spectrum measurement."""
    freq_hz: int
    power_dbm: float
    timestamp: float = field(default_factory=time.time)


class SignalDatabase:
    """In-memory ring buffer for spectrum measurements.

    Stores recent measurements in a deque with configurable max size.
    Provides query and peak-detection methods for the spectrum analyzer.
    """

    def __init__(self, max_size: int = 100_000):
        self._measurements: deque[SignalMeasurement] = deque(maxlen=max_size)

    def store(self, freq_hz: int, power_dbm: float, timestamp: float | None = None) -> None:
        """Store a single measurement."""
        ts = timestamp if timestamp is not None else time.time()
        self._measurements.append(SignalMeasurement(
            freq_hz=freq_hz,
            power_dbm=power_dbm,
            timestamp=ts,
        ))

    def store_batch(self, measurements: list[dict]) -> None:
        """Store a batch of measurements efficiently.

        Each dict should have 'freq_hz' and 'power_dbm', optionally 'timestamp'.
        """
        ts = time.time()
        for m in measurements:
            self._measurements.append(SignalMeasurement(
                freq_hz=m["freq_hz"],
                power_dbm=m["power_dbm"],
                timestamp=m.get("timestamp", ts),
            ))

    def query(
        self,
        freq_start: int | None = None,
        freq_end: int | None = None,
        since: float | None = None,
    ) -> list[dict]:
        """Query measurements within frequency and time range.

        Args:
            freq_start: Minimum frequency in Hz (inclusive).
            freq_end: Maximum frequency in Hz (inclusive).
            since: Only return measurements after this Unix timestamp.

        Returns:
            List of dicts with freq_hz, power_dbm, timestamp.
        """
        results = []
        for m in self._measurements:
            if since is not None and m.timestamp < since:
                continue
            if freq_start is not None and m.freq_hz < freq_start:
                continue
            if freq_end is not None and m.freq_hz > freq_end:
                continue
            results.append({
                "freq_hz": m.freq_hz,
                "power_dbm": m.power_dbm,
                "timestamp": m.timestamp,
            })
        return results

    def get_peaks(self, threshold_dbm: float = -30.0) -> list[dict]:
        """Return frequencies with power above threshold.

        Groups by frequency and returns the latest measurement for each
        frequency that exceeds the threshold.

        Args:
            threshold_dbm: Minimum power level in dBm.

        Returns:
            List of dicts with freq_hz, power_dbm, timestamp sorted by power descending.
        """
        # Keep latest measurement per frequency
        latest: dict[int, SignalMeasurement] = {}
        for m in self._measurements:
            if m.power_dbm >= threshold_dbm:
                existing = latest.get(m.freq_hz)
                if existing is None or m.timestamp > existing.timestamp:
                    latest[m.freq_hz] = m

        peaks = [
            {"freq_hz": m.freq_hz, "power_dbm": m.power_dbm, "timestamp": m.timestamp}
            for m in latest.values()
        ]
        peaks.sort(key=lambda p: p["power_dbm"], reverse=True)
        return peaks

    def get_latest_sweep(self) -> list[dict]:
        """Return the most recent complete sweep (all measurements sharing the latest timestamp batch).

        Returns measurements from the last 2 seconds as a single sweep.
        """
        if not self._measurements:
            return []
        latest_ts = self._measurements[-1].timestamp
        cutoff = latest_ts - 2.0  # Within 2 seconds of latest
        results = []
        for m in reversed(self._measurements):
            if m.timestamp < cutoff:
                break
            results.append({
                "freq_hz": m.freq_hz,
                "power_dbm": m.power_dbm,
                "timestamp": m.timestamp,
            })
        results.reverse()
        return results

    def clear(self) -> None:
        """Clear all stored measurements."""
        self._measurements.clear()

    @property
    def count(self) -> int:
        """Number of stored measurements."""
        return len(self._measurements)

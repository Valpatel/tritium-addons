# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ISM band monitor for detecting device transmissions.

Monitors ISM (Industrial, Scientific, Medical) frequency bands for
radio transmissions from common devices: garage door openers, car key fobs,
weather stations, LoRa devices, Zigbee, Z-Wave, TPMS sensors, etc.

Uses hackrf_sweep for continuous scanning and hackrf_transfer for
targeted captures when signals are detected.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("hackrf.decoders.ism")

# ISM band definitions (name, center_mhz, bandwidth_mhz, common devices)
ISM_BANDS: list[dict] = [
    {
        "name": "315 MHz",
        "center_mhz": 315,
        "start_mhz": 310,
        "end_mhz": 320,
        "devices": ["TPMS", "garage doors", "car key fobs", "remote controls"],
    },
    {
        "name": "433 MHz",
        "center_mhz": 433,
        "start_mhz": 430,
        "end_mhz": 440,
        "devices": ["TPMS (EU)", "weather stations", "remote controls", "car key fobs",
                     "wireless doorbells", "tire monitors"],
    },
    {
        "name": "868 MHz",
        "center_mhz": 868,
        "start_mhz": 863,
        "end_mhz": 870,
        "devices": ["LoRa (EU)", "Z-Wave (EU)", "smart meters", "alarm systems"],
    },
    {
        "name": "915 MHz",
        "center_mhz": 915,
        "start_mhz": 902,
        "end_mhz": 928,
        "devices": ["LoRa (US)", "Z-Wave (US)", "Zigbee", "ISM devices",
                     "smart meters", "RFID"],
    },
]


@dataclass
class ISMTransmission:
    """A detected ISM band transmission."""
    timestamp: float
    freq_hz: int
    power_dbm: float
    band: str               # Which ISM band
    duration_s: float = 0.0  # Estimated duration if available
    device_id: str = ""     # Fingerprint/pattern hash

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "freq_hz": self.freq_hz,
            "freq_mhz": round(self.freq_hz / 1_000_000, 3),
            "power_dbm": round(self.power_dbm, 1),
            "band": self.band,
            "duration_s": round(self.duration_s, 3),
            "device_id": self.device_id,
            "age_seconds": round(time.time() - self.timestamp, 1),
        }


@dataclass
class ISMDevice:
    """A tracked ISM band device identified by transmission pattern."""
    device_id: str
    band: str
    freq_hz: int
    first_seen: float
    last_seen: float
    transmission_count: int = 0
    avg_power_dbm: float = -100.0
    avg_interval_s: float = 0.0
    classification: str = "unknown"  # Type guess based on pattern

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "band": self.band,
            "freq_hz": self.freq_hz,
            "freq_mhz": round(self.freq_hz / 1_000_000, 3),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "transmission_count": self.transmission_count,
            "avg_power_dbm": round(self.avg_power_dbm, 1),
            "avg_interval_s": round(self.avg_interval_s, 1),
            "classification": self.classification,
            "age_seconds": round(time.time() - self.last_seen, 1),
        }


class ISMBandMonitor:
    """Continuous ISM band monitor.

    Uses hackrf_sweep to scan ISM bands and detect signal activity.
    Tracks unique devices by their transmission frequency and pattern.
    """

    def __init__(self, threshold_dbm: float = -50.0):
        self._running = False
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task | None = None
        self._threshold_dbm = threshold_dbm
        self._transmissions: deque[ISMTransmission] = deque(maxlen=50_000)
        self._devices: dict[str, ISMDevice] = {}
        self._scan_count: int = 0
        self._last_scan_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    async def start_monitoring(self, threshold_dbm: float | None = None) -> dict:
        """Start continuous ISM band monitoring.

        Uses hackrf_sweep to scan all ISM bands continuously.

        Args:
            threshold_dbm: Signal detection threshold in dBm.

        Returns:
            Status dict.
        """
        if self._running:
            return {"success": False, "error": "ISM monitoring already running"}

        if threshold_dbm is not None:
            self._threshold_dbm = threshold_dbm

        import shutil
        if not shutil.which("hackrf_sweep"):
            return {"success": False, "error": "hackrf_sweep not found on PATH"}

        self._running = True
        self._monitor_task = asyncio.create_task(self._sweep_loop())

        log.info(f"ISM band monitoring started (threshold={self._threshold_dbm} dBm)")
        return {
            "success": True,
            "bands": [b["name"] for b in ISM_BANDS],
            "threshold_dbm": self._threshold_dbm,
        }

    async def stop_monitoring(self) -> dict:
        """Stop ISM band monitoring."""
        if not self._running:
            return {"success": False, "error": "ISM monitoring not running"}

        self._running = False

        # Stop sweep subprocess
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                if self._process:
                    self._process.kill()
            self._process = None

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        log.info(f"ISM monitoring stopped: {self._scan_count} scans, "
                 f"{len(self._transmissions)} transmissions, "
                 f"{len(self._devices)} unique devices")
        return {
            "success": True,
            "scan_count": self._scan_count,
            "total_transmissions": len(self._transmissions),
            "unique_devices": len(self._devices),
        }

    async def _sweep_loop(self) -> None:
        """Background loop: run hackrf_sweep over ISM bands."""
        try:
            while self._running:
                # Build frequency range covering all ISM bands
                # Sweep each band separately for better resolution
                for band in ISM_BANDS:
                    if not self._running:
                        break

                    try:
                        await self._sweep_band(band)
                    except Exception as e:
                        log.warning(f"Sweep error on {band['name']}: {e}")
                        await asyncio.sleep(1.0)

                self._scan_count += 1
                self._last_scan_time = time.time()

                # Brief pause between full cycles
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    async def _sweep_band(self, band: dict) -> None:
        """Run a single hackrf_sweep over one ISM band and parse results."""
        cmd = [
            "hackrf_sweep",
            "-f", f"{band['start_mhz']}:{band['end_mhz']}",
            "-w", "100000",  # 100 kHz bin width for good resolution
            "-1",            # One sweep only
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=30.0,
        )

        if proc.returncode != 0:
            return

        output = stdout.decode(errors="replace")
        ts = time.time()

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            self._parse_sweep_line(line, band["name"], ts)

    def _parse_sweep_line(self, line: str, band_name: str, timestamp: float) -> None:
        """Parse a hackrf_sweep CSV line and detect signals above threshold.

        Format: date, time, freq_low_hz, freq_high_hz, bin_width_hz, num_samples, dB1, dB2, ...
        """
        parts = line.split(",")
        if len(parts) < 7:
            return

        try:
            freq_low = int(parts[2].strip())
            freq_high = int(parts[3].strip())
            bin_width = float(parts[4].strip())
            power_values = parts[6:]
        except (ValueError, IndexError):
            return

        if bin_width <= 0:
            return

        bin_width_int = int(bin_width)
        freq = freq_low

        for pv in power_values:
            pv = pv.strip()
            if not pv:
                continue
            try:
                power_dbm = float(pv)
            except ValueError:
                freq += bin_width_int
                continue

            center_freq = freq + bin_width_int // 2

            if power_dbm >= self._threshold_dbm:
                # Signal detected above threshold
                device_id = self._fingerprint(center_freq, power_dbm)

                tx = ISMTransmission(
                    timestamp=timestamp,
                    freq_hz=center_freq,
                    power_dbm=power_dbm,
                    band=band_name,
                    device_id=device_id,
                )
                self._transmissions.append(tx)
                self._update_device(device_id, tx, band_name)

            freq += bin_width_int

    def _fingerprint(self, freq_hz: int, power_dbm: float) -> str:
        """Generate a device fingerprint from frequency.

        Groups signals by frequency (within 50 kHz) to track
        the same transmitter across sweeps.

        Args:
            freq_hz: Signal frequency in Hz.
            power_dbm: Signal power.

        Returns:
            Device ID string.
        """
        # Round frequency to nearest 50 kHz to group same-transmitter signals
        freq_rounded = round(freq_hz / 50_000) * 50_000
        return f"ism_{freq_rounded // 1000}kHz"

    def _update_device(self, device_id: str, tx: ISMTransmission, band_name: str) -> None:
        """Update or create a device registry entry."""
        now = time.time()
        if device_id in self._devices:
            dev = self._devices[device_id]
            # Update running average of power
            n = dev.transmission_count
            dev.avg_power_dbm = (dev.avg_power_dbm * n + tx.power_dbm) / (n + 1)
            # Update interval estimate
            if n > 0:
                interval = now - dev.last_seen
                dev.avg_interval_s = (dev.avg_interval_s * (n - 1) + interval) / n
            dev.last_seen = now
            dev.transmission_count += 1
        else:
            self._devices[device_id] = ISMDevice(
                device_id=device_id,
                band=band_name,
                freq_hz=tx.freq_hz,
                first_seen=now,
                last_seen=now,
                transmission_count=1,
                avg_power_dbm=tx.power_dbm,
                classification=self._classify_device(tx.freq_hz, band_name),
            )
            log.info(f"New ISM device: {device_id} on {band_name}")

    def _classify_device(self, freq_hz: int, band_name: str) -> str:
        """Classify a device based on frequency and band.

        Basic heuristic classification — can be improved with
        ML or pattern analysis.
        """
        freq_mhz = freq_hz / 1_000_000

        if 314.5 <= freq_mhz <= 315.5:
            return "tpms_or_keyfob"
        elif 310 <= freq_mhz <= 312:
            return "garage_door"
        elif 433.5 <= freq_mhz <= 434.5:
            return "weather_station_or_remote"
        elif 433.0 <= freq_mhz <= 434.0:
            return "ism_433"
        elif 868.0 <= freq_mhz <= 868.6:
            return "lora_eu_or_zwave"
        elif 902 <= freq_mhz <= 928:
            if 915.0 <= freq_mhz <= 916.0:
                return "lora_us"
            return "ism_915"
        return "unknown"

    def get_active_devices(self, max_age_s: float = 300.0) -> list[dict]:
        """List recently active ISM devices.

        Args:
            max_age_s: Maximum age in seconds to consider "active" (default 5 min).

        Returns:
            List of device dicts, sorted by last seen (newest first).
        """
        now = time.time()
        active = [
            dev.to_dict()
            for dev in self._devices.values()
            if now - dev.last_seen <= max_age_s
        ]
        active.sort(key=lambda d: d["last_seen"], reverse=True)
        return active

    def get_all_devices(self) -> list[dict]:
        """Return all tracked ISM devices."""
        devices = [dev.to_dict() for dev in self._devices.values()]
        devices.sort(key=lambda d: d["transmission_count"], reverse=True)
        return devices

    def get_transmission_log(self, limit: int = 200) -> list[dict]:
        """Return recent transmission log.

        Args:
            limit: Maximum number of entries.

        Returns:
            List of transmission dicts, newest first.
        """
        txs = list(self._transmissions)
        txs.reverse()
        return [tx.to_dict() for tx in txs[:limit]]

    def get_band_summary(self) -> list[dict]:
        """Return activity summary per ISM band."""
        now = time.time()
        summaries = []
        for band in ISM_BANDS:
            band_devices = [
                d for d in self._devices.values()
                if d.band == band["name"]
            ]
            active_devices = [
                d for d in band_devices
                if now - d.last_seen <= 300.0
            ]
            recent_txs = sum(
                1 for tx in self._transmissions
                if tx.band == band["name"] and now - tx.timestamp <= 60.0
            )
            summaries.append({
                "name": band["name"],
                "center_mhz": band["center_mhz"],
                "start_mhz": band["start_mhz"],
                "end_mhz": band["end_mhz"],
                "common_devices": band["devices"],
                "total_devices": len(band_devices),
                "active_devices": len(active_devices),
                "recent_transmissions_1m": recent_txs,
            })
        return summaries

    def get_status(self) -> dict:
        """Return ISM monitor status."""
        return {
            "running": self._running,
            "threshold_dbm": self._threshold_dbm,
            "scan_count": self._scan_count,
            "last_scan_time": self._last_scan_time,
            "total_transmissions": len(self._transmissions),
            "unique_devices": len(self._devices),
            "active_devices": len(self.get_active_devices()),
            "bands": self.get_band_summary(),
        }

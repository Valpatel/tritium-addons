# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TPMS (Tire Pressure Monitoring System) decoder.

TPMS transmitters broadcast on 315 MHz (US) or 433.92 MHz (EU) using
ASK/OOK modulation. Each sensor has a unique 32-bit ID that persists
for the life of the tire, making TPMS an excellent vehicle tracking signal.

This decoder captures IQ samples via hackrf_transfer, performs envelope
detection, and extracts signal bursts with timing and energy information.
Full packet decode for Schrader/Sensata/Continental protocols is planned.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("hackrf.decoders.tpms")

# TPMS frequencies
TPMS_FREQ_US = 315_000_000      # 315 MHz (North America)
TPMS_FREQ_EU = 433_920_000      # 433.92 MHz (Europe)

# Capture parameters
TPMS_SAMPLE_RATE = 2_000_000    # 2 MSPS — enough for OOK demod
TPMS_BANDWIDTH = 200_000        # TPMS signal bandwidth ~200 kHz

# Detection thresholds
DEFAULT_NOISE_MULT = 3.0        # Signal must be 3x noise floor
MIN_BURST_SAMPLES = 50          # Minimum burst length in samples
MAX_BURST_SAMPLES = 50_000      # Maximum burst length (25 ms at 2 MSPS)
MIN_BURST_GAP = 100             # Minimum gap between bursts in samples

# Known TPMS protocols
TPMS_PROTOCOLS = {
    "schrader": {
        "description": "Schrader EZ-sensor",
        "modulation": "OOK",
        "data_rate_bps": 9600,
        "preamble_bits": 8,
        "id_bits": 32,
        "pressure_bits": 8,
        "temp_bits": 8,
    },
    "sensata": {
        "description": "Sensata/Continental",
        "modulation": "FSK",
        "data_rate_bps": 19200,
        "preamble_bits": 16,
        "id_bits": 32,
        "pressure_bits": 8,
        "temp_bits": 8,
    },
    "continental": {
        "description": "Continental AG",
        "modulation": "ASK",
        "data_rate_bps": 4800,
        "preamble_bits": 12,
        "id_bits": 28,
        "pressure_bits": 8,
        "temp_bits": 8,
    },
}


@dataclass
class TPMSTransmission:
    """A detected TPMS signal burst."""
    timestamp: float
    freq_hz: int
    power_dbm: float
    duration_us: float          # Burst duration in microseconds
    burst_samples: int          # Number of samples in burst
    energy: float               # Total burst energy (arbitrary units)
    sensor_id: str | None = None  # Decoded sensor ID (hex) if available
    pressure_kpa: float | None = None
    temperature_c: float | None = None
    protocol: str | None = None

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "freq_hz": self.freq_hz,
            "freq_mhz": self.freq_hz / 1_000_000,
            "power_dbm": round(self.power_dbm, 1),
            "duration_us": round(self.duration_us, 0),
            "burst_samples": self.burst_samples,
            "energy": round(self.energy, 2),
        }
        if self.sensor_id is not None:
            d["sensor_id"] = self.sensor_id
        if self.pressure_kpa is not None:
            d["pressure_kpa"] = self.pressure_kpa
        if self.temperature_c is not None:
            d["temperature_c"] = self.temperature_c
        if self.protocol is not None:
            d["protocol"] = self.protocol
        return d


@dataclass
class TPMSSensor:
    """A tracked TPMS sensor (one per tire)."""
    sensor_id: str
    first_seen: float
    last_seen: float
    freq_hz: int
    transmission_count: int = 0
    last_pressure_kpa: float | None = None
    last_temperature_c: float | None = None
    protocol: str | None = None

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "freq_hz": self.freq_hz,
            "freq_mhz": self.freq_hz / 1_000_000,
            "transmission_count": self.transmission_count,
            "last_pressure_kpa": self.last_pressure_kpa,
            "last_temperature_c": self.last_temperature_c,
            "protocol": self.protocol,
            "age_seconds": round(time.time() - self.last_seen, 1),
        }


class TPMSDecoder:
    """TPMS signal decoder and vehicle tracker.

    Captures IQ samples on TPMS frequencies, detects OOK signal bursts,
    and extracts transmission characteristics. Each unique sensor ID
    can be used to track a specific vehicle.
    """

    def __init__(self, capture_dir: str | Path | None = None):
        self._capture_dir = Path(capture_dir) if capture_dir else Path("/tmp/hackrf_tpms")
        self._transmissions: deque[TPMSTransmission] = deque(maxlen=10_000)
        self._sensors: dict[str, TPMSSensor] = {}
        self._running = False
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task | None = None
        self._freq_hz: int = TPMS_FREQ_US

    @property
    def is_running(self) -> bool:
        return self._running

    async def capture_tpms(
        self,
        duration_s: float = 30.0,
        freq_hz: int | None = None,
        sample_rate: int = TPMS_SAMPLE_RATE,
    ) -> list[TPMSTransmission]:
        """Capture and decode TPMS transmissions.

        Args:
            duration_s: Capture duration in seconds.
            freq_hz: Center frequency (default 315 MHz US).
            sample_rate: IQ sample rate.

        Returns:
            List of detected TPMS transmissions.
        """
        if freq_hz is None:
            freq_hz = self._freq_hz
        self._freq_hz = freq_hz

        self._capture_dir.mkdir(parents=True, exist_ok=True)
        capture_file = self._capture_dir / f"tpms_{freq_hz // 1_000_000}MHz_{int(time.time())}.raw"

        num_samples = int(sample_rate * duration_s)

        cmd = [
            "hackrf_transfer",
            "-r", str(capture_file),
            "-f", str(freq_hz),
            "-s", str(sample_rate),
            "-l", "40",    # Max LNA gain for weak TPMS signals
            "-g", "40",    # High VGA gain
            "-n", str(num_samples),
        ]

        log.info(f"TPMS capture: {freq_hz / 1_000_000:.3f} MHz, {duration_s}s")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=duration_s + 30.0,
        )

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"hackrf_transfer failed (rc={proc.returncode}): {err}")

        if not capture_file.exists() or capture_file.stat().st_size == 0:
            raise RuntimeError("hackrf_transfer produced no output")

        log.info(f"Captured {capture_file.stat().st_size} bytes, decoding...")
        transmissions = self.decode_packets(capture_file, freq_hz, sample_rate)

        # Clean up capture file (can be large)
        try:
            capture_file.unlink()
        except OSError:
            pass

        return transmissions

    def decode_packets(
        self,
        iq_data: np.ndarray | Path | str,
        freq_hz: int = TPMS_FREQ_US,
        sample_rate: int = TPMS_SAMPLE_RATE,
    ) -> list[TPMSTransmission]:
        """Detect OOK signal bursts in IQ data.

        Pipeline:
        1. Load interleaved int8 IQ from hackrf_transfer
        2. Compute signal envelope (magnitude)
        3. Estimate noise floor
        4. Threshold to find signal bursts above noise
        5. Extract burst timing, energy, and power

        Args:
            iq_data: Complex IQ samples or path to raw IQ file.
            freq_hz: Center frequency for metadata.
            sample_rate: IQ sample rate.

        Returns:
            List of detected transmissions.
        """
        # Load IQ data
        if isinstance(iq_data, (str, Path)):
            raw = np.fromfile(str(iq_data), dtype=np.int8)
            if len(raw) % 2 != 0:
                raw = raw[:len(raw) - 1]
            iq_i = raw[0::2].astype(np.float32) / 128.0
            iq_q = raw[1::2].astype(np.float32) / 128.0
        elif isinstance(iq_data, np.ndarray):
            if np.issubdtype(iq_data.dtype, np.complexfloating):
                iq_i = np.real(iq_data).astype(np.float32)
                iq_q = np.imag(iq_data).astype(np.float32)
            else:
                d = iq_data.astype(np.float32)
                iq_i = d[0::2] / 128.0
                iq_q = d[1::2] / 128.0
        else:
            raise TypeError(f"Unsupported iq_data type: {type(iq_data)}")

        if len(iq_i) < 1000:
            log.warning(f"IQ data too short for TPMS decode: {len(iq_i)} samples")
            return []

        # Step 1: Envelope detection (magnitude)
        envelope = np.sqrt(iq_i ** 2 + iq_q ** 2)

        # Step 2: Moving average smoothing to reduce noise spikes
        # Window size ~ 20 samples at 2 MSPS = 10 us
        window = min(20, len(envelope) // 10)
        if window > 1:
            kernel = np.ones(window) / window
            envelope_smooth = np.convolve(envelope, kernel, mode="same")
        else:
            envelope_smooth = envelope

        # Step 3: Estimate noise floor (median of envelope)
        noise_floor = float(np.median(envelope_smooth))
        threshold = noise_floor * DEFAULT_NOISE_MULT

        if threshold < 0.01:
            threshold = 0.01  # Absolute minimum threshold

        # Step 4: Find signal bursts above threshold
        above = envelope_smooth > threshold
        transmissions: list[TPMSTransmission] = []

        # Find rising and falling edges
        edges = np.diff(above.astype(np.int8))
        rising = np.where(edges == 1)[0]
        falling = np.where(edges == -1)[0]

        # Handle edge cases
        if len(rising) == 0 and len(falling) == 0:
            log.info(f"No TPMS bursts detected (noise_floor={noise_floor:.4f}, threshold={threshold:.4f})")
            return []

        # If signal starts above threshold
        if len(falling) > 0 and (len(rising) == 0 or falling[0] < rising[0]):
            rising = np.insert(rising, 0, 0)

        # If signal ends above threshold
        if len(rising) > len(falling):
            falling = np.append(falling, len(envelope_smooth) - 1)

        ts_base = time.time()
        sample_period = 1.0 / sample_rate

        for i in range(min(len(rising), len(falling))):
            start = rising[i]
            end = falling[i]
            burst_len = end - start

            # Filter by burst length
            if burst_len < MIN_BURST_SAMPLES or burst_len > MAX_BURST_SAMPLES:
                continue

            # Skip if too close to previous burst (merge protection)
            if i > 0 and start - falling[i - 1] < MIN_BURST_GAP:
                continue

            # Extract burst characteristics
            burst_envelope = envelope[start:end]
            peak_power = float(np.max(burst_envelope))
            # Convert linear amplitude to approximate dBm
            # Assuming 50 ohm impedance, full scale = 0 dBm
            if peak_power > 0:
                power_dbm = 20.0 * np.log10(peak_power) - 30.0
            else:
                power_dbm = -100.0

            duration_us = burst_len * sample_period * 1_000_000
            energy = float(np.sum(burst_envelope ** 2))
            burst_time = ts_base + start * sample_period

            # Attempt basic OOK bit extraction for sensor ID
            sensor_id = self._extract_sensor_id(burst_envelope, sample_rate)

            tx = TPMSTransmission(
                timestamp=burst_time,
                freq_hz=freq_hz,
                power_dbm=power_dbm,
                duration_us=duration_us,
                burst_samples=burst_len,
                energy=energy,
                sensor_id=sensor_id,
            )

            transmissions.append(tx)
            self._transmissions.append(tx)

            # Update sensor registry if we got an ID
            if sensor_id:
                self._update_sensor(sensor_id, tx)

        log.info(f"Decoded {len(transmissions)} TPMS bursts "
                 f"(noise={noise_floor:.4f}, threshold={threshold:.4f})")
        return transmissions

    def _extract_sensor_id(
        self,
        burst_envelope: np.ndarray,
        sample_rate: int,
    ) -> str | None:
        """Attempt to extract a sensor ID from an OOK burst.

        Uses bit-slicing at common TPMS data rates to find
        consistent patterns that could be sensor IDs.

        Returns hex string sensor ID or None if decode fails.
        """
        # Try common TPMS data rates
        for data_rate in [9600, 4800, 19200]:
            samples_per_bit = sample_rate / data_rate

            if len(burst_envelope) < samples_per_bit * 40:
                continue  # Need at least 40 bits (preamble + ID)

            # Threshold the burst into binary
            mid = float(np.median(burst_envelope))
            bits = []
            pos = 0.0
            while pos + samples_per_bit <= len(burst_envelope):
                start_idx = int(pos)
                end_idx = int(pos + samples_per_bit)
                segment = burst_envelope[start_idx:end_idx]
                bit_val = 1 if float(np.mean(segment)) > mid else 0
                bits.append(bit_val)
                pos += samples_per_bit

            if len(bits) < 40:
                continue

            # Look for preamble pattern (alternating 0101... or 1010...)
            preamble_start = None
            for j in range(len(bits) - 8):
                # Check for alternating pattern
                alternating = True
                for k in range(7):
                    if bits[j + k] == bits[j + k + 1]:
                        alternating = False
                        break
                if alternating:
                    preamble_start = j + 8  # Skip preamble
                    break

            if preamble_start is None:
                continue

            # Extract 32 bits after preamble as sensor ID
            if preamble_start + 32 <= len(bits):
                id_bits = bits[preamble_start:preamble_start + 32]
                # Convert to hex
                id_val = 0
                for b in id_bits:
                    id_val = (id_val << 1) | b
                if id_val != 0 and id_val != 0xFFFFFFFF:
                    return f"{id_val:08x}"

        return None

    def _update_sensor(self, sensor_id: str, tx: TPMSTransmission) -> None:
        """Update or create a sensor registry entry."""
        now = time.time()
        if sensor_id in self._sensors:
            sensor = self._sensors[sensor_id]
            sensor.last_seen = now
            sensor.transmission_count += 1
            if tx.pressure_kpa is not None:
                sensor.last_pressure_kpa = tx.pressure_kpa
            if tx.temperature_c is not None:
                sensor.last_temperature_c = tx.temperature_c
        else:
            self._sensors[sensor_id] = TPMSSensor(
                sensor_id=sensor_id,
                first_seen=now,
                last_seen=now,
                freq_hz=tx.freq_hz,
                transmission_count=1,
                last_pressure_kpa=tx.pressure_kpa,
                last_temperature_c=tx.temperature_c,
                protocol=tx.protocol,
            )
            log.info(f"New TPMS sensor: {sensor_id}")

    async def start_monitoring(
        self,
        freq_hz: int | None = None,
        cycle_s: float = 30.0,
    ) -> dict:
        """Start continuous TPMS monitoring in background.

        Captures IQ data in cycles, decoding each batch.

        Args:
            freq_hz: Monitoring frequency (default 315 MHz US).
            cycle_s: Duration of each capture cycle.

        Returns:
            Status dict.
        """
        if self._running:
            return {"success": False, "error": "TPMS monitoring already running"}

        if freq_hz is not None:
            self._freq_hz = freq_hz

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(cycle_s))

        log.info(f"TPMS monitoring started on {self._freq_hz / 1_000_000:.3f} MHz")
        return {
            "success": True,
            "freq_hz": self._freq_hz,
            "freq_mhz": self._freq_hz / 1_000_000,
            "cycle_seconds": cycle_s,
        }

    async def stop_monitoring(self) -> dict:
        """Stop TPMS monitoring."""
        if not self._running:
            return {"success": False, "error": "TPMS monitoring not running"}

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        log.info("TPMS monitoring stopped")
        return {
            "success": True,
            "total_transmissions": len(self._transmissions),
            "unique_sensors": len(self._sensors),
        }

    async def _monitor_loop(self, cycle_s: float) -> None:
        """Background monitoring loop."""
        try:
            while self._running:
                try:
                    txs = await self.capture_tpms(
                        duration_s=cycle_s,
                        freq_hz=self._freq_hz,
                    )
                    if txs:
                        log.info(f"TPMS cycle: {len(txs)} transmissions, "
                                 f"{len(self._sensors)} unique sensors")
                except RuntimeError as e:
                    log.warning(f"TPMS capture cycle failed: {e}")
                    await asyncio.sleep(5.0)
                except Exception as e:
                    log.error(f"TPMS monitor error: {e}")
                    await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            pass

    def get_sensors(self) -> list[dict]:
        """Return all detected TPMS sensors."""
        return [s.to_dict() for s in self._sensors.values()]

    def get_transmissions(self, limit: int = 100) -> list[dict]:
        """Return recent TPMS transmissions.

        Args:
            limit: Maximum number of transmissions to return.

        Returns:
            List of transmission dicts, newest first.
        """
        txs = list(self._transmissions)
        txs.reverse()
        return [tx.to_dict() for tx in txs[:limit]]

    def get_status(self) -> dict:
        """Return TPMS decoder status."""
        return {
            "running": self._running,
            "freq_hz": self._freq_hz,
            "freq_mhz": self._freq_hz / 1_000_000,
            "total_transmissions": len(self._transmissions),
            "unique_sensors": len(self._sensors),
            "sensors": self.get_sensors(),
        }

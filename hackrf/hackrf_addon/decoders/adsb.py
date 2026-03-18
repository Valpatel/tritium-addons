# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ADS-B (1090 MHz) aircraft decoder for HackRF.

Decodes Mode S transponder messages broadcast by aircraft on 1090 MHz.
Each aircraft transmits its ICAO address (24-bit unique ID), altitude,
position (latitude/longitude via CPR encoding), velocity, and callsign.

ADS-B message types:
- Type 1-4:  Aircraft identification (callsign)
- Type 9-18: Airborne position (lat/lon/altitude via CPR)
- Type 19:   Airborne velocity (ground speed, heading, vertical rate)

Pipeline:
1. Capture 1090 MHz IQ samples via hackrf_transfer (2 MHz sample rate)
2. Compute signal magnitude from I/Q
3. Detect 8us preamble pattern (pulse positions at 0, 1, 3.5, 4.5 us)
4. Extract 112-bit Mode S long message (or 56-bit short)
5. Validate CRC-24 checksum
6. Decode message fields based on downlink format and type code

References:
- ICAO Annex 10, Volume IV
- DO-260B (ADS-B standard)
- https://mode-s.org/decode/
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("hackrf.decoders.adsb")

# ADS-B parameters
ADSB_FREQ_HZ = 1_090_000_000   # 1090 MHz
ADSB_SAMPLE_RATE = 2_000_000   # 2 MSPS
SAMPLES_PER_US = 2             # At 2 MSPS, 1 us = 2 samples

# Preamble: 8 us long, pulses at 0, 1, 3.5, 4.5 us
# In samples (at 2 MSPS): pulses at 0, 2, 7, 9
PREAMBLE_SAMPLES = 16  # 8 us * 2 samples/us
PREAMBLE_PULSE_POSITIONS = [0, 2, 7, 9]
PREAMBLE_QUIET_POSITIONS = [1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15]

# Message lengths
SHORT_MSG_BITS = 56
LONG_MSG_BITS = 112
LONG_MSG_SAMPLES = LONG_MSG_BITS * SAMPLES_PER_US

# CRC-24 generator polynomial for Mode S
CRC24_GENERATOR = 0xFFF409

# ADS-B callsign character lookup (6-bit encoding)
CALLSIGN_CHARS = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"

# CPR NL (Number of Longitude zones) lookup table
# Precomputed for latitudes 0-87 degrees
_NL_TABLE = [
    59, 59, 59, 59, 58, 58, 58, 57, 57, 56, 56, 55, 54, 54, 53, 52,
    51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 38, 37, 36, 34,
    33, 31, 30, 28, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1,
]


@dataclass
class Aircraft:
    """Tracked aircraft from ADS-B messages."""
    icao: str              # 24-bit ICAO address as hex string
    callsign: str = ""
    altitude_ft: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    velocity_kt: float | None = None
    heading: float | None = None
    vertical_rate_fpm: int | None = None
    squawk: str = ""
    on_ground: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    message_count: int = 0
    # CPR position tracking (need both even and odd frames)
    _cpr_even_lat: float = 0.0
    _cpr_even_lon: float = 0.0
    _cpr_even_time: float = 0.0
    _cpr_odd_lat: float = 0.0
    _cpr_odd_lon: float = 0.0
    _cpr_odd_time: float = 0.0

    def to_dict(self) -> dict:
        d: dict = {
            "icao": self.icao,
            "callsign": self.callsign,
            "altitude_ft": self.altitude_ft,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "velocity_kt": self.velocity_kt,
            "heading": self.heading,
            "vertical_rate_fpm": self.vertical_rate_fpm,
            "squawk": self.squawk,
            "on_ground": self.on_ground,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "message_count": self.message_count,
            "age_s": round(time.time() - self.last_seen, 1),
        }
        return d


def crc24(data: bytes) -> int:
    """Compute CRC-24 checksum for Mode S message.

    Args:
        data: Message bytes (without the final 3 CRC bytes).

    Returns:
        24-bit CRC value.
    """
    crc = 0
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= CRC24_GENERATOR
    return crc & 0xFFFFFF


def validate_crc(msg_bytes: bytes) -> bool:
    """Validate CRC-24 of a complete Mode S message (including CRC bytes).

    For DF17 (ADS-B), the last 3 bytes are the CRC. The CRC of the
    entire message (including CRC bytes) should be 0.

    Args:
        msg_bytes: Complete message bytes (7 or 14 bytes).

    Returns:
        True if CRC is valid.
    """
    if len(msg_bytes) < 7:
        return False
    # Compute CRC over the data portion
    data_len = len(msg_bytes) - 3
    computed = crc24(msg_bytes[:data_len])
    # Extract transmitted CRC
    transmitted = (msg_bytes[-3] << 16) | (msg_bytes[-2] << 8) | msg_bytes[-1]
    return computed == transmitted


def _nl(lat: float) -> int:
    """Number of Longitude zones at a given latitude (NL function).

    Used in CPR decoding to determine the number of longitude zones.

    Args:
        lat: Latitude in degrees.

    Returns:
        Number of longitude zones (1-59).
    """
    abs_lat = abs(lat)
    if abs_lat >= 87.0:
        return 1
    idx = int(abs_lat / 1.8)  # ~50 entries covering 0-87 degrees
    if idx >= len(_NL_TABLE):
        return 1
    return _NL_TABLE[idx]


def decode_callsign(data: bytes) -> str:
    """Decode aircraft callsign from ADS-B identification message.

    The callsign is encoded as 8 characters, each using 6 bits,
    packed into bytes 5-10 of the ME field.

    Args:
        data: The 7-byte ME (message extended) field.

    Returns:
        Callsign string (up to 8 characters, stripped).
    """
    if len(data) < 7:
        return ""
    # Pack the relevant bytes into a 48-bit integer
    # Callsign bits start at bit 8 of ME field (after type code)
    bits = 0
    for b in data[:7]:
        bits = (bits << 8) | b

    chars = []
    # Extract 8 characters, 6 bits each, starting from bit position 40 down
    # The type code occupies the top 5 bits of data[0], and the callsign
    # starts at bit 2 of data[0] (within the 56-bit ME field).
    # Actually: ME is 56 bits. TC is bits 1-5. Callsign chars are bits 6-53.
    val = 0
    for b in data:
        val = (val << 8) | b
    # Shift right to align: 56 bits total, TC=5 bits, then 8*6=48 bits of callsign
    # So callsign starts at bit 51 (56-5=51) down to bit 3
    shift = 56 - 5 - 6  # Start of first char
    for _ in range(8):
        if shift < 0:
            break
        idx = (val >> shift) & 0x3F
        if idx < len(CALLSIGN_CHARS):
            ch = CALLSIGN_CHARS[idx]
            if ch != '#':
                chars.append(ch)
        shift -= 6

    return "".join(chars).strip()


def decode_altitude(msg_bytes: bytes) -> int | None:
    """Decode altitude from ADS-B airborne position message.

    Barometric altitude is encoded in bits 41-52 of the full 112-bit message.
    Bit 48 is the Q-bit: if Q=1, altitude = N*25 - 1000 (in feet).

    Args:
        msg_bytes: Full 14-byte message.

    Returns:
        Altitude in feet, or None if cannot decode.
    """
    if len(msg_bytes) < 11:
        return None

    # ME field starts at byte 4 (after DF + ICAO)
    # Altitude is in ME bits 8-19 (12 bits)
    # ME byte 1 bits and ME byte 2 bits
    alt_bits = ((msg_bytes[5] & 0xFF) << 4) | ((msg_bytes[6] >> 4) & 0x0F)

    # Check Q-bit (bit 48 in the full message, which is bit 4 of this 12-bit field)
    q_bit = (alt_bits >> 4) & 1

    if q_bit:
        # Remove Q-bit and compute: N * 25 - 1000
        n = ((alt_bits & 0xFF0) >> 1) | (alt_bits & 0x00F)
        altitude = n * 25 - 1000
        if -1000 <= altitude <= 100000:
            return altitude

    return None


def decode_cpr_position(
    even_lat: float, even_lon: float, even_time: float,
    odd_lat: float, odd_lon: float, odd_time: float,
) -> tuple[float, float] | None:
    """Decode global CPR position from even and odd frames.

    Compact Position Reporting (CPR) requires both an even (F=0) and
    odd (F=1) frame to compute a global position. The more recent
    frame determines which formula variant to use.

    Args:
        even_lat: CPR latitude from even frame (0-1 normalized).
        even_lon: CPR longitude from even frame (0-1 normalized).
        even_time: Timestamp of even frame.
        odd_lat: CPR latitude from odd frame (0-1 normalized).
        odd_lon: CPR longitude from odd frame (0-1 normalized).
        odd_time: Timestamp of odd frame.

    Returns:
        (latitude, longitude) tuple in degrees, or None if invalid.
    """
    import math

    d_lat_even = 360.0 / 60
    d_lat_odd = 360.0 / 59

    # Compute latitude index
    j = int(math.floor(59 * even_lat - 60 * odd_lat + 0.5))

    lat_even = d_lat_even * (j % 60 + even_lat)
    lat_odd = d_lat_odd * (j % 59 + odd_lat)

    # Normalize to [-90, 90]
    if lat_even >= 270:
        lat_even -= 360
    if lat_odd >= 270:
        lat_odd -= 360

    # Check NL consistency
    if _nl(lat_even) != _nl(lat_odd):
        return None  # Frames cross a latitude zone boundary

    # Use the more recent frame
    if even_time >= odd_time:
        lat = lat_even
        nl = _nl(lat)
        ni = max(nl, 1)
        d_lon = 360.0 / ni if ni > 0 else 360.0
        m = int(math.floor(even_lon * (nl - 1) - odd_lon * nl + 0.5))
        lon = d_lon * (m % ni + even_lon)
    else:
        lat = lat_odd
        nl = _nl(lat) - 1
        ni = max(nl, 1)
        d_lon = 360.0 / ni if ni > 0 else 360.0
        m = int(math.floor(even_lon * (nl) - odd_lon * (nl + 1) + 0.5))
        lon = d_lon * (m % ni + odd_lon)

    if lon >= 180:
        lon -= 360

    # Validate ranges
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None

    return (round(lat, 6), round(lon, 6))


def decode_velocity(msg_bytes: bytes) -> dict | None:
    """Decode velocity from ADS-B airborne velocity message (Type 19).

    Supports subtype 1 (ground speed) and subtype 2 (ground speed, supersonic).

    Args:
        msg_bytes: Full 14-byte message.

    Returns:
        Dict with velocity_kt, heading, vertical_rate_fpm, or None.
    """
    import math

    if len(msg_bytes) < 14:
        return None

    # ME field starts at byte 4
    me = msg_bytes[4:11]  # 7 bytes of ME

    subtype = me[0] & 0x07

    if subtype not in (1, 2):
        return None  # Only decode ground speed subtypes

    # East-West velocity
    ew_sign = (me[1] >> 2) & 1  # 0=east, 1=west
    ew_vel = ((me[1] & 0x03) << 8) | me[2]

    # North-South velocity
    ns_sign = (me[3] >> 7) & 1  # 0=north, 1=south
    ns_vel = ((me[3] & 0x7F) << 3) | ((me[4] >> 5) & 0x07)

    if ew_vel == 0 or ns_vel == 0:
        return None

    ew_vel -= 1  # Offset encoding
    ns_vel -= 1

    if subtype == 2:
        ew_vel *= 4  # Supersonic scale
        ns_vel *= 4

    # Apply sign
    vx = -ew_vel if ew_sign else ew_vel
    vy = -ns_vel if ns_sign else ns_vel

    speed = math.sqrt(vx * vx + vy * vy)
    heading = math.degrees(math.atan2(vx, vy)) % 360

    # Vertical rate
    vr_sign = (me[4] >> 3) & 1  # 0=up, 1=down
    vr_raw = ((me[4] & 0x07) << 6) | ((me[5] >> 2) & 0x3F)

    if vr_raw == 0:
        vertical_rate = 0
    else:
        vertical_rate = (vr_raw - 1) * 64
        if vr_sign:
            vertical_rate = -vertical_rate

    return {
        "velocity_kt": round(speed, 1),
        "heading": round(heading, 1),
        "vertical_rate_fpm": vertical_rate,
    }


class ADSBDecoder:
    """ADS-B aircraft decoder for HackRF One.

    Captures 1090 MHz IQ samples, detects Mode S preambles,
    extracts and decodes messages, and maintains a registry
    of tracked aircraft.
    """

    # Stale timeout for ADS-B targets in the TargetTracker (seconds)
    ADSB_STALE_TIMEOUT = 120.0

    def __init__(self, capture_dir: str | Path | None = None):
        self._capture_dir = Path(capture_dir) if capture_dir else Path("/tmp/hackrf_adsb")
        self._aircraft: dict[str, Aircraft] = {}
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._messages_decoded: int = 0
        self._messages_failed_crc: int = 0
        self._preambles_detected: int = 0
        self._start_time: float = 0.0
        self.target_tracker = None  # Set by HackRFAddon.register()

    @property
    def is_running(self) -> bool:
        return self._running

    def get_aircraft(self) -> list[dict]:
        """Return all tracked aircraft as dicts."""
        return [a.to_dict() for a in sorted(
            self._aircraft.values(),
            key=lambda a: a.last_seen,
            reverse=True,
        )]

    def get_aircraft_by_icao(self, icao: str) -> dict | None:
        """Look up a specific aircraft by ICAO hex address."""
        ac = self._aircraft.get(icao.lower())
        return ac.to_dict() if ac else None

    def get_stats(self) -> dict:
        """Return decoder statistics."""
        now = time.time()
        active = sum(1 for a in self._aircraft.values() if now - a.last_seen < 60)
        return {
            "running": self._running,
            "aircraft_total": len(self._aircraft),
            "aircraft_active": active,
            "messages_decoded": self._messages_decoded,
            "messages_failed_crc": self._messages_failed_crc,
            "preambles_detected": self._preambles_detected,
            "uptime_s": round(now - self._start_time, 0) if self._start_time else 0,
        }

    async def start_monitoring(self, cycle_s: float = 10.0) -> dict:
        """Start continuous ADS-B monitoring.

        Captures IQ data in cycles, decoding each batch.

        Args:
            cycle_s: Duration of each capture cycle in seconds.

        Returns:
            Status dict.
        """
        if self._running:
            return {"success": False, "error": "ADS-B monitoring already running"}

        self._running = True
        self._start_time = time.time()
        self._monitor_task = asyncio.create_task(self._monitor_loop(cycle_s))

        log.info("ADS-B monitoring started on 1090 MHz")
        return {
            "success": True,
            "freq_mhz": 1090,
            "sample_rate": ADSB_SAMPLE_RATE,
            "cycle_s": cycle_s,
        }

    async def stop_monitoring(self) -> dict:
        """Stop ADS-B monitoring."""
        if not self._running:
            return {"success": False, "error": "ADS-B monitoring not running"}

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        log.info(f"ADS-B monitoring stopped: {len(self._aircraft)} aircraft, "
                 f"{self._messages_decoded} messages decoded")
        return {
            "success": True,
            "aircraft": len(self._aircraft),
            "messages_decoded": self._messages_decoded,
        }

    async def _monitor_loop(self, cycle_s: float) -> None:
        """Background loop: capture and decode ADS-B."""
        import shutil

        try:
            while self._running:
                if not shutil.which("hackrf_transfer"):
                    log.warning("hackrf_transfer not available, sleeping")
                    await asyncio.sleep(10)
                    continue

                try:
                    await self._capture_and_decode(cycle_s)
                except RuntimeError as e:
                    log.warning(f"ADS-B capture error: {e}")
                    await asyncio.sleep(5)
                except Exception as e:
                    log.error(f"ADS-B monitor error: {e}")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    async def _capture_and_decode(self, duration_s: float) -> None:
        """Capture IQ data and decode ADS-B messages."""
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        capture_file = self._capture_dir / f"adsb_{int(time.time())}.raw"
        num_samples = int(ADSB_SAMPLE_RATE * duration_s)

        cmd = [
            "hackrf_transfer",
            "-r", str(capture_file),
            "-f", str(ADSB_FREQ_HZ),
            "-s", str(ADSB_SAMPLE_RATE),
            "-l", "40",   # Max LNA gain
            "-g", "40",   # High VGA gain
            "-n", str(num_samples),
        ]

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
            raise RuntimeError(f"hackrf_transfer failed: {err}")

        if capture_file.exists() and capture_file.stat().st_size > 0:
            self.decode_file(capture_file)
            try:
                capture_file.unlink()
            except OSError:
                pass

    def decode_file(self, filepath: Path | str) -> list[dict]:
        """Decode ADS-B messages from a raw IQ file.

        Args:
            filepath: Path to raw IQ file from hackrf_transfer.

        Returns:
            List of decoded message dicts.
        """
        raw = np.fromfile(str(filepath), dtype=np.int8)
        return self.decode_iq(raw)

    def decode_iq(self, iq_raw: np.ndarray) -> list[dict]:
        """Decode ADS-B messages from raw interleaved I/Q samples.

        Args:
            iq_raw: Interleaved int8 I/Q samples from hackrf_transfer.

        Returns:
            List of decoded message dicts.
        """
        if len(iq_raw) < 4:
            return []

        # Ensure even length
        if len(iq_raw) % 2 != 0:
            iq_raw = iq_raw[:len(iq_raw) - 1]

        # Compute magnitude (faster than complex conversion for detection)
        i_samples = iq_raw[0::2].astype(np.float32)
        q_samples = iq_raw[1::2].astype(np.float32)
        magnitude = np.sqrt(i_samples ** 2 + q_samples ** 2)

        return self._detect_and_decode(magnitude)

    def _detect_and_decode(self, magnitude: np.ndarray) -> list[dict]:
        """Detect preambles and decode Mode S messages from magnitude signal.

        Args:
            magnitude: Signal magnitude array (1 sample per I/Q pair).

        Returns:
            List of decoded message dicts.
        """
        decoded_messages = []
        min_len = PREAMBLE_SAMPLES + LONG_MSG_BITS * SAMPLES_PER_US
        n = len(magnitude)

        if n < min_len:
            return []

        # Compute noise floor for threshold
        noise_floor = float(np.median(magnitude))
        threshold = max(noise_floor * 2.0, 0.1)

        i = 0
        while i < n - min_len:
            # Quick check: is the signal strong enough at preamble start?
            if magnitude[i] < threshold:
                i += 1
                continue

            # Check preamble pattern
            if not self._check_preamble(magnitude, i, threshold):
                i += 1
                continue

            self._preambles_detected += 1

            # Extract message bits starting after preamble
            msg_start = i + PREAMBLE_SAMPLES
            msg_bits = self._extract_bits(magnitude, msg_start, LONG_MSG_BITS)

            if msg_bits is None:
                i += PREAMBLE_SAMPLES
                continue

            # Convert bits to bytes
            msg_bytes = self._bits_to_bytes(msg_bits)

            # Validate CRC
            if validate_crc(msg_bytes):
                decoded = self._decode_message(msg_bytes)
                if decoded:
                    decoded_messages.append(decoded)
                    self._messages_decoded += 1
            else:
                self._messages_failed_crc += 1

            # Skip past this message
            i = msg_start + LONG_MSG_BITS * SAMPLES_PER_US
            continue

        return decoded_messages

    def _check_preamble(self, mag: np.ndarray, pos: int, threshold: float) -> bool:
        """Check if the signal at pos matches the ADS-B preamble pattern.

        The preamble has pulses at positions 0, 2, 7, 9 (in samples at 2 MSPS)
        and quiet periods between them.

        Args:
            mag: Signal magnitude array.
            pos: Starting sample position.
            threshold: Minimum signal level for a pulse.

        Returns:
            True if preamble pattern matches.
        """
        if pos + PREAMBLE_SAMPLES >= len(mag):
            return False

        # Check that pulse positions are above threshold
        for p in PREAMBLE_PULSE_POSITIONS:
            if mag[pos + p] < threshold:
                return False

        # Check that some quiet positions are below the pulse level
        pulse_min = min(mag[pos + p] for p in PREAMBLE_PULSE_POSITIONS)
        quiet_count = 0
        for p in PREAMBLE_QUIET_POSITIONS:
            if p < PREAMBLE_SAMPLES and mag[pos + p] < pulse_min * 0.5:
                quiet_count += 1

        # At least half of quiet positions should be quiet
        return quiet_count >= len(PREAMBLE_QUIET_POSITIONS) // 2

    def _extract_bits(self, mag: np.ndarray, start: int, num_bits: int) -> list[int] | None:
        """Extract message bits using pulse position modulation.

        Each bit occupies 1 us (2 samples at 2 MSPS).
        Bit 1: first half high, second half low
        Bit 0: first half low, second half high

        Args:
            mag: Signal magnitude array.
            start: Starting sample index.
            num_bits: Number of bits to extract.

        Returns:
            List of bit values (0 or 1), or None if extraction fails.
        """
        end_needed = start + num_bits * SAMPLES_PER_US
        if end_needed > len(mag):
            return None

        bits = []
        for b in range(num_bits):
            idx = start + b * SAMPLES_PER_US
            # Compare first half vs second half of bit period
            first_half = float(mag[idx])
            second_half = float(mag[idx + 1]) if idx + 1 < len(mag) else 0.0
            bits.append(1 if first_half > second_half else 0)

        return bits

    def _bits_to_bytes(self, bits: list[int]) -> bytes:
        """Convert a list of bits to bytes.

        Args:
            bits: List of 0/1 values.

        Returns:
            Bytes object.
        """
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val = (byte_val << 1) | bits[i + j]
                else:
                    byte_val <<= 1
            result.append(byte_val)
        return bytes(result)

    def _decode_message(self, msg_bytes: bytes) -> dict | None:
        """Decode a validated Mode S message.

        Args:
            msg_bytes: CRC-validated message bytes (14 bytes for DF17).

        Returns:
            Decoded message dict, or None if not an ADS-B message.
        """
        if len(msg_bytes) < 7:
            return None

        # Downlink Format (DF) is the first 5 bits
        df = (msg_bytes[0] >> 3) & 0x1F

        # We primarily decode DF17 (ADS-B extended squitter)
        if df != 17:
            return None

        if len(msg_bytes) < 14:
            return None

        # ICAO address (bytes 1-3)
        icao = f"{msg_bytes[1]:02x}{msg_bytes[2]:02x}{msg_bytes[3]:02x}"

        # Type code (first 5 bits of ME field, byte 4)
        tc = (msg_bytes[4] >> 3) & 0x1F

        now = time.time()

        # Get or create aircraft
        ac = self._aircraft.get(icao)
        if ac is None:
            ac = Aircraft(icao=icao, first_seen=now)
            self._aircraft[icao] = ac
        ac.last_seen = now
        ac.message_count += 1

        result = {"icao": icao, "df": df, "tc": tc}

        # Decode based on type code
        if 1 <= tc <= 4:
            # Aircraft identification
            callsign = decode_callsign(msg_bytes[4:11])
            if callsign:
                ac.callsign = callsign
                result["callsign"] = callsign

        elif 9 <= tc <= 18:
            # Airborne position
            altitude = decode_altitude(msg_bytes)
            if altitude is not None:
                ac.altitude_ft = altitude
                result["altitude_ft"] = altitude

            # CPR position
            me = msg_bytes[4:11]
            cpr_flag = (me[2] >> 2) & 1  # 0=even, 1=odd
            cpr_lat = ((me[2] & 0x03) << 15) | (me[3] << 7) | (me[4] >> 1)
            cpr_lon = ((me[4] & 0x01) << 16) | (me[5] << 8) | me[6]
            cpr_lat_norm = cpr_lat / 131072.0  # 2^17
            cpr_lon_norm = cpr_lon / 131072.0

            result["cpr_flag"] = cpr_flag
            result["cpr_lat"] = cpr_lat_norm
            result["cpr_lon"] = cpr_lon_norm

            if cpr_flag == 0:
                ac._cpr_even_lat = cpr_lat_norm
                ac._cpr_even_lon = cpr_lon_norm
                ac._cpr_even_time = now
            else:
                ac._cpr_odd_lat = cpr_lat_norm
                ac._cpr_odd_lon = cpr_lon_norm
                ac._cpr_odd_time = now

            # Try global decode if we have both frames
            if ac._cpr_even_time > 0 and ac._cpr_odd_time > 0:
                # Frames must be within 10 seconds of each other
                if abs(ac._cpr_even_time - ac._cpr_odd_time) < 10.0:
                    pos = decode_cpr_position(
                        ac._cpr_even_lat, ac._cpr_even_lon, ac._cpr_even_time,
                        ac._cpr_odd_lat, ac._cpr_odd_lon, ac._cpr_odd_time,
                    )
                    if pos:
                        ac.latitude, ac.longitude = pos
                        result["latitude"] = pos[0]
                        result["longitude"] = pos[1]

        elif tc == 19:
            # Airborne velocity
            vel = decode_velocity(msg_bytes)
            if vel:
                ac.velocity_kt = vel["velocity_kt"]
                ac.heading = vel["heading"]
                ac.vertical_rate_fpm = vel["vertical_rate_fpm"]
                result.update(vel)

        # Push aircraft with decoded position to the TargetTracker
        if self.target_tracker and ac.latitude is not None and ac.longitude is not None:
            try:
                update_fn = getattr(self.target_tracker, 'update_from_adsb', None)
                if update_fn is None:
                    # Fallback to update_from_mesh if adsb method not available
                    update_fn = self.target_tracker.update_from_mesh
                update_fn({
                    "target_id": f"adsb_{icao}",
                    "name": ac.callsign if ac.callsign else f"ICAO {icao.upper()}",
                    "lat": ac.latitude,
                    "lng": ac.longitude,
                    "alt": float(ac.altitude_ft * 0.3048) if ac.altitude_ft else 0.0,
                    "heading": ac.heading or 0.0,
                    "speed": ac.velocity_kt or 0.0,
                    "alliance": "unknown",
                    "asset_type": "aircraft",
                    "callsign": ac.callsign,
                    "icao": icao,
                    "altitude_ft": ac.altitude_ft,
                    "squawk": ac.squawk,
                })
            except Exception as e:
                log.debug(f"Failed to update target tracker for aircraft {icao}: {e}")

        return result

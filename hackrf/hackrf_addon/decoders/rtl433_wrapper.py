# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""rtl_433 wrapper — decode ISM band device protocols via HackRF.

rtl_433 supports 200+ device protocols including:
- TPMS tire pressure sensors (Schrader, Continental, etc.)
- Weather stations (Acurite, Oregon Scientific, LaCrosse, etc.)
- Garage door openers
- Car key fobs
- Smart home sensors (temperature, humidity, motion)
- Utility meters (AMR)
- Smoke detectors
- Doorbell chimes

We use it with HackRF via the SoapySDR driver.

Usage:
    wrapper = RTL433Wrapper()
    await wrapper.start_monitoring(freq_hz=315000000)
    # ... runs in background, decoding packets ...
    devices = wrapper.get_devices()
    await wrapper.stop_monitoring()
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("hackrf.rtl433")

# Maximum decoded events to keep in memory
MAX_EVENTS = 5000
# Maximum unique devices to track
MAX_DEVICES = 500


@dataclass
class DecodedEvent:
    """A single decoded device transmission."""
    timestamp: float = 0.0
    protocol: str = ""      # e.g. "Schrader-TPMS", "Acurite-Tower"
    model: str = ""         # device model name
    device_id: str = ""     # unique device identifier
    freq_hz: int = 0
    data: dict = field(default_factory=dict)  # all decoded fields

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "protocol": self.protocol,
            "model": self.model,
            "device_id": self.device_id,
            "freq_mhz": round(self.freq_hz / 1e6, 3) if self.freq_hz else 0,
            "data": self.data,
        }


@dataclass
class TrackedDevice:
    """A unique device seen via rtl_433."""
    device_id: str = ""
    protocol: str = ""
    model: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    event_count: int = 0
    last_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "protocol": self.protocol,
            "model": self.model,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "age_s": round(time.time() - self.last_seen, 0),
            "event_count": self.event_count,
            "last_data": self.last_data,
        }


class RTL433Wrapper:
    """Wraps rtl_433 subprocess for ISM band device decoding with HackRF."""

    def __init__(self):
        self._process = None
        self._reader_task = None
        self._running = False
        self._freq_hz = 315000000  # Default: US TPMS
        self._events: deque[DecodedEvent] = deque(maxlen=MAX_EVENTS)
        self._devices: dict[str, TrackedDevice] = {}
        self._rtl433_path = shutil.which("rtl_433")

    @property
    def is_available(self) -> bool:
        return self._rtl433_path is not None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start_monitoring(
        self,
        freq_hz: int = 315000000,
        protocols: list[int] | None = None,
    ) -> dict:
        """Start rtl_433 monitoring on the given frequency.

        Args:
            freq_hz: Center frequency in Hz (315M for US TPMS, 433.92M for EU)
            protocols: List of rtl_433 protocol numbers to enable (None = all)
        """
        if self._running:
            return {"success": False, "error": "Already monitoring"}
        if not self._rtl433_path:
            return {"success": False, "error": "rtl_433 not installed"}

        self._freq_hz = freq_hz

        cmd = [
            self._rtl433_path,
            "-d", "driver=hackrf",
            "-f", str(freq_hz),
            "-F", "json",           # JSON output
            "-M", "level",          # Include signal level
            "-M", "protocol",       # Include protocol number
            "-M", "time:unix",      # Unix timestamps
        ]

        if protocols:
            for p in protocols:
                cmd.extend(["-R", str(p)])

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

        self._running = True
        self._reader_task = asyncio.create_task(self._read_output())
        log.info(f"rtl_433 monitoring started on {freq_hz/1e6:.3f} MHz")
        return {
            "success": True,
            "freq_hz": freq_hz,
            "freq_mhz": freq_hz / 1e6,
        }

    async def stop_monitoring(self) -> dict:
        """Stop rtl_433 monitoring."""
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._process.kill()
            self._process = None
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        return {"success": True, "events": len(self._events), "devices": len(self._devices)}

    def get_events(self, limit: int = 100) -> list[dict]:
        """Get recent decoded events."""
        events = list(self._events)[-limit:]
        return [e.to_dict() for e in events]

    def get_devices(self) -> list[dict]:
        """Get all tracked unique devices."""
        devices = sorted(self._devices.values(), key=lambda d: -d.last_seen)
        return [d.to_dict() for d in devices[:MAX_DEVICES]]

    def get_tpms_sensors(self) -> list[dict]:
        """Get only TPMS tire pressure sensors."""
        return [
            d.to_dict() for d in self._devices.values()
            if "tpms" in d.protocol.lower() or "tire" in d.model.lower()
        ]

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        now = time.time()
        active = sum(1 for d in self._devices.values() if now - d.last_seen < 300)
        protocols = set(d.protocol for d in self._devices.values())
        return {
            "running": self._running,
            "freq_mhz": self._freq_hz / 1e6,
            "total_events": len(self._events),
            "unique_devices": len(self._devices),
            "active_devices": active,
            "protocols_seen": sorted(protocols),
        }

    async def _read_output(self):
        """Background task: read and parse rtl_433 JSON output."""
        if not self._process or not self._process.stdout:
            return

        try:
            while self._running:
                line_bytes = await self._process.stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode(errors="replace").strip()
                if not line or not line.startswith("{"):
                    continue

                try:
                    data = json.loads(line)
                    self._process_event(data)
                except json.JSONDecodeError:
                    continue

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"rtl_433 reader error: {e}")
        finally:
            self._running = False

    def _process_event(self, data: dict):
        """Process a decoded event from rtl_433."""
        now = time.time()

        # Extract common fields
        model = data.get("model", "Unknown")
        protocol = data.get("protocol", model)
        device_id = str(data.get("id", data.get("device", data.get("sensor_id", ""))))

        if not device_id:
            # Generate ID from available fields
            device_id = f"{model}_{hash(json.dumps(data, sort_keys=True)) % 10000}"

        event = DecodedEvent(
            timestamp=data.get("time", now) if isinstance(data.get("time"), (int, float)) else now,
            protocol=protocol,
            model=model,
            device_id=device_id,
            freq_hz=int(data.get("freq", self._freq_hz)),
            data=data,
        )
        self._events.append(event)

        # Update device tracker
        if device_id in self._devices:
            dev = self._devices[device_id]
            dev.last_seen = now
            dev.event_count += 1
            dev.last_data = data
        else:
            self._devices[device_id] = TrackedDevice(
                device_id=device_id,
                protocol=protocol,
                model=model,
                first_seen=now,
                last_seen=now,
                event_count=1,
                last_data=data,
            )

        log.debug(f"Decoded: {model} id={device_id}")

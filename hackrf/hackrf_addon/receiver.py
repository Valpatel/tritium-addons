# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""FM receiver using hackrf_transfer for IQ capture.

Currently captures raw IQ samples to file. FM demodulation can be added later
using numpy/scipy or an external tool like multimon-ng.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("hackrf.receiver")

# Default capture directory
DEFAULT_CAPTURE_DIR = Path("/tmp/hackrf_captures")


class FMReceiver:
    """FM receiver using hackrf_transfer for IQ sample capture.

    Phase 1: Capture raw IQ samples to file via hackrf_transfer.
    Phase 2 (future): Real-time FM demodulation and audio streaming.
    """

    def __init__(self, capture_dir: str | Path | None = None):
        self._process: asyncio.subprocess.Process | None = None
        self._running = False
        self._freq_hz: int = 100_000_000  # Default 100 MHz (FM broadcast)
        self._sample_rate: int = 2_000_000  # 2 MSPS
        self._lna_gain: int = 32  # LNA gain (0-40 dB, 8 dB steps)
        self._vga_gain: int = 20  # VGA gain (0-62 dB, 2 dB steps)
        self._capture_dir = Path(capture_dir) if capture_dir else DEFAULT_CAPTURE_DIR
        self._current_file: Path | None = None
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        """Whether IQ capture is currently active."""
        return self._running and self._process is not None

    @property
    def frequency_hz(self) -> int:
        """Current tuned frequency in Hz."""
        return self._freq_hz

    def tune(self, freq_hz: int) -> dict:
        """Set the receive frequency.

        If capture is running, it must be stopped and restarted.

        Args:
            freq_hz: Center frequency in Hz (1 MHz to 6 GHz).

        Returns:
            Status dict.
        """
        if freq_hz < 1_000_000 or freq_hz > 6_000_000_000:
            return {"success": False, "error": f"Frequency {freq_hz} Hz out of range (1 MHz - 6 GHz)"}

        was_running = self._running
        self._freq_hz = freq_hz
        result = {
            "success": True,
            "freq_hz": freq_hz,
            "freq_mhz": freq_hz / 1_000_000,
            "needs_restart": was_running,
        }
        log.info(f"Tuned to {freq_hz / 1_000_000:.3f} MHz")
        return result

    async def start(
        self,
        freq_hz: int | None = None,
        sample_rate: int | None = None,
        lna_gain: int | None = None,
        vga_gain: int | None = None,
        duration_seconds: int | None = None,
    ) -> dict:
        """Start IQ capture using hackrf_transfer.

        Args:
            freq_hz: Center frequency in Hz (uses current if not specified).
            sample_rate: Sample rate in Hz (default 2 MSPS).
            lna_gain: LNA gain 0-40 dB in 8 dB steps.
            vga_gain: VGA gain 0-62 dB in 2 dB steps.
            duration_seconds: Capture duration in seconds (None = continuous).

        Returns:
            Status dict with capture file path.
        """
        if self._running:
            return {"success": False, "error": "Capture already running"}

        if not shutil.which("hackrf_transfer"):
            return {"success": False, "error": "hackrf_transfer not found on PATH"}

        if freq_hz is not None:
            self._freq_hz = freq_hz
        if sample_rate is not None:
            self._sample_rate = sample_rate
        if lna_gain is not None:
            self._lna_gain = max(0, min(40, lna_gain))
        if vga_gain is not None:
            self._vga_gain = max(0, min(62, vga_gain))

        # Create capture directory
        self._capture_dir.mkdir(parents=True, exist_ok=True)

        # Generate capture filename with frequency and timestamp
        freq_mhz = self._freq_hz / 1_000_000
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._current_file = self._capture_dir / f"iq_{freq_mhz:.1f}MHz_{ts}.raw"

        cmd = [
            "hackrf_transfer",
            "-r", str(self._current_file),
            "-f", str(self._freq_hz),
            "-s", str(self._sample_rate),
            "-l", str(self._lna_gain),
            "-g", str(self._vga_gain),
        ]

        if duration_seconds is not None:
            # hackrf_transfer uses -n for number of samples
            num_samples = self._sample_rate * duration_seconds
            cmd.extend(["-n", str(num_samples)])

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            log.error(f"Failed to start hackrf_transfer: {e}")
            return {"success": False, "error": str(e)}

        self._running = True
        self._start_time = time.time()

        log.info(f"IQ capture started: {freq_mhz:.3f} MHz, {self._sample_rate} SPS -> {self._current_file}")
        return {
            "success": True,
            "freq_hz": self._freq_hz,
            "freq_mhz": freq_mhz,
            "sample_rate": self._sample_rate,
            "lna_gain": self._lna_gain,
            "vga_gain": self._vga_gain,
            "capture_file": str(self._current_file),
        }

    async def stop(self) -> dict:
        """Stop the running IQ capture.

        Returns:
            Status dict with capture file info.
        """
        if not self._running:
            return {"success": False, "error": "No capture running"}

        self._running = False
        duration = time.time() - self._start_time

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None

        file_size = 0
        if self._current_file and self._current_file.exists():
            file_size = self._current_file.stat().st_size

        log.info(f"IQ capture stopped after {duration:.1f}s, {file_size} bytes")
        return {
            "success": True,
            "capture_file": str(self._current_file) if self._current_file else None,
            "file_size_bytes": file_size,
            "duration_seconds": round(duration, 1),
        }

    def get_audio_url(self) -> str | None:
        """Return URL for audio stream (future: WebSocket FM demod endpoint).

        Currently returns None since FM demodulation is not yet implemented.
        """
        # Future: return "/api/addons/hackrf/audio/stream"
        return None

    def get_captures(self) -> list[dict]:
        """List all IQ capture files in the capture directory.

        Returns:
            List of dicts with filename, size, and metadata.
        """
        if not self._capture_dir.exists():
            return []

        captures = []
        for f in sorted(self._capture_dir.glob("iq_*.raw"), reverse=True):
            stat = f.stat()
            captures.append({
                "filename": f.name,
                "path": str(f),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
            })
        return captures

    def get_status(self) -> dict:
        """Return current receiver status."""
        return {
            "running": self.is_running,
            "freq_hz": self._freq_hz,
            "freq_mhz": self._freq_hz / 1_000_000,
            "sample_rate": self._sample_rate,
            "lna_gain": self._lna_gain,
            "vga_gain": self._vga_gain,
            "capture_file": str(self._current_file) if self._current_file else None,
            "duration_seconds": round(time.time() - self._start_time, 1) if self._running else 0,
        }

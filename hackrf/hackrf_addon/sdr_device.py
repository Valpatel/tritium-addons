# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Real-hardware adapter that satisfies the ``tritium_lib.sdr.SDRDevice`` ABC.

Why this exists (control plane vs data plane)
---------------------------------------------
``HackRFDevice`` (``device.py``) is the HackRF **control-plane manager**:
detection, firmware flashing, clock (CLKIN/CLKOUT) config, Opera Cake antenna
switching, bias-tee, diagnostics. Its ``detect()`` deliberately returns a rich
``dict`` that the rest of the addon (``__init__.py``, ``health_check``,
``_poll_loop``, ``router``) consumes as a dict. That contract does **not** match
the data-plane ``SDRDevice`` ABC (``detect() -> SDRInfo``, ``sweep() ->
SweepResult``, ``tune``, ``stop``, ``read_iq``), and forcing ``HackRFDevice`` to
inherit the ABC would either break every dict consumer or lie about return
types. So ``HackRFDevice`` stays the control plane, unchanged.

``HackRFSDRDevice`` is the **data-plane** adapter that genuinely implements the
ABC against real HackRF CLI tools, so the one real SDR backend satisfies the
contract (previously only ``SimulatedSDR`` did):

  * ``detect()``  -> ``hackrf_info`` (reused via ``HackRFDevice``) mapped to ``SDRInfo``
  * ``sweep()``   -> a single ``hackrf_sweep`` pass parsed into ``SweepResult``
  * ``tune()``    -> records tuning params for a subsequent ``read_iq()``
  * ``stop()``    -> terminates any in-flight sweep subprocess
  * ``read_iq()`` -> ``hackrf_transfer -r`` capture, interleaved int8 IQ -> complex

Every hardware path degrades gracefully: if a ``hackrf_*`` binary (or numpy) is
absent the methods return empty/neutral results (or raise a clear
``NotImplementedError`` for the optional ``read_iq`` sample path) instead of
crashing, so the class stays importable and ABC-conformant on hosts with no
radio.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import tempfile
import time
from typing import Any, Optional

from tritium_lib.sdr.base import SDRDevice, SDRInfo, SweepResult, SweepPoint

from .device import HackRFDevice

log = logging.getLogger("hackrf.sdr_device")

# HackRF One hardware envelope (Great Scott Gadgets datasheet).
HACKRF_FREQ_MIN_HZ = 1_000_000          # 1 MHz
HACKRF_FREQ_MAX_HZ = 6_000_000_000      # 6 GHz
HACKRF_SAMPLE_RATE_MAX = 20_000_000     # 20 Msps


class HackRFSDRDevice(SDRDevice):
    """HackRF One as a ``tritium_lib.sdr.SDRDevice`` (data-plane interface).

    Composes a :class:`HackRFDevice` for detection/identity and drives the
    ``hackrf_sweep`` / ``hackrf_transfer`` CLI tools for the acquisition path.

    Usage::

        dev = HackRFSDRDevice()
        info = await dev.detect()                # SDRInfo
        if info.detected:
            result = await dev.sweep(88_000_000, 108_000_000, bin_width_hz=100_000)
            for pk in result.get_peaks(threshold_dbm=-40.0):
                print(pk.freq_hz, pk.power_dbm)
    """

    def __init__(self, control: Optional[HackRFDevice] = None):
        super().__init__()
        self._control = control if control is not None else HackRFDevice()
        self._tuned_freq: int = 0
        self._sample_rate: int = 0
        self._bandwidth: int = 0
        self._sweep_proc: Optional[asyncio.subprocess.Process] = None

    # -- Identity ----------------------------------------------------------

    async def detect(self) -> SDRInfo:
        """Detect the HackRF via ``hackrf_info`` and map it to an ``SDRInfo``.

        Reuses the control-plane :meth:`HackRFDevice.detect` (which returns a
        dict) and translates it into the ABC's ``SDRInfo`` dataclass. Never
        raises: a missing binary or absent device yields ``detected=False``.
        """
        raw = await self._control.detect()
        raw = raw if isinstance(raw, dict) else {}

        # HackRFDevice.detect returns {"connected": False, ...} on error, or a
        # parsed info dict (serial/board_name/firmware_version/...) on success.
        errored = raw.get("connected") is False
        detected = (not errored) and bool(raw.get("serial") or raw.get("board_name"))

        info = SDRInfo(
            detected=detected,
            name=raw.get("board_name", "HackRF One") if detected else "",
            serial=raw.get("serial", ""),
            firmware=raw.get("firmware_version", ""),
            api_version=raw.get("api_version", ""),
            hardware_id=str(raw.get("board_id", "")),
            hardware_rev=raw.get("hardware_revision", ""),
            freq_min_hz=HACKRF_FREQ_MIN_HZ if detected else 0,
            freq_max_hz=HACKRF_FREQ_MAX_HZ if detected else 0,
            sample_rate_max=HACKRF_SAMPLE_RATE_MAX if detected else 0,
            bandwidth_max=HACKRF_SAMPLE_RATE_MAX if detected else 0,
            has_tx=True,          # HackRF One is half-duplex TX-capable
            has_bias_tee=True,    # HackRF One has a bias tee on the antenna port
            error=raw.get("error", ""),
        )
        self._info = info
        return info

    # -- Broadband sweep ---------------------------------------------------

    async def sweep(
        self,
        freq_start_hz: int,
        freq_end_hz: int,
        bin_width_hz: int = 500_000,
    ) -> SweepResult:
        """Run one ``hackrf_sweep`` pass and parse it into a ``SweepResult``.

        Degrades gracefully: if ``hackrf_sweep`` is not on PATH (or the sweep
        fails/times out) an empty ``SweepResult`` carrying the requested band
        metadata is returned rather than raising.
        """
        t0 = time.time()
        if not shutil.which("hackrf_sweep"):
            log.warning("hackrf_sweep not found on PATH — returning empty sweep")
            return self._empty_sweep(freq_start_hz, freq_end_hz, bin_width_hz, t0)

        # hackrf_sweep takes integer-MHz bounds; widen to fully cover the band.
        start_mhz = max(0, int(freq_start_hz // 1_000_000))
        end_mhz = int(math.ceil(freq_end_hz / 1_000_000))
        if end_mhz <= start_mhz:
            end_mhz = start_mhz + 1

        cmd = [
            "hackrf_sweep",
            "-f", f"{start_mhz}:{end_mhz}",
            "-w", str(int(bin_width_hz)),
            "-1",  # single sweep
        ]

        try:
            self._sweep_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                self._sweep_proc.communicate(), timeout=30.0,
            )
            rc = self._sweep_proc.returncode
        except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
            log.error(f"hackrf_sweep failed: {e}")
            return self._empty_sweep(freq_start_hz, freq_end_hz, bin_width_hz, t0)
        finally:
            self._sweep_proc = None

        if rc != 0:
            log.warning(f"hackrf_sweep exit {rc}: {stderr.decode(errors='replace').strip()}")
            return self._empty_sweep(freq_start_hz, freq_end_hz, bin_width_hz, t0)

        points = self._parse_sweep_csv(
            stdout.decode(errors="replace"), freq_start_hz, freq_end_hz, t0,
        )
        return SweepResult(
            points=points,
            freq_start_hz=freq_start_hz,
            freq_end_hz=freq_end_hz,
            bin_width_hz=bin_width_hz,
            sweep_time_ms=(time.time() - t0) * 1000.0,
            timestamp=t0,
        )

    @staticmethod
    def _empty_sweep(start_hz: int, end_hz: int, bin_width_hz: int, t0: float) -> SweepResult:
        return SweepResult(
            points=[],
            freq_start_hz=start_hz,
            freq_end_hz=end_hz,
            bin_width_hz=bin_width_hz,
            sweep_time_ms=(time.time() - t0) * 1000.0,
            timestamp=t0,
        )

    @staticmethod
    def _parse_sweep_csv(
        text: str,
        freq_start_hz: int,
        freq_end_hz: int,
        timestamp: float,
    ) -> list[SweepPoint]:
        """Parse ``hackrf_sweep`` CSV output into ``SweepPoint``s (max per bin).

        Line format::

            date, time, hz_low, hz_high, hz_bin_width, num_samples, dB, dB, ...

        Each dB field after index 6 is the power of one bin starting at
        ``hz_low``. Points outside ``[freq_start_hz, freq_end_hz]`` (hackrf_sweep
        can overshoot the requested band) are dropped.
        """
        measurements: dict[int, float] = {}  # freq_hz -> max power_dbm
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                hz_low = int(float(parts[2].strip()))
                hz_bin_width = int(float(parts[4].strip()))
            except (ValueError, IndexError):
                continue
            for i, db_str in enumerate(parts[6:]):
                try:
                    db_val = float(db_str.strip())
                except ValueError:
                    continue
                freq_hz = hz_low + i * hz_bin_width
                if freq_hz < freq_start_hz or freq_hz > freq_end_hz:
                    continue
                prev = measurements.get(freq_hz)
                if prev is None or db_val > prev:
                    measurements[freq_hz] = db_val

        return [
            SweepPoint(freq_hz=f, power_dbm=round(p, 2), timestamp=timestamp)
            for f, p in sorted(measurements.items())
        ]

    # -- Tuning / streaming ------------------------------------------------

    async def tune(self, freq_hz: int, sample_rate: int = 2_000_000, bandwidth: int = 0):
        """Record the receive tuning for a subsequent :meth:`read_iq`.

        HackRF has no persistent "tune and hold" without an active transfer, so
        the parameters are captured here and applied when IQ is actually read.
        """
        self._tuned_freq = int(freq_hz)
        self._sample_rate = int(sample_rate)
        self._bandwidth = int(bandwidth)

    async def stop(self):
        """Terminate any in-flight sweep subprocess and clear tuning state."""
        proc = self._sweep_proc
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()
            except ProcessLookupError:
                pass
        self._sweep_proc = None
        self._tuned_freq = 0

    async def read_iq(self, n_samples: int) -> "Any":
        """Capture ``n_samples`` of complex baseband IQ via ``hackrf_transfer``.

        Requires a prior :meth:`tune`. Uses ``hackrf_transfer -r`` to capture
        interleaved signed-8-bit IQ, then scales to ``complex64`` in [-1, 1].

        Raises:
            RuntimeError: if the device has not been tuned.
            NotImplementedError: if ``hackrf_transfer`` or numpy is unavailable
                (no real sample path on this host).
        """
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover - numpy present in test env
            raise NotImplementedError("read_iq requires numpy") from e

        if n_samples <= 0:
            return np.zeros(0, dtype=np.complex64)
        if not self._tuned_freq:
            raise RuntimeError("read_iq() requires tune() to be called first")
        if not shutil.which("hackrf_transfer"):
            raise NotImplementedError("hackrf_transfer not found on PATH")

        tmp = tempfile.NamedTemporaryFile(prefix="hackrf_iq_", suffix=".c8", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cmd = [
                "hackrf_transfer",
                "-r", tmp_path,
                "-f", str(self._tuned_freq),
                "-s", str(self._sample_rate or 2_000_000),
                "-n", str(int(n_samples)),
            ]
            if self._bandwidth:
                cmd += ["-b", str(self._bandwidth)]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30.0)

            raw = np.fromfile(tmp_path, dtype=np.int8)
            if raw.size < 2:
                return np.zeros(0, dtype=np.complex64)
            raw = raw[: (raw.size // 2) * 2]
            i = raw[0::2].astype(np.float32) / 128.0
            q = raw[1::2].astype(np.float32) / 128.0
            return (i + 1j * q).astype(np.complex64)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # -- Convenience -------------------------------------------------------

    @property
    def tuned_frequency(self) -> int:
        """Last frequency passed to :meth:`tune` (0 if not tuned)."""
        return self._tuned_freq

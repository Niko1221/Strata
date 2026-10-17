"""Apple Silicon GPU telemetry; no root, bindings, or Metal workload in the server.

IORegistry provides whole-device utilization and driver allocations. Optional `macmon pipe`
provides IOReport/SMC power, temperature and clock readings (brew install macmon).
Both interfaces are OS/driver dependent: missing or stale readings stay None.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import plistlib
import select
import shutil
import subprocess
import time


def number(value, low=0, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value if value >= low and (high is None or value <= high) else None


def registry_gpu(nodes):
    """Find the AGX device, including on macOS versions that nest registry entries."""
    if not isinstance(nodes, list):
        return {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if isinstance(node.get("PerformanceStatistics"), dict):
            return node
        found = registry_gpu(node.get("IORegistryEntryChildren", []))
        if found:
            return found
    return {}


class AppleGpu:
    STALE_S = 5
    RETRY_S = 30

    def __init__(self, index=0):
        self.index = index
        self.proc = None
        self.buffer = b""
        self.latest = {}
        self.seen = self.started = self.retry_at = 0
        self.closed = False
        # GUI/launchd environments often omit Homebrew from PATH.
        self.macmon = shutil.which("macmon") or next(
            (p for p in ("/opt/homebrew/bin/macmon", "/usr/local/bin/macmon") if os.access(p, os.X_OK)), None)
        self.device = self._registry() if index == 0 else {}
        try:
            self.total = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], timeout=1))
        except (OSError, ValueError, subprocess.SubprocessError):
            self.total = None
        self.metadata = {"gpu_backend": "apple", "gpu_memory_kind": "unified",
                         "gpu_sensor_helper": bool(self.macmon)}
        if index == 0:
            atexit.register(self.close)

    def ok(self):
        return self.index == 0

    def name(self):
        model = self.device.get("model")
        if isinstance(model, bytes):
            model = model.rstrip(b"\0").decode(errors="replace")
        return model if isinstance(model, str) and model else "Apple GPU"

    @staticmethod
    def _registry():
        try:
            raw = subprocess.check_output(["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-a"], timeout=1)
            return registry_gpu(plistlib.loads(raw))
        except (OSError, ValueError, plistlib.InvalidFileException, subprocess.SubprocessError):
            return {}

    def _stop(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=.5)
            proc.stdout.close()
        self.latest, self.buffer, self.seen = {}, b"", 0

    def close(self):
        self.closed = True
        self._stop()
        atexit.unregister(self.close)

    def _sensors(self):
        """Drain one long-lived helper without blocking the one-second sampler or HTTP threads."""
        now = time.monotonic()
        if self.proc and (self.proc.poll() is not None or now - (self.seen or self.started) > self.STALE_S):
            self._stop()
            self.retry_at = now + self.RETRY_S
        if not self.proc and self.macmon and not self.closed and now >= self.retry_at:
            self.retry_at = now + self.RETRY_S
            try:
                self.proc = subprocess.Popen([self.macmon, "pipe", "--soc-info", "-i", "1000"],
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                self.started = now
            except OSError:
                pass
        if self.proc:
            # Bound the work even if a broken helper floods its pipe.
            for _ in range(16):
                if not select.select([self.proc.stdout], [], [], 0)[0]:
                    break
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                self.buffer += chunk
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            self.latest, self.seen = value, now
                    except (ValueError, UnicodeError):
                        pass
                if len(self.buffer) > 1048576:
                    self.buffer = b""
        return self.latest if self.seen and now - self.seen <= self.STALE_S else {}

    def read(self):
        stats = self._registry().get("PerformanceStatistics", {})
        try:
            sensors = self._sensors()
        except (OSError, ValueError):
            self._stop()
            self.retry_at = time.monotonic() + self.RETRY_S
            sensors = {}
        temp = sensors.get("temp") if isinstance(sensors.get("temp"), dict) else {}
        soc = sensors.get("soc") if isinstance(sensors.get("soc"), dict) else {}
        virtual = soc.get("vm", False)
        active = number(sensors.get("gpu_active_ratio"), high=1)
        # Allocated GPU driver buffers, across ALL processes. Not system RAM in use,
        # per-process Metal allocation, dedicated VRAM, or free memory for model sizing.
        return {"util": active * 100 if active is not None else number(stats.get("Device Utilization %"), high=100),
                "mem_used": number(stats.get("Alloc system memory")), "mem_total": self.total,
                "temp": number(temp.get("gpu_temp_avg"), low=1, high=150) if not virtual else None,
                "power": number(sensors.get("gpu_power")) if not virtual else None,
                "freq_mhz": number(sensors.get("gpu_freq_mhz"), low=1) if not virtual else None}

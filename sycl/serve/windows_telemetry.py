"""Read-only Intel Windows telemetry, isolated from the server in an IGCL helper."""
from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


class _Rates:
    def __init__(self):
        self.previous = None

    def convert(self, sample):
        result = {k: sample[k] for k in ("temp", "vram_temp", "mem_used", "mem_total", "pcie_gen",
                                        "pcie_gen_max", "pcie_width") if _number(sample.get(k))}
        used, total = result.get("mem_used"), result.get("mem_total")
        if total is None or total <= 0 or used is None or used > total:
            result.pop("mem_used", None)
            result.pop("mem_total", None)
        result.update(backend="igcl", power_scope="gpu", pci_bdf=sample.get("pci_bdf"))
        previous, self.previous = self.previous, sample
        if previous and _number(sample.get("timestamp")) and _number(previous.get("timestamp")):
            dt = sample["timestamp"] - previous["timestamp"]
            if 0 < dt <= 5:
                for counter, key, scale in (("energy", "power", 1), ("activity", "util", 100)):
                    current, old = sample.get(counter), previous.get(counter)
                    if _number(current) and _number(old) and current >= old:
                        value = (current - old) / dt * scale
                        if math.isfinite(value):
                            result[key] = min(value, 100) if key == "util" else value
        return result


class _WindowsIntelGpu:
    STALE_SECONDS = 5

    def __init__(self, index=0):
        self._process = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._latest = {}
        self._received = 0
        self._name = None
        self._thread = None
        if sys.platform != "win32":
            return
        helper = os.environ.get("STRATA_INTEL_TELEMETRY_EXE") or str(
            Path(__file__).resolve().parents[2] / "build-sycl-windows" / "strata-intel-telemetry.exe")
        command = [helper, "--index", str(index), "--parent", str(os.getpid())]
        if os.environ.get("STRATA_INTEL_PCI_BDF"):
            command += ["--pci-bdf", os.environ["STRATA_INTEL_PCI_BDF"]]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                             encoding="utf-8", errors="replace",
                                             creationflags=subprocess.CREATE_NO_WINDOW)
            self._thread = threading.Thread(target=self._consume, name="intel-telemetry", daemon=True)
            self._thread.start()
            self._ready.wait(3)
            if not self._name:
                self.close()
            else:
                atexit.register(self.close)
        except OSError:
            self.close()

    def _consume(self):
        rates = _Rates()
        try:
            for line in self._process.stdout:
                try:
                    sample = json.loads(line)
                    if not isinstance(sample, dict) or not isinstance(sample.get("name"), str):
                        continue
                    result = rates.convert(sample)
                except (ValueError, TypeError, OverflowError):
                    continue
                with self._lock:
                    self._name = sample["name"]
                    self._latest = result
                    self._received = time.monotonic()
                self._ready.set()
        except (OSError, ValueError):
            pass
        finally:
            self._ready.set()

    def ok(self):
        return self._name is not None

    def name(self):
        return self._name

    def read(self):
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or time.monotonic() - self._received > self.STALE_SECONDS:
                return {}
            return dict(self._latest)

    def close(self):
        atexit.unregister(self.close)
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
            if self._thread and self._thread is not threading.current_thread():
                self._thread.join(timeout=1)
            if process.poll() is not None and process.stdout:
                process.stdout.close()
        self._process = None

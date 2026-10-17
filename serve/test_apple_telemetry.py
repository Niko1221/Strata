"""macOS metrics, missing sensors, helper recovery and sampler lifetime; no GPU needed."""
import json
import os
import plistlib
import subprocess
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import telemetry
from serve.apple_telemetry import AppleGpu, number, registry_gpu


DEVICE = {"model": b"Apple M3 Pro\0", "PerformanceStatistics": {
    "Device Utilization %": 37, "Alloc system memory": 12 << 30}}
SENSORS = {"temp": {"gpu_temp_avg": 52.5}, "gpu_power": 7.25,
           "gpu_freq_mhz": 924, "gpu_active_ratio": .625}
REGISTRY_READER = AppleGpu._registry


class AppleTelemetry(unittest.TestCase):
    def setUp(self):
        mock.patch.dict(os.environ, {"STRATA_MACOS_TELEMETRY": "1"}).start()
        self.registry = mock.patch.object(AppleGpu, "_registry", return_value=DEVICE).start()
        mock.patch("serve.apple_telemetry.subprocess.check_output", return_value=b"38654705664\n").start()
        mock.patch("serve.apple_telemetry.shutil.which", return_value="/test/macmon").start()
        self.addCleanup(mock.patch.stopall)
        self.gpu = AppleGpu()
        self.addCleanup(self.gpu.close)

    def test_readings_are_device_wide_and_unified_not_free_vram(self):
        with mock.patch.object(self.gpu, "_sensors", return_value=SENSORS):
            self.assertEqual(self.gpu.read(), {"util": 62.5, "mem_used": 12 << 30, "mem_total": 36 << 30,
                                              "temp": 52.5, "power": 7.25, "freq_mhz": 924})
        self.assertEqual(self.gpu.name(), "Apple M3 Pro")
        self.assertEqual(self.gpu.metadata["gpu_memory_kind"], "unified")
        with mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry.platform, "machine", return_value="arm64"), \
                mock.patch.object(telemetry, "_Metal") as metal, \
                mock.patch.object(telemetry, "gpu_reader") as reader:
            metal.return_value.read.return_value = {"mem_total": 32 << 30, "mem_used": 2 << 30}
            self.assertEqual(telemetry.free_vram_mib(), 30 << 10)
            reader.assert_not_called()

    def test_missing_invalid_and_zero_readings(self):
        with mock.patch.object(self.gpu, "_sensors", return_value={}):
            out = self.gpu.read()
            self.assertEqual(out["util"], 37)
            self.assertTrue(all(out[k] is None for k in ("temp", "power", "freq_mhz")))
        for invalid in (float("nan"), float("inf"), -1, "42", True):
            self.assertIsNone(number(invalid))
        with mock.patch.object(self.gpu, "_sensors", return_value={"gpu_power": 0, "temp": 42, "soc": []}):
            self.assertEqual(self.gpu.read()["power"], 0)
        with mock.patch.object(self.gpu, "_sensors", return_value={**SENSORS, "soc": {"vm": True}}):
            self.assertTrue(all(self.gpu.read()[k] is None for k in ("temp", "power", "freq_mhz")))
        self.registry.return_value = {}
        with mock.patch.object(self.gpu, "_sensors", return_value={}):
            self.assertIsNone(self.gpu.read()["mem_used"])
            self.assertIsNone(self.gpu.read()["util"])

    def test_registry_parsing_and_failure(self):
        self.assertEqual(registry_gpu([{"IORegistryEntryChildren": [DEVICE]}]), DEVICE)
        self.assertEqual(registry_gpu([{}, "bad"]), {})
        with mock.patch("serve.apple_telemetry.subprocess.check_output", return_value=plistlib.dumps([DEVICE])):
            self.assertEqual(REGISTRY_READER(), DEVICE)
        for failure in (OSError("denied"), subprocess.TimeoutExpired("ioreg", 1)):
            with mock.patch("serve.apple_telemetry.subprocess.check_output", side_effect=failure):
                self.assertEqual(REGISTRY_READER(), {})
        with mock.patch("serve.apple_telemetry.subprocess.check_output", return_value=b"bad plist"):
            self.assertEqual(REGISTRY_READER(), {})

    def test_stream_fragments_bad_json_and_timeout_restart(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.stdout.fileno.return_value = 123
        self.gpu.proc, self.gpu.started = proc, 100
        data = json.dumps(SENSORS).encode()
        with mock.patch("serve.apple_telemetry.time.monotonic", return_value=102), \
                mock.patch("serve.apple_telemetry.select.select", side_effect=[([proc.stdout], [], []), ([], [], [])]), \
                mock.patch("serve.apple_telemetry.os.read", return_value=b"bad json\n" + data[:12]):
            self.assertEqual(self.gpu._sensors(), {})
        with mock.patch("serve.apple_telemetry.time.monotonic", return_value=103), \
                mock.patch("serve.apple_telemetry.select.select", side_effect=[([proc.stdout], [], []), ([], [], [])]), \
                mock.patch("serve.apple_telemetry.os.read", return_value=data[12:] + b"\n"):
            self.assertEqual(self.gpu._sensors(), SENSORS)
        with mock.patch("serve.apple_telemetry.time.monotonic", return_value=110):
            self.assertEqual(self.gpu._sensors(), {})
            self.assertIsNone(self.gpu.proc)
            proc.terminate.assert_called_once()
            proc.wait.assert_called_once()
        with mock.patch("serve.apple_telemetry.time.monotonic", return_value=111), \
                mock.patch("serve.apple_telemetry.subprocess.Popen") as spawn:
            self.gpu._sensors()
            spawn.assert_not_called()
        with mock.patch("serve.apple_telemetry.time.monotonic", return_value=141), \
                mock.patch("serve.apple_telemetry.subprocess.Popen", side_effect=OSError("denied")):
            self.assertEqual(self.gpu._sensors(), {})

    def test_helper_absent_and_cleanup(self):
        self.gpu.macmon = None
        with mock.patch("serve.apple_telemetry.subprocess.Popen") as spawn:
            self.assertEqual(self.gpu.read()["util"], 37)
            spawn.assert_not_called()
        self.gpu.close()
        self.assertTrue(self.gpu.closed)

    def test_sampler_metadata_and_close(self):
        with mock.patch.object(telemetry, "gpu_reader", return_value=self.gpu), \
                mock.patch.object(self.gpu, "_sensors", return_value=SENSORS):
            sampler = telemetry.Telemetry()
            try:
                sampler.thread.join(timeout=.05)
                snap = sampler.snapshot()
                self.assertEqual(snap["static"]["gpu_backend"], "apple")
                self.assertEqual(snap["now"]["gpu_freq_mhz"], 924)
                self.assertEqual(snap["history"]["gpu_freq_mhz"], [924])
            finally:
                sampler.close()
            self.assertFalse(sampler.thread.is_alive())
            self.assertTrue(self.gpu.closed)

    def test_http_metrics_and_server_cleanup(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        with mock.patch.object(telemetry, "gpu_reader", return_value=self.gpu), \
                mock.patch.object(self.gpu, "_sensors", return_value=SENSORS):
            httpd = serve(svc, port=0)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}/metrics", timeout=5) as reply:
                    metrics = json.load(reply)
                self.assertEqual(metrics["hardware"]["gpu_power"], 7.25)
                self.assertEqual(metrics["hardware_static"]["gpu_memory_kind"], "unified")
                self.assertEqual(metrics["history"]["gpu_freq_mhz"], [924])
            finally:
                httpd.shutdown()
                httpd.server_close()
            self.assertFalse(svc.telemetry.thread.is_alive())
            self.assertTrue(self.gpu.closed)

    def test_default_macos_reader_is_unchanged(self):
        with mock.patch.dict(os.environ, {"STRATA_MACOS_TELEMETRY": "0"}), \
                mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry, "_Metal") as metal, \
                mock.patch("serve.apple_telemetry.AppleGpu") as apple:
            self.assertIs(telemetry.gpu_reader(0), metal.return_value)
            apple.assert_not_called()
            metal.assert_called_once_with(0)

    def test_platform_routing_preserves_pc_readers(self):
        with mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry.platform, "machine", return_value="arm64"), \
                mock.patch("serve.apple_telemetry.AppleGpu", return_value=self.gpu) as apple:
            self.assertIs(telemetry.gpu_reader(), self.gpu)
            apple.assert_called_once_with(0)
            with mock.patch.object(telemetry, "_Amd") as amd:
                telemetry.gpu_reader(2, amd=True)
                amd.assert_called_once_with(2)
        with mock.patch.object(telemetry.sys, "platform", "linux"), \
                mock.patch.object(telemetry, "_Nvml") as nvml:
            telemetry.gpu_reader(1)
            nvml.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()

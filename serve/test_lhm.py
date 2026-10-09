"""LibreHardwareMonitor CPU sensor reader tests; these do not depend on a live sensor service."""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from serve import telemetry


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class LhmCpuTests(unittest.TestCase):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = self.server.body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    def server(self, body):
        httpd = HTTPServer(("127.0.0.1", 0), self.Handler)
        httpd.body = body
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, thread

    @staticmethod
    def tree(temp="46,0 °C", power="46.0 W"):
        return {"Children": [{"Text": "CPU", "Children": [
            {"SensorId": "/intelcpu/0/temperature/0", "Text": "Core Max", "Type": "Temperature", "Value": "99 °C"},
            {"SensorId": "/intelcpu/0/temperature/10", "Text": "CPU Package", "Type": "Temperature", "Value": temp},
            {"SensorId": "/intelcpu/0/power/0", "Text": "CPU Package", "Type": "Power", "Value": power},
            {"SensorId": "/intelcpu/0/power/1", "Text": "CPU Cores", "Type": "Power", "Value": "1 W"},
        ]}]}

    def test_fetches_tree_and_parses_both_decimal_marks(self):
        httpd, thread = self.server(json.dumps(self.tree()).encode())
        clock = FakeClock()
        timeouts = []
        def fetch(url, timeout):
            timeouts.append(timeout)
            return telemetry._fetch_lhm(url, timeout)
        try:
            reader = telemetry._LhmCpu(f"http://127.0.0.1:{httpd.server_port}/data.json", fetch, clock)
            self.assertEqual(reader.read(), {"cpu_temp": 46.0, "cpu_power": 46.0})
            clock.now += 2
            httpd.body = json.dumps(self.tree("46.0 °C", "85,7 W")).encode()
            self.assertEqual(reader.read(), {"cpu_temp": 46.0, "cpu_power": 85.7})
            self.assertEqual(timeouts, [0.5, 0.5])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_sensors_are_picked_by_name_for_intel_and_amd(self):
        pick = telemetry._LhmCpu._sensors
        def sensor(sid, text, value):
            return {"SensorId": sid, "Text": text, "Value": value}
        # An Intel CPU whose package temperature has another index than the i9-9900KF's /temperature/10.
        intel = {"Children": [sensor("/intelcpu/0/temperature/3", "CPU Package", "51 °C"),
                              sensor("/intelcpu/0/power/0", "CPU Package", "30 W")]}
        self.assertEqual(pick(intel), {"cpu_temp": 51.0, "cpu_power": 30.0})
        # AMD Zen: LHM names the die temperature "Core (Tctl/Tdie)" and the power "Package".
        amd = {"Children": [sensor("/amdcpu/0/temperature/2", "CCD1 (Tdie)", "60 °C"),
                            sensor("/amdcpu/0/temperature/0", "Core (Tctl/Tdie)", "64,5 °C"),
                            sensor("/amdcpu/0/power/0", "Package", "88,1 W")]}
        self.assertEqual(pick(amd), {"cpu_temp": 64.5, "cpu_power": 88.1})
        # No package sensor: the hottest core. A motherboard "CPU" reading or a second CPU is never used.
        other = {"Children": [sensor("/lpc/nct6798d/0/temperature/1", "CPU Package", "10 °C"),
                              sensor("/intelcpu/1/temperature/9", "CPU Package", "11 °C"),
                              sensor("/intelcpu/0/temperature/0", "Core Max", "70 °C")]}
        self.assertEqual(pick(other), {"cpu_temp": 70.0, "cpu_power": None})

    def test_missing_sensor_is_none(self):
        body = json.dumps({"Children": [{"SensorId": "/intelcpu/0/power/0", "Text": "CPU Package", "Value": "38,4 W"}]}).encode()
        httpd, thread = self.server(body)
        try:
            reader = telemetry._LhmCpu(f"http://127.0.0.1:{httpd.server_port}/data.json", clock=FakeClock())
            self.assertEqual(reader.read(), {"cpu_temp": None, "cpu_power": 38.4})
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_bad_json_returns_none(self):
        httpd, thread = self.server(b"{")
        try:
            reader = telemetry._LhmCpu(f"http://127.0.0.1:{httpd.server_port}/data.json", clock=FakeClock())
            self.assertEqual(reader.read(), {"cpu_temp": None, "cpu_power": None})
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_closed_port_backs_off_for_ten_samples(self):
        httpd = HTTPServer(("127.0.0.1", 0), self.Handler)
        port = httpd.server_port
        httpd.server_close()
        attempts = []
        clock = FakeClock()
        def fetch(url, timeout):
            attempts.append((url, timeout))
            return telemetry._fetch_lhm(url, timeout)
        reader = telemetry._LhmCpu(f"http://127.0.0.1:{port}/data.json", fetch, clock)
        for _ in range(10):
            self.assertEqual(reader.read(), {"cpu_temp": None, "cpu_power": None})
            clock.now += 1
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0][1], 0.5)

    def test_telemetry_records_cpu_and_measured_power_history(self):
        class Gpu:
            def ok(self):
                return True

            def name(self):
                return "Fake GPU"

            def read(self):
                return {"power": 30.0}

        from serve import telemetry as telemetry_module
        with mock.patch.dict(os.environ, {"STRATA_LHM_URL": ""}), \
                mock.patch.object(telemetry_module, "gpu_reader", return_value=Gpu()), \
                mock.patch.object(telemetry_module._Nvml, "count", return_value=1), \
                mock.patch.object(telemetry_module.threading.Thread, "start"):
            instance = telemetry_module.Telemetry()
        instance.lhm = telemetry_module._LhmCpu(
            "http://unused", fetch=lambda _url, _timeout: self.tree("55,5 °C", "85,7 W"), clock=FakeClock())
        sample = instance.sample()
        instance.record(sample)
        self.assertEqual(sample["measured_power"], 115.7)
        history = instance.snapshot()["history"]
        self.assertEqual(history["cpu_temp"], [55.5])
        self.assertEqual(history["cpu_power"], [85.7])
        self.assertEqual(history["measured_power"], [115.7])

    def test_empty_url_disables_reader(self):
        reader = telemetry._LhmCpu(url="", fetch=mock.Mock(side_effect=AssertionError("fetch called")))
        self.assertFalse(reader.enabled)
        self.assertEqual(reader.read(), {})


class HwmonCpuTempTests(unittest.TestCase):
    """Linux CPU temperature from fake /sys/class/hwmon folders."""

    def hwmon(self, *chips):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        for i, (name, temps) in enumerate(chips):
            folder = os.path.join(root, f"hwmon{i}")
            os.makedirs(folder)
            with open(os.path.join(folder, "name"), "w") as f:
                f.write(name + "\n")
            for n, (label, millic) in enumerate(temps, 1):
                if label is not None:
                    with open(os.path.join(folder, f"temp{n}_label"), "w") as f:
                        f.write(label + "\n")
                with open(os.path.join(folder, f"temp{n}_input"), "w") as f:
                    f.write(f"{millic}\n")
        return root

    def test_amd_k10temp_tctl_among_other_chips(self):
        # the sensors output posted on #1114: amdgpu and nvme come first, k10temp's Tctl is the CPU package
        root = self.hwmon(("amdgpu", [("edge", 45000), ("junction", 64000)]), ("nvme", [("Composite", 39850)]),
                          ("k10temp", [("Tccd1", 33375), ("Tctl", 45375)]))
        self.assertEqual(telemetry._HwmonCpuTemp(root).read(), {"cpu_temp": 45.375})

    def test_intel_coretemp_package(self):
        root = self.hwmon(("coretemp", [("Core 0", 41000), ("Package id 0", 47000)]))
        self.assertEqual(telemetry._HwmonCpuTemp(root).read(), {"cpu_temp": 47.0})

    def test_unlabelled_k10temp_uses_temp1(self):
        root = self.hwmon(("k10temp", [(None, 52125)]))
        self.assertEqual(telemetry._HwmonCpuTemp(root).read(), {"cpu_temp": 52.125})

    def test_no_cpu_chip_disables_reader(self):
        root = self.hwmon(("amdgpu", [("edge", 45000)]), ("acpitz", [(None, 27800)]))
        reader = telemetry._HwmonCpuTemp(root)
        self.assertFalse(reader.enabled)
        self.assertEqual(reader.read(), {})
        self.assertFalse(telemetry._HwmonCpuTemp(os.path.join(root, "missing")).enabled)

    def test_telemetry_uses_hwmon_when_lhm_has_none(self):
        root = self.hwmon(("k10temp", [("Tctl", 45375)]))
        with mock.patch.dict(os.environ, {"STRATA_LHM_URL": ""}), \
                mock.patch.object(telemetry._Nvml, "count", return_value=0), \
                mock.patch.object(telemetry.threading.Thread, "start"):
            instance = telemetry.Telemetry()
        instance.hwmon_cpu = telemetry._HwmonCpuTemp(root)
        sample = instance.sample()
        instance.record(sample)
        self.assertEqual(sample["cpu_temp"], 45.375)
        history = instance.snapshot()["history"]
        self.assertEqual(history["cpu_temp"], [45.38])
        self.assertNotIn("cpu_power", history)


if __name__ == "__main__":
    unittest.main()
